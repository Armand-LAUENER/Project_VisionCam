import os

# OpenBLAS ne lit cette variable qu'au chargement de numpy : elle doit précéder
# tous les imports. Ses threads se disputaient le CPU avec torch et
# onnxruntime pour des matrices trop petites pour en profiter (distances
# cosinus de DeepSORT) : sur MOT17-04, l'association passait de 2 à 24 ms et le
# prétraitement des crops de 3 à 23 ms. setdefault laisse la main à un réglage
# explicite de l'environnement.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import hmac
import json
import logging
import queue
import secrets
import signal
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash

import config
from core.cameras import parse_cameras
from core.db_writer import DbWriter
from core.endurance import EnduranceLog
from core.event_log import EventLog, ForwardClock, TrackEvents
from core.events import EventBus, StageTimer, UnknownWatcher
from core.face_body_tracker import FaceBodyTracker
from core.face_recognition import FaceRecognizer
from core.history import history_names, presence_history
from core.pose_estimation import PoseEstimator
from core.pose_from_keypoints import KeypointPoseEstimator
from core.presence_log import FaceConfirmedNames, PresenceLog
from core.system_probe import Nvml, WindowsCounters, process_stats, system_stats
from core.video_source import FileSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Clé en mémoire par défaut : _persist_secret_key() la remplace au lancement
# réel, sans que l'import (les tests) n'écrive quoi que ce soit sur disque.
app.secret_key = config.SECRET_KEY or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=config.SESSION_DAYS),
)

# ══════════════════════════════════════════
# INITIALISATION DES MODULES
# ══════════════════════════════════════════
logger.info("Chargement des modules...")
logger.info("Source vidéo : %s", "WEBCAM" if config.USE_LOCAL_CAM else config.REMOTE_SOURCE)

face_recognizer = FaceRecognizer(
    config.KNOWN_FACES_DIR,
    threshold=config.RECOGNITION_THRESHOLD,
    cache_path=config.EMBEDDINGS_CACHE_PATH,
)
tracker = FaceBodyTracker(face_recognizer)
if config.POSE_SOURCE == "yolo":
    pose_estimator = KeypointPoseEstimator()
elif config.POSE_SOURCE == "mediapipe":
    pose_estimator = PoseEstimator()
else:
    raise ValueError(
        f"POSE_SOURCE invalide : {config.POSE_SOURCE!r} — attendu 'mediapipe' ou 'yolo'."
    )
logger.info("Source d'orientation : %s", config.POSE_SOURCE)

logger.info("Visages connus : %s", list(face_recognizer.known_names))
logger.info("Modules chargés")


# ══════════════════════════════════════════
# ÉTAT GLOBAL (thread-safe)
# ══════════════════════════════════════════
class CameraState:
    """État propre à une caméra : image, personnes suivies, FPS, flux MJPEG."""

    def __init__(self):
        self.lock = threading.Lock()
        # Images jetées faute de place dans la file (pipeline en retard), et
        # sautées par une source fichier en realtime : compteurs du journal d'endurance.
        self.frames_dropped = 0
        self.source_skipped = 0
        # Réveille les flux MJPEG à chaque nouvelle frame (partage self.lock).
        self.frame_ready = threading.Condition(self.lock)
        self.frame_seq = 0
        self.stream_clients = 0
        self.current_frame = None
        self.currently_present = []
        self.fps = 0.0
        # Dernière frame BRUTE (sans overlay) et TrackedPerson associés.
        # Alimentent /bench/pose, qui a besoin des crops et des keypoints,
        # pas du JPEG annoté envoyé au navigateur.
        self.bench_frame = None
        self.bench_persons: list = []


class AppState(CameraState):
    """État global de l'application ; c'est aussi l'état de la première caméra."""

    def __init__(self):
        super().__init__()
        self.total_known = len(face_recognizer.known_names)
        self.running = True


state = AppState()
# Toutes les écritures de presence.db passent par un seul thread : la boucle
# vidéo ne les attend jamais (un fsync bloquait des images jusqu'à 14 s).
db_writer = DbWriter(config.PRESENCE_DB_PATH, background=True)
presence_log = PresenceLog(config.PRESENCE_DB_PATH, gap=config.PRESENCE_LOG_GAP_S,
                           retention=config.PRESENCE_RETENTION_DAYS * 86400 or None,
                           writer=db_writer)
# Journal d'événements bruts (roadmap 2.1), même base que l'historique. Une
# piste est finie après PRESENCE_TIMEOUT sans être visible.
event_clock = ForwardClock()
event_log = EventLog(config.PRESENCE_DB_PATH, run_id=uuid.uuid4().hex[:12], clock=event_clock,
                     retention=config.PRESENCE_RETENTION_DAYS * 86400 or None,
                     writer=db_writer)
track_events = TrackEvents(config.CAMERA_ID, config.PRESENCE_TIMEOUT,
                           face_max_age=config.PRESENCE_FACE_MAX_AGE_S,
                           face_interval=config.FACE_EVENT_INTERVAL_S)
events = EventBus()
timings = StageTimer()
STARTED_AT = time.time()


def _publish(event_type, **fields):
    """Diffuse un événement aux pages ouvertes (/api/events)."""
    events.publish({'type': event_type, 'time': datetime.now().isoformat(timespec='seconds'),
                    **fields})

# Queue inter-thread caméra → AI. Taille 2 : on garde toujours la frame la plus fraîche.
_frame_queue: queue.Queue = queue.Queue(maxsize=2)


class Camera:
    """Une caméra (roadmap 2.2) : sa source, sa file d'images, son tracker, son état.

    La reconnaissance faciale, l'historique de présence et le journal
    d'événements sont partagés : une personne vue par n'importe quelle caméra
    est présente. L'anti-clonage reste propre à chaque caméra (cf. roadmap 2.3).
    """

    def __init__(self, spec, camera_state, camera_tracker, frame_queue,
                 camera_track_events=None):
        self.id = spec.id
        self.spec = spec
        self.state = camera_state
        self.tracker = camera_tracker
        self.queue = frame_queue
        self.track_events = camera_track_events or TrackEvents(
            spec.id, config.PRESENCE_TIMEOUT, face_max_age=config.PRESENCE_FACE_MAX_AGE_S,
            face_interval=config.FACE_EVENT_INTERVAL_S)
        self.face_confirmed = FaceConfirmedNames(config.PRESENCE_FACE_MAX_AGE_S)
        self.unknown_watcher = UnknownWatcher(config.UNKNOWN_ALERT_S)
        # FramePipeline de la caméra, créé par son thread de traitement.
        self.pipeline = None

    @property
    def is_file(self) -> bool:
        return self.spec.is_file if self.spec.source is not None else bool(config.VIDEO_FILE)

    @property
    def source(self):
        """Caméra unique : lue dans la configuration à chaque ouverture."""
        if self.spec.source is not None:
            return self.spec.source
        return config.VIDEO_FILE or config.CAMERA_SOURCE


# La première caméra reprend l'état, le tracker et la file d'avant : l'usage à
# une caméra ne change pas. Les suivantes ont les leurs (un modèle YOLO chacune).
cameras: list[Camera] = []
for _index, _spec in enumerate(parse_cameras(config.CAMERAS, config.CAMERA_ID)):
    if _index == 0:
        cameras.append(Camera(_spec, state, tracker, _frame_queue, track_events))
    else:
        cameras.append(Camera(_spec, CameraState(), FaceBodyTracker(face_recognizer),
                              queue.Queue(maxsize=2)))
cameras_by_id = {camera.id: camera for camera in cameras}
face_confirmed = cameras[0].face_confirmed
unknown_watcher = cameras[0].unknown_watcher
if len(cameras) > 1:
    logger.info("Caméras : %s", ", ".join(f"{c.id}={c.source}" for c in cameras))


# ══════════════════════════════════════════
# HELPERS PIPELINE
# ══════════════════════════════════════════
def _estimate_pose_safe(frame, person):
    """
    Calcule l'orientation d'une personne, avec log explicite en cas d'échec.

    Deux sources selon config.POSE_SOURCE : les keypoints YOLO déjà produits
    sur GPU, ou une inférence Mediapipe sur le crop du corps.
    """
    try:
        if config.POSE_SOURCE == "yolo":
            return pose_estimator.estimate(person.pose_kps)

        x1, y1, x2, y2 = person.body_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return pose_estimator.estimate(crop)
    except Exception as e:
        logger.warning("Échec pose sur track #%s : %s: %s",
                       person.track_id, type(e).__name__, e)
        return None


def _draw_person(display_frame, person, pose, frame_count):
    """
    Dessine bbox + label + pose pour une TrackedPerson.

    Modes d'affichage selon l'état d'identification :
      - Vert  (0, 245, 160) : visage reconnu récemment (< FACE_FRESHNESS_FRAMES)
      - Orange(0, 165, 255) : identité connue mais visage perdu — tracking par corps
      - Bleu  (100, 100, 255): personne inconnue
    """
    x1, y1, x2, y2 = person.body_bbox

    frames_since_face = frame_count - person.last_face_frame

    if person.name == 'Inconnu':
        color = (100, 100, 255)
        mode_tag = None
    elif frames_since_face <= config.FACE_FRESHNESS_FRAMES:
        color = (0, 245, 160)
        mode_tag = None
    else:
        color = (0, 165, 255)
        mode_tag = "BODY"

    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

    label = f"{person.name} #{person.track_id}"
    if person.confidence > 0:
        label += f" ({person.confidence * 100:.0f}%)"
    if mode_tag:
        label += f" [{mode_tag}]"

    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
    cv2.putText(display_frame, label, (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if pose:
        cv2.putText(display_frame, f"Pose: {pose}",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _open_camera(camera=None):
    """
    Ouvre la caméra avec un timeout court pour ne pas bloquer le thread.
    Retourne un objet VideoCapture ouvert, ou None si l'ouverture échoue.
    Une source fichier (VIDEO_FILE, ou un chemin dans CAMERAS) remplace la caméra,
    avec la même interface.
    """
    camera = camera or cameras[0]
    if camera.is_file:
        source = FileSource(camera.source, mode=config.VIDEO_MODE, loop=config.VIDEO_LOOP)
        if not source.isOpened():
            logger.error("Vidéo illisible : %s", camera.source)
            return None
        logger.info("[%s] Vidéo à la place de la caméra : %s (%s, %.0f i/s%s)", camera.id,
                    camera.source, config.VIDEO_MODE, source.fps,
                    ", en boucle" if config.VIDEO_LOOP else "")
        return source

    cap = cv2.VideoCapture(camera.source)

    # Timeout de connexion et de lecture (ms). Sans ça, VideoCapture sur HTTP
    # peut bloquer jusqu'à 30s sans aucun log — invisible pour l'utilisateur.
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.CAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.CAM_READ_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)

    if not cap.isOpened():
        cap.release()
        return None
    logger.info("[%s] Caméra ouverte : %dx%d", camera.id, int(cap.get(3)), int(cap.get(4)))
    return cap


def _reconnect_camera(cap, camera=None):
    """
    Tente de rouvrir la caméra avec exponential backoff (1s → 30s max).
    Bloque jusqu'à la reconnexion ou l'arrêt de l'application.
    Retourne un nouveau VideoCapture ouvert, ou None si state.running devient False.
    """
    if cap is not None:
        cap.release()

    delay = 1.0
    attempt = 0
    while state.running:
        attempt += 1
        logger.warning("Reconnexion tentative #%d dans %.0fs...", attempt, delay)
        time.sleep(delay)

        new_cap = _open_camera(camera)
        if new_cap is not None:
            logger.info("Reconnexion réussie après %d tentative(s)", attempt)
            return new_cap

        delay = min(delay * 2, 30.0)

    return None


# ══════════════════════════════════════════
# THREAD A — LECTURE CAMÉRA
# ══════════════════════════════════════════
def camera_loop(camera=None):
    """
    Lit les frames depuis la caméra et les pousse dans _frame_queue, avec
    leur heure de réception (perf_counter) pour mesurer la latence.
    Entièrement découplé du pipeline AI : cap.read() ne bloque jamais
    le calcul YOLO/InsightFace. La queue conserve toujours la frame
    la plus récente (drop oldest si pleine).
    """
    camera = camera or cameras[0]
    camera_state, frame_queue = camera.state, camera.queue
    logger.info("[%s] Thread caméra démarré.", camera.id)
    cap = _open_camera(camera)
    if cap is None and camera.is_file:
        return   # un fichier illisible ne se répare pas en réessayant
    if cap is None:
        logger.error("[%s] Impossible d'ouvrir la caméra au démarrage : %s",
                     camera.id, camera.source)
        cap = _reconnect_camera(None, camera)
        if cap is None:
            return

    consecutive_failures = 0

    while state.running:
        ret, frame = cap.read()
        if not ret and isinstance(cap, FileSource):
            logger.info("[%s] Fin de la vidéo : %s", camera.id, camera.source)
            break
        if not ret:
            consecutive_failures += 1
            if consecutive_failures == 1:
                logger.warning("Frame perdue...")
            elif consecutive_failures >= config.CAM_MAX_FAILURES:
                logger.error("%d échecs consécutifs — déconnexion détectée.", consecutive_failures)
                cap = _reconnect_camera(cap, camera)
                if cap is None:
                    break
                consecutive_failures = 0
            else:
                time.sleep(0.05)
            continue

        consecutive_failures = 0
        item = (time.perf_counter(), frame)
        if isinstance(cap, FileSource):
            camera_state.source_skipped = cap.skipped

        # every_frame : aucune image perdue, la lecture attend le pipeline.
        if isinstance(cap, FileSource) and cap.mode == "every_frame":
            _put_waiting(frame_queue, item)
            continue

        # Toujours garder la frame la plus fraîche : drop oldest si queue pleine.
        try:
            frame_queue.put_nowait(item)
        except queue.Full:
            camera_state.frames_dropped += 1
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(item)
            except queue.Full:
                pass

    cap.release()
    logger.info("[%s] Thread caméra arrêté.", camera.id)


def _put_waiting(frame_queue, item) -> None:
    """Met l'image en file en attendant qu'une place se libère (mode every_frame)."""
    while state.running:
        try:
            frame_queue.put(item, timeout=0.5)
            return
        except queue.Full:
            continue


# ══════════════════════════════════════════
# THREAD B — PIPELINE AI
# ══════════════════════════════════════════
@dataclass
class FrameResult:
    """Ce que le pipeline d'une caméra produit pour une image.

    Frontière entre le traitement d'image (tracker, pose, dessin) et ce qui en
    découle pour toute l'application (historique, journal d'événements,
    alertes, flux web) : la première partie pourra tourner dans un autre
    processus que la seconde.
    """
    frame: np.ndarray            # image brute, pour /api/capture et /bench/pose
    display_frame: np.ndarray    # image envoyée au flux MJPEG
    persons: list                # TrackedPerson visibles sur cette image
    present_list: list           # présents, PRESENCE_TIMEOUT compris
    fps: float
    received: float              # réception par le thread caméra (perf_counter)
    frame_start: float           # début du traitement (perf_counter)


class FramePipeline:
    """Traitement des images d'une caméra : tracker, pose, présents, FPS, dessin.

    Ne touche à aucun état partagé de l'application : les durées des étapes
    passent par `record`, le reste sort dans le FrameResult.
    """

    def __init__(self, camera_id, camera_tracker, record, tracks_path=""):
        self.camera_id = camera_id
        self.tracker = camera_tracker
        self.record = record
        self.frame_count = 0
        self.fps = 0.0
        # Horloge monotone : sous WSL2, l'horloge murale recule de ~0,8 s toutes
        # les 30 s environ, ce qui faisait bondir le FPS affiché.
        self._fps_time = time.monotonic()
        self._fps_counter = 0
        self.persons = []
        self._pose_cache = {}
        # PRESENCE_TIMEOUT : { track_id: (person_dict, last_seen_timestamp) }
        self.last_seen: dict = {}
        # Écrit ligne par ligne (buffering=1) : un arrêt par SIGTERM ne perd rien.
        self._tracks_log = (open(tracks_path, "w", newline="", buffering=1)
                            if tracks_path else None)
        self._tracks_writer = csv.writer(self._tracks_log) if self._tracks_log else None

    def process(self, received, frame) -> FrameResult:
        self.frame_count += 1
        self._fps_counter += 1
        frame_start = time.perf_counter()
        # Sans annotations, la frame brute part telle quelle dans le flux : ni
        # copie ni dessin (l'interface web dessine ses boîtes elle-même).
        display_frame = frame.copy() if config.MJPEG_ANNOTATE else frame

        # ── Pipeline Body-First (YOLO + DeepSORT + InsightFace) ──────────────
        try:
            self.persons = self.tracker.update(frame, self.frame_count)
            for stage, seconds in self.tracker.last_timings.items():
                self.record(stage, seconds)
            if self._tracks_writer:
                _log_tracks(self._tracks_writer, self.frame_count, self.persons)
        except Exception as e:
            logger.error("[%s] Erreur traitement : %s: %s", self.camera_id, type(e).__name__, e)

        # ── Pose estimation (toutes les FRAME_SKIP frames, sur le crop corps) ─
        if self.frame_count % config.FRAME_SKIP == 0:
            pose_start = time.perf_counter()
            self._pose_cache = {
                p.track_id: _estimate_pose_safe(frame, p)
                for p in self.persons
            }
            self.record('pose', time.perf_counter() - pose_start)

        # ── Dessin + construction de la liste de présence ────────────────────
        # Chaque bloc de la suite est chronométré : un blocage hors des étapes
        # mesurées (jusqu'à 14 s en endurance) se lit ainsi dans le journal.
        block_start = time.perf_counter()
        now = time.time()
        present_list = []
        present_ids = set()

        for person in self.persons:
            pose = self._pose_cache.get(person.track_id)
            if config.MJPEG_ANNOTATE:
                _draw_person(display_frame, person, pose, self.frame_count)
            entry = {
                'name': person.name,
                'track_id': person.track_id,
                'confidence': person.confidence,
                'pose': pose,
                'last_face_frame': person.last_face_frame,
            }
            present_list.append(entry)
            present_ids.add(person.track_id)
            self.last_seen[person.track_id] = (entry, now)

        # ── PRESENCE_TIMEOUT : réinjecter les personnes récemment vues ───────
        for tid, (person_entry, last_time) in list(self.last_seen.items()):
            if tid not in present_ids:
                if now - last_time <= config.PRESENCE_TIMEOUT:
                    present_list.append(person_entry)
                else:
                    del self.last_seen[tid]

        self.record('present_list', time.perf_counter() - block_start)

        # ── FPS ──────────────────────────────────────────────────────────────
        elapsed = time.monotonic() - self._fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = time.monotonic()

        if config.MJPEG_ANNOTATE:
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
            cv2.putText(display_frame, f"Personnes: {len(present_list)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 245, 160), 2)

        return FrameResult(frame=frame, display_frame=display_frame, persons=self.persons,
                           present_list=present_list, fps=self.fps, received=received,
                           frame_start=frame_start)

    def close(self):
        self.tracker.release()
        if self._tracks_log:
            self._tracks_log.close()


def _apply_frame_result(camera, result: FrameResult) -> None:
    """Répercute une image traitée sur l'application : historique, journal
    d'événements, alertes, flux MJPEG et état de la caméra.

    La reconnaissance faciale, l'historique et les journaux sont partagés entre
    caméras : cette partie reste dans le processus de l'application.
    """
    camera_state = camera.state
    persons = result.persons
    now = time.time()

    # ── Historique des présences ─────────────────────────────────────────────
    block_start = time.perf_counter()
    # Seuls les noms dont le visage a été reconnu récemment : une piste
    # nommée garde son nom de dos, y compris après un échange de pistes.
    # Une erreur de base de données ne doit pas arrêter le pipeline vidéo.
    confirmed = camera.face_confirmed.update(
        [(p.track_id, p.name, p.last_face_frame) for p in persons], now)
    try:
        for event in presence_log.update(confirmed, now):
            logger.info("Présence : %s %s", event['name'],
                        "arrivé(e)" if event['type'] == 'arrival' else "parti(e)")
            events.publish(event)
    except Exception as e:
        logger.warning("Historique de présence indisponible : %s: %s", type(e).__name__, e)
    timings.record('presence', time.perf_counter() - block_start)
    block_start = time.perf_counter()
    try:
        event_log.write(camera.track_events.update(
            [(p.track_id, p.name, p.last_face_frame) for p in persons], event_clock()))
    except Exception as e:
        logger.warning("Journal d'événements indisponible : %s: %s", type(e).__name__, e)
    timings.record('events', time.perf_counter() - block_start)

    # ── Alerte : personne visible restée inconnue ────────────────────────────
    block_start = time.perf_counter()
    for track_id in camera.unknown_watcher.update([(p.track_id, p.name)
                                                   for p in persons], now):
        _publish('unknown', track_id=track_id, camera=camera.id)
    timings.record('alerts', time.perf_counter() - block_start)

    # ── Mise à jour de l'état global ─────────────────────────────────────────
    with timings.measure('encode'):
        _publish_display_frame(result.display_frame, camera_state)
    timings.record('frame', time.perf_counter() - result.frame_start)
    # Réception par le thread caméra → image et noms publiés pour les pages :
    # attente en file comprise. En every_frame, l'attente est voulue.
    timings.record('latency', time.perf_counter() - result.received)
    with camera_state.lock:
        camera_state.currently_present = result.present_list
        camera_state.fps = result.fps
        camera_state.bench_frame = result.frame
        camera_state.bench_persons = persons


def processing_loop(camera=None):
    """
    Consomme les frames de la file d'une caméra et applique le pipeline AI.
    Ne touche plus à la caméra — entièrement découplé de camera_loop.
    Un thread par caméra ; la reconnaissance faciale et les journaux sont partagés.
    """
    camera = camera or cameras[0]
    logger.info("[%s] Thread de traitement démarré.", camera.id)

    pipeline = FramePipeline(
        camera.id, camera.tracker, timings.record,
        _per_camera_path(config.TRACKS_LOG_PATH, camera) if config.TRACKS_LOG_PATH else "")
    camera.pipeline = pipeline
    endurance = probes = None
    # Un seul journal d'endurance, tenu par la première caméra, pour toutes.
    if config.ENDURANCE_LOG_PATH and camera is cameras[0]:
        probes = _EnduranceProbes(config.ENDURANCE_INTERVAL_S)
        endurance = EnduranceLog(config.ENDURANCE_LOG_PATH, config.ENDURANCE_INTERVAL_S,
                                 lambda: _endurance_probe(pipeline.frame_count, pipeline.fps,
                                                          pipeline.persons, probes))

    while state.running:
        try:
            received, frame = camera.queue.get(timeout=1.0)
        except queue.Empty:
            continue

        _apply_frame_result(camera, pipeline.process(received, frame))
        if endurance:
            # Une mesure qui échoue ne doit pas arrêter le pipeline vidéo.
            try:
                endurance.maybe_write()
            except Exception as e:
                logger.warning("Journal d'endurance indisponible : %s: %s", type(e).__name__, e)

    pipeline.close()
    if endurance:
        endurance.close()
        probes.windows.stop()
    logger.info("[%s] Thread de traitement arrêté.", camera.id)


def _per_camera_path(path: str, camera) -> str:
    """Un fichier par caméra s'il y en a plusieurs : tracks.csv → tracks-cam1.csv."""
    if len(cameras) == 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}-{camera.id}{ext}"


class _EnduranceProbes:
    """Sources de mesure du journal d'endurance, créées seulement s'il est activé."""

    def __init__(self, interval_s: float) -> None:
        import psutil

        timings.track_intervals()
        self.process = psutil.Process()
        self.nvml = Nvml()
        # Toutes les ~interval : un appel PowerShell prend quelques secondes.
        self.windows = WindowsCounters(interval_s)
        self.windows.start()
        process_stats(self.process)   # amorce cpu_percent : la 1re valeur vaut 0
        system_stats()


def _endurance_probe(frame_count, fps, persons, probes) -> dict:
    """Mesures du test d'endurance (cf. config.ENDURANCE_LOG_PATH, core/system_probe.py).

    Colonnes `ctx_*` : environnement (GPU entier, Windows, WSL), affichées par
    le rapport sans être jugées. torch_reserved_mb ne compte que l'allocateur
    de torch, sans TensorRT ni onnxruntime ; wsl_vram_mb (compteurs Windows)
    est la VRAM de tout WSL, donc de VisionCam pendant un run.
    """
    import torch

    row = {'frames': frame_count, 'frames_dropped': sum(c.state.frames_dropped for c in cameras),
           'source_skipped': sum(c.state.source_skipped for c in cameras), 'fps': round(fps, 2),
           'rss_mb': round(probes.process.memory_info().rss / 2**20, 1)}
    row.update(process_stats(probes.process))
    if torch.cuda.is_available():
        row['torch_reserved_mb'] = round(torch.cuda.memory_reserved() / 2**20)
    # Chaque étape sur toute la minute écoulée, latence de bout en bout comprise.
    for stage, stats in timings.drain_intervals().items():
        row[f'stage_{stage}_p50_ms'] = stats['median_ms']
        row[f'stage_{stage}_p95_ms'] = stats['p95_ms']
        row[f'stage_{stage}_max_ms'] = stats['max_ms']
    row['visible_tracks'] = sum(len(c.state.bench_persons) for c in cameras)
    if len(cameras) > 1:
        # FPS par caméra : le partage du GPU se lit caméra par caméra (roadmap 2.2).
        for c in cameras:
            row[f'cam_{c.id}_fps'] = round(c.state.fps, 2)
    # Grossit normalement (sessions de présence) : affichée, pas jugée.
    if os.path.exists(config.PRESENCE_DB_PATH):
        row['presence_db_kb'] = round(os.path.getsize(config.PRESENCE_DB_PATH) / 1024)
    sizes: dict = {}
    for c in cameras:
        for k, v in c.tracker.state_sizes().items():
            sizes[k] = sizes.get(k, 0) + v
    row.update({f'size_{k}': v for k, v in sizes.items()})
    row.update({
        'size_last_seen': sum(len(c.pipeline.last_seen) for c in cameras if c.pipeline),
        'size_unknown_watcher': sum(len(c.unknown_watcher._first_unknown)
                                    + len(c.unknown_watcher._alerted) for c in cameras),
        'size_face_confirmed': sum(len(c.face_confirmed._recognized) for c in cameras),
        'size_presence_open': len(presence_log._open),
        'size_track_events': sum(len(c.track_events._tracks) for c in cameras),
        'size_event_subscribers': len(events._subscribers),
    })
    row.update(probes.windows.latest())
    row.update(probes.nvml.stats())
    row.update(system_stats())
    return row



def _log_tracks(writer, frame_count, persons) -> None:
    """Une ligne par piste visible : image,track_id,x,y,largeur,hauteur,nom."""
    for p in persons:
        x1, y1, x2, y2 = p.body_bbox
        writer.writerow([frame_count, p.track_id, x1, y1, x2 - x1, y2 - y1, p.name])


# ══════════════════════════════════════════
# ROUTES FLASK
# ══════════════════════════════════════════
def _publish_display_frame(display_frame, camera_state=None):
    """
    Encode la frame annotée pour le flux MJPEG et réveille les clients.

    L'encodage JPEG (plusieurs ms en 1080p) se fait hors de state.lock, que
    /status et les autres routes attendaient sinon à chaque frame. Sans
    client connecté, rien n'est encodé : /api/capture et /bench/pose lisent la
    frame brute, pas ce JPEG.
    """
    camera_state = camera_state or state
    with camera_state.lock:
        if camera_state.stream_clients == 0:
            return
    _, buffer = cv2.imencode(
        '.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
    )
    with camera_state.frame_ready:
        camera_state.current_frame = buffer.tobytes()
        camera_state.frame_seq += 1
        camera_state.frame_ready.notify_all()


def generate_frames(camera_state=None):
    """
    Flux MJPEG : n'envoie une frame que lorsqu'elle est nouvelle.

    Renvoyer la dernière frame à cadence fixe dupliquait l'image dès que le
    pipeline tournait moins vite que MJPEG_FPS_LIMIT. MJPEG_FPS_LIMIT reste un
    plafond, pour les clients lents.
    """
    camera_state = camera_state or state
    interval = 1.0 / max(1, config.MJPEG_FPS_LIMIT)
    with camera_state.lock:
        camera_state.stream_clients += 1
        last_seq = camera_state.frame_seq
    try:
        while True:
            with camera_state.frame_ready:
                camera_state.frame_ready.wait_for(lambda: camera_state.frame_seq != last_seq,
                                                  timeout=1.0)
                if camera_state.frame_seq == last_seq:
                    continue
                last_seq = camera_state.frame_seq
                frame = camera_state.current_frame
            sent_at = time.monotonic()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(max(0.0, interval - (time.monotonic() - sent_at)))
    finally:
        # Déconnexion du navigateur : Flask ferme le générateur.
        with camera_state.lock:
            camera_state.stream_clients -= 1


# ══════════════════════════════════════════
# ACCÈS PROTÉGÉ
# ══════════════════════════════════════════
# Échecs de connexion récents par adresse : { ip: deque[horodatages] }.
_login_failures: dict = defaultdict(deque)
_login_failures_lock = threading.Lock()
# Accessibles sans session : la page de connexion et ses ressources.
_PUBLIC_ENDPOINTS = {'login', 'logout', 'static'}


def auth_enabled():
    return bool(config.ADMIN_PASSWORD_HASH or config.ADMIN_PASSWORD)


def _password_ok(candidate):
    if config.ADMIN_PASSWORD_HASH:
        return check_password_hash(config.ADMIN_PASSWORD_HASH, candidate)
    return hmac.compare_digest(candidate.encode(), config.ADMIN_PASSWORD.encode())


def _safe_next(target):
    """Chemin interne uniquement : pas de redirection vers un autre site."""
    if target and target.startswith('/') and not target.startswith('//') and '\\' not in target:
        return target
    return url_for('index')


def _persist_secret_key():
    """Clé de session stable entre deux lancements (data/secret_key, droits 600)."""
    if config.SECRET_KEY:
        return
    path = config.SECRET_KEY_PATH
    if os.path.exists(path):
        with open(path) as f:
            app.secret_key = f.read().strip()
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(app.secret_key)


@app.before_request
def require_login():
    if not auth_enabled() or session.get('authenticated') or request.endpoint in _PUBLIC_ENDPOINTS:
        return None
    wants_page = request.method == 'GET' and request.accept_mimetypes.accept_html \
        and not request.path.startswith(('/api/', '/status', '/video'))
    if wants_page:
        return redirect(url_for('login', next=request.full_path.rstrip('?')))
    return jsonify({'success': False, 'message': 'Connexion requise.'}), 401


@app.route('/login', methods=['GET', 'POST'])
def login():
    if not auth_enabled():
        return redirect(url_for('index'))
    error, status_code = None, 200
    if request.method == 'POST':
        ip = request.remote_addr or '?'
        now = time.time()
        with _login_failures_lock:
            failures = _login_failures[ip]
            while failures and now - failures[0] > config.LOGIN_FAILURE_WINDOW_S:
                failures.popleft()
            blocked = len(failures) >= config.LOGIN_MAX_FAILURES
        if blocked:
            error, status_code = 'Trop de tentatives : réessayez dans quelques minutes.', 429
        elif _password_ok(request.form.get('password', '')):
            with _login_failures_lock:
                _login_failures.pop(ip, None)
            session.clear()
            session.permanent = True
            session['authenticated'] = True
            logger.info("Connexion réussie depuis %s", ip)
            return redirect(_safe_next(request.args.get('next')))
        else:
            with _login_failures_lock:
                _login_failures[ip].append(now)
            logger.warning("Échec de connexion depuis %s", ip)
            error, status_code = 'Mot de passe incorrect.', 401
    return render_template('login.html', error=error), status_code


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login') if auth_enabled() else url_for('index'))


@app.route('/')
def index():
    return _page('live.html', 'live')


def _page(template, page):
    return render_template(template, page=page, auth_enabled=auth_enabled())


@app.route('/people')
def people_page():
    return _page('people.html', 'people')


@app.route('/history')
def history_page():
    return _page('history.html', 'history')


@app.route('/diagnostics')
def diagnostics_page():
    return _page('diagnostics.html', 'diagnostics')


def _requested_camera(camera_id=None):
    """Caméra demandée (`?camera=` ou champ `camera`), la première par défaut ; None si inconnue."""
    camera_id = camera_id or request.args.get('camera')
    if not camera_id:
        return cameras[0]
    return cameras_by_id.get(camera_id)


def _unknown_camera():
    return jsonify({'success': False, 'message': 'Caméra inconnue.',
                    'cameras': [c.id for c in cameras]}), 404


@app.route('/video')
def video_feed():
    """Flux MJPEG d'une caméra (`?camera=id`, la première par défaut)."""
    camera = _requested_camera()
    if camera is None:
        return _unknown_camera()
    return Response(generate_frames(camera.state),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


_rebuild_lock = threading.Lock()


@app.route('/rebuild', methods=['POST'])
def rebuild():
    """
    Reconstruit la base d'embeddings depuis known_faces/ à chaud.
    Lance le rebuild dans un thread daemon pour ne pas bloquer la réponse HTTP.

    Réponse immédiate :
        202 { started: true,  message: str }
        409 { started: false, message: str }  ← reconstruction déjà en cours
    """
    # Deux rebuilds en parallèle écriraient le cache chacun de leur côté :
    # une seule reconstruction à la fois, rendue à la fin du thread.
    if not _rebuild_lock.acquire(blocking=False):
        return jsonify({'started': False, 'message': 'Une reconstruction est déjà en cours.'}), 409

    def _do_rebuild():
        try:
            logger.info("Rebuild base embeddings demandé via UI...")
            face_recognizer.rebuild_database()
            with state.lock:
                state.total_known = len(face_recognizer.known_names)
            logger.info("Rebuild terminé : %d entrée(s)", state.total_known)
            _publish('rebuilt', total_known=state.total_known)
        finally:
            _rebuild_lock.release()

    threading.Thread(target=_do_rebuild, daemon=True).start()
    return jsonify({'started': True, 'message': 'Reconstruction lancée en arrière-plan.'}), 202


# ══════════════════════════════════════════
# GESTION DES PERSONNES
# ══════════════════════════════════════════
# Issue d'une opération de FaceRecognizer → code HTTP.
_PEOPLE_STATUS = {'ok': 200, 'invalid': 400, 'not_found': 404, 'conflict': 409}
THUMBNAIL_SIZE = 160


def _refresh_total_known():
    with state.lock:
        state.total_known = len(face_recognizer.known_names)


def _find_person(name):
    return next((p for p in face_recognizer.list_people() if p['name'] == name), None)


@app.route('/api/people')
def api_people():
    """
    Personnes connues.

    Réponse : 200 { people: [{ name, templates[], photos, enrolled, thumbnail_url|null }] }
    """
    people = [
        {**{k: v for k, v in p.items() if k != 'thumbnail'},
         'thumbnail_url': f"/api/people/{p['name']}/thumbnail" if p['thumbnail'] else None}
        for p in face_recognizer.list_people()
    ]
    return jsonify({'people': people})


@app.route('/api/people/<name>/thumbnail')
def api_person_thumbnail(name):
    """Première photo de la personne, réduite à THUMBNAIL_SIZE px : 200 image/jpeg ou 404."""
    person = _find_person(name)
    image = cv2.imread(person['thumbnail']) if person and person['thumbnail'] else None
    if image is None:
        abort(404)
    scale = THUMBNAIL_SIZE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buffer.tobytes(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-cache'})


@app.route('/api/people/<name>/rename', methods=['POST'])
def api_rename_person(name):
    """
    Entrée (JSON) : { new_name: str }
    Réponse : 200 | 400 nom invalide | 404 inconnue | 409 nom déjà pris — { success, message }
    """
    new_name = str((request.get_json(silent=True) or {}).get('new_name', '')).strip()
    status, message = face_recognizer.rename_person(name, new_name)
    if status == 'ok' and new_name != name:
        presence_log.rename(name, new_name)
        event_log.rename(name, new_name)
        _publish('renamed', name=new_name, old_name=name)
    return jsonify({'success': status == 'ok', 'message': message}), _PEOPLE_STATUS[status]


@app.route('/api/people/<name>', methods=['DELETE'])
def api_delete_person(name):
    """Supprime photos, entrées et historique. Réponse : 200 | 400 | 404 — { success, message }"""
    status, message = face_recognizer.delete_person(name)
    if status == 'ok':
        # Droit à l'effacement : l'historique de présence et les événements de
        # ses pistes partent avec la personne.
        presence_log.forget(name)
        event_log.forget(name)
        _refresh_total_known()
        _publish('deleted', name=name)
    return jsonify({'success': status == 'ok', 'message': message}), _PEOPLE_STATUS[status]


@app.route('/api/people/<name>/photos', methods=['POST'])
def api_add_photos(name):
    """
    Ajoute des photos à une personne (ou la crée) et recalcule son embedding.

    Entrée (multipart) : images — fichier(s) image
    Réponse : 200 | 400 aucune image lisible | 422 aucun visage — { success, message, total_known }
    """
    images = []
    for f in request.files.getlist('images'):
        img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    if not images:
        return jsonify({'success': False, 'message': 'Aucune image lisible (champ "images").'}), 400
    success, message = face_recognizer.enroll_person_average(name, images)
    if success:
        _publish('enrolled', name=name)
    _refresh_total_known()
    return jsonify({'success': success, 'message': message,
                    'total_known': state.total_known}), 200 if success else 422


# ══════════════════════════════════════════
# MESURE COMPARATIVE DES SOURCES D'ORIENTATION
# ══════════════════════════════════════════
_bench_lock = threading.Lock()
_bench_estimators: dict = {}


def _bench_get_estimators():
    """
    Instancie les deux estimateurs à la première mesure.

    Mediapipe met ~1 s à charger : le faire à la demande évite de payer ce coût
    au démarrage quand POSE_SOURCE vaut "yolo".
    """
    if not _bench_estimators:
        _bench_estimators['mediapipe'] = PoseEstimator()
        _bench_estimators['yolo'] = KeypointPoseEstimator()
    return _bench_estimators['mediapipe'], _bench_estimators['yolo']


def _percentile_ms(values, ratio):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(int(len(ordered) * ratio), len(ordered) - 1)], 3)


@app.route('/bench/pose', methods=['POST'])
def bench_pose():
    """
    Compare les deux sources d'orientation sur le flux en cours.

    Entrée (form) :
        samples : nombre d'échantillons de frames (5-200, défaut 30)

    Réponse :
        200 { success: true, ... mesures ... }
        400 { success: false, message }  ← paramètre invalide
        409 { success: false, message }  ← mesure déjà en cours
        503 { success: false, message }  ← aucune personne suivie
    """
    raw = request.form.get('samples', '30')
    try:
        samples = int(raw)
    except ValueError:
        return jsonify({'success': False,
                        'message': f'Paramètre "samples" invalide : {raw!r}.'}), 400
    samples = max(5, min(samples, 200))
    camera = _requested_camera(request.form.get('camera'))
    if camera is None:
        return _unknown_camera()

    # Mediapipe n'est pas réentrant : une seule mesure à la fois.
    if not _bench_lock.acquire(blocking=False):
        return jsonify({'success': False, 'message': 'Une mesure est déjà en cours.'}), 409

    try:
        mediapipe_est, keypoint_est = _bench_get_estimators()
        mp_times, kp_times, spans = [], [], []
        confusion: dict = {}
        agree = compared = mp_silent = kp_silent = 0

        for _ in range(samples):
            with camera.state.lock:
                frame = camera.state.bench_frame
                persons = list(camera.state.bench_persons)

            if frame is None or not persons:
                time.sleep(0.05)
                continue

            for person in persons:
                x1, y1, x2, y2 = person.body_bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                t0 = time.perf_counter()
                mp_verdict = mediapipe_est.estimate(crop)
                mp_times.append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                kp_verdict = keypoint_est.estimate(person.pose_kps)
                kp_times.append((time.perf_counter() - t0) * 1000)

                if person.pose_kps:
                    spans.append(abs(person.pose_kps[5][0] - person.pose_kps[6][0]))

                key = f"{mp_verdict} → {kp_verdict}"
                confusion[key] = confusion.get(key, 0) + 1
                mp_silent += mp_verdict is None
                kp_silent += kp_verdict is None
                if mp_verdict is not None:
                    compared += 1
                    agree += mp_verdict == kp_verdict

            time.sleep(0.05)

        if not mp_times:
            return jsonify({'success': False,
                            'message': 'Aucune personne suivie pendant la mesure.'}), 503

        return jsonify({
            'success': True,
            'observations': len(mp_times),
            'pose_source_actif': config.POSE_SOURCE,
            'mediapipe_ms': {'moy': round(statistics.mean(mp_times), 2),
                             'med': round(statistics.median(mp_times), 2),
                             'p95': _percentile_ms(mp_times, 0.95)},
            'yolo_ms': {'moy': round(statistics.mean(kp_times), 3),
                        'med': round(statistics.median(kp_times), 3),
                        'p95': _percentile_ms(kp_times, 0.95)},
            'economie_ms_par_frame': round(
                (statistics.mean(mp_times) - statistics.mean(kp_times))
                * len(mp_times) / max(1, samples), 2),
            'accord': {'compares': compared, 'accords': agree,
                       'taux': round(100 * agree / compared, 1) if compared else None},
            'abstentions': {'mediapipe': mp_silent, 'yolo': kp_silent},
            'ecart_epaules_px': {'min': round(min(spans), 1),
                                 'med': round(statistics.median(spans), 1),
                                 'max': round(max(spans), 1)} if spans else None,
            'seuil_fiabilite_px': config.POSE_MIN_SHOULDER_DIST_PX,
            'confusion': sorted(confusion.items(), key=lambda kv: -kv[1]),
        }), 200
    finally:
        _bench_lock.release()


@app.route('/status')
def status():
    """
    État courant d'une caméra (`?camera=id`, la première par défaut).

    Réponse : { currently_present[], fps, total_known,
                tracks: [{ track_id, name, bbox: [x1, y1, x2, y2], pose }],
                frame_size: [largeur, hauteur] | null, camera, cameras[] }

    `tracks` ne contient que les personnes visibles sur la dernière image de
    cette caméra, en pixels de cette image : la page les superpose au flux
    pour cliquer dessus. `currently_present` couvre toutes les caméras, chaque
    entrée avec sa caméra.
    """
    camera = _requested_camera()
    if camera is None:
        return _unknown_camera()
    return jsonify(_status_payload(camera))


def _status_payload(camera=None):
    camera = camera or cameras[0]
    present = []
    for c in cameras:
        with c.state.lock:
            present += [{**p, 'camera': c.id} for p in c.state.currently_present]
    camera_state = camera.state
    with camera_state.lock:
        poses = {p['track_id']: p['pose'] for p in camera_state.currently_present}
        tracks = [{'track_id': str(p.track_id), 'name': p.name,
                   'bbox': [int(v) for v in p.body_bbox], 'pose': poses.get(p.track_id)}
                  for p in camera_state.bench_persons]
        frame = camera_state.bench_frame
        return {
            'currently_present': present,
            'fps': camera_state.fps,
            'total_known': state.total_known,
            'tracks': tracks,
            'frame_size': [frame.shape[1], frame.shape[0]] if frame is not None else None,
            'camera': camera.id,
            'cameras': [c.id for c in cameras],
        }


# ══════════════════════════════════════════
# TEMPS RÉEL ET DIAGNOSTIC
# ══════════════════════════════════════════
# Période d'envoi de l'état aux pages abonnées à /api/events.
STATUS_PUSH_INTERVAL = 0.5


@app.route('/api/events')
def api_events():
    """
    Server-Sent Events : un message `data: {json}` par événement.

    - { type: "status", ... }  l'état de /status, toutes les STATUS_PUSH_INTERVAL s
    - { type: "arrival" | "departure", name, time }  historique des présences
    - { type: "unknown", track_id, time }  personne restée inconnue UNKNOWN_ALERT_S
    - { type: "enrolled" | "deleted", name, time }, { type: "renamed", name, old_name, time }
    - { type: "rebuilt", total_known, time }

    Chaque page abonnée garde un thread du serveur (cf. SERVER_THREADS).
    """
    camera = _requested_camera()
    if camera is None:
        return _unknown_camera()

    def stream():
        subscription = events.subscribe()
        try:
            # Reconnexion automatique du navigateur après 2 s si le flux coupe.
            yield 'retry: 2000\n\n'
            next_status = 0.0
            while state.running:
                now = time.monotonic()
                if now >= next_status:
                    yield f"data: {json.dumps({'type': 'status', **_status_payload(camera)})}\n\n"
                    next_status = now + STATUS_PUSH_INTERVAL
                try:
                    event = subscription.get(timeout=max(0.0, next_status - time.monotonic()))
                except queue.Empty:
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            events.unsubscribe(subscription)

    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/diagnostics')
def api_diagnostics():
    """
    Réponse : { uptime_s, fps, timings: {étape: {median_ms, p95_ms, samples}},
                gpu: { name, memory_used_mb, memory_total_mb } | null,
                models: {...}, recognition: {...}, tracking: {...},
                camera: {...}, clients: { video, events } }
    """
    import torch

    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {'name': torch.cuda.get_device_name(0),
               'memory_used_mb': round((total - free) / 2**20),
               'memory_total_mb': round(total / 2**20)}
    camera = _requested_camera()
    if camera is None:
        return _unknown_camera()
    with camera.state.lock:
        fps = camera.state.fps
        frame = camera.state.bench_frame
    video_clients = sum(c.state.stream_clients for c in cameras)
    return jsonify({
        'auth_enabled': auth_enabled(),
        'uptime_s': round(time.time() - STARTED_AT),
        'fps': fps,
        'timings': timings.summary(),
        'gpu': gpu,
        'models': {
            'detector': config.YOLO_MODEL,
            'appearance_embedder': config.DEEPSORT_EMBEDDER_ENGINE or 'MobileNetV2 PyTorch',
            'tracker_backend': config.TRACKER_BACKEND,
            'face_model': config.INSIGHTFACE_MODEL,
            'pose_source': config.POSE_SOURCE,
        },
        'recognition': {
            'threshold': config.RECOGNITION_THRESHOLD,
            'min_face_px': config.RECOGNITION_MIN_FACE_PX,
            'known_entries': state.total_known,
        },
        'tracking': {
            'max_cosine_distance': config.DEEPSORT_MAX_COSINE_DISTANCE,
            'max_iou_distance': config.DEEPSORT_MAX_IOU_DISTANCE,
            'max_age': config.DEEPSORT_MAX_AGE,
            'n_init': config.DEEPSORT_N_INIT,
        },
        'camera': {
            'id': camera.id,
            'source': _source_label(camera),
            'frame_size': [frame.shape[1], frame.shape[0]] if frame is not None else None,
        },
        'cameras': [{'id': c.id, 'source': _source_label(c), 'fps': c.state.fps}
                    for c in cameras],
        'clients': {'video': video_clients, 'events': events.subscriber_count},
    })


def _source_label(camera) -> str:
    if camera.is_file:
        return f"vidéo {os.path.basename(str(camera.source).rstrip('/'))}"
    if isinstance(camera.source, int):
        return f"webcam {camera.source}"
    return str(camera.source)


# ══════════════════════════════════════════
# CAPTURE D'UNE PERSONNE DU FLUX
# ══════════════════════════════════════════
# Poses du mode guidé : un template par angle (enroll_person_multitemplate).
CAPTURE_LABELS = ('Face', 'ProfilG', 'ProfilD')


@app.route('/api/capture', methods=['POST'])
def api_capture():
    """
    Enrôle la personne désignée dans le flux (clic sur sa boîte).

    Entrée (JSON) : { name, track_id, label? } — label ∈ CAPTURE_LABELS pour le
    mode guidé (un template par angle), absent pour ajouter une photo.

    Réponse : 200 | 400 champ invalide | 404 personne plus visible
              | 422 aucun visage, ou visage trop petit — { success, message, total_known }
    """
    body = request.get_json(silent=True) or {}
    name = str(body.get('name', '')).strip()
    track_id = str(body.get('track_id', '')).strip()
    label = body.get('label')
    if not name or not track_id:
        return jsonify({'success': False, 'message': 'Champs "name" et "track_id" requis.'}), 400
    if label is not None and label not in CAPTURE_LABELS:
        return jsonify({'success': False,
                        'message': f'Label invalide : attendu {", ".join(CAPTURE_LABELS)}.'}), 400

    camera = _requested_camera(body.get('camera'))
    if camera is None:
        return _unknown_camera()
    with camera.state.lock:
        frame = camera.state.bench_frame
        person = next((p for p in camera.state.bench_persons
                       if str(p.track_id) == track_id), None)
    if frame is None or person is None:
        return jsonify({'success': False,
                        'message': "Cette personne n'est plus visible : cliquez à nouveau."}), 404

    box = FaceBodyTracker.head_crop_box(frame.shape, person.body_bbox, person.pose_kps)
    if box is None:
        return jsonify({'success': False, 'message': 'Personne trop petite dans l\'image.'}), 422
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2].copy()

    # Même visage que celui que la reconnaissance verra : le plus proche du
    # centre du recadrage de tête.
    face = face_recognizer.recognize_center_face(crop)
    if face is None:
        return jsonify({'success': False,
                        'message': 'Aucun visage détecté : la personne doit regarder vers la caméra.'}), 422
    fx1, fy1, fx2, fy2 = face['bbox']
    min_side = 2 * config.RECOGNITION_MIN_FACE_PX
    if min(fx2 - fx1, fy2 - fy1) < min_side:
        return jsonify({'success': False,
                        'message': f'Visage trop petit ({min(fx2 - fx1, fy2 - fy1)} px, '
                                   f'{min_side} px minimum) : rapprochez-vous de la caméra.'}), 422

    if label is None:
        success, message = face_recognizer.enroll_person_average(name, [crop])
    else:
        success, message = face_recognizer.enroll_person_multitemplate(name, {label: crop})
    if success:
        _publish('enrolled', name=name)
    _refresh_total_known()
    code = 200 if success else (400 if message.startswith(('Nom invalide', 'Label')) else 422)
    return jsonify({'success': success, 'message': message,
                    'total_known': state.total_known}), code


# ══════════════════════════════════════════
# HISTORIQUE DES PRÉSENCES
# ══════════════════════════════════════════
def _history_query():
    """Filtres communs de /api/history : (sessions, None) ou (None, réponse d'erreur 400).

    Paramètres : name, from et to (AAAA-MM-JJ, heure locale ; `to` inclus), limit.
    """
    try:
        start = datetime.strptime(request.args['from'], '%Y-%m-%d') if request.args.get('from') else None
        end = datetime.strptime(request.args['to'], '%Y-%m-%d') + timedelta(days=1) \
            if request.args.get('to') else None
        limit = max(1, min(int(request.args.get('limit', 1000)), 10000))
    except ValueError:
        return None, (jsonify({'message': 'Filtres invalides : dates AAAA-MM-JJ, limit entier.'}), 400)
    # Recalculées du journal d'événements, la donnée source (roadmap 2.1).
    sessions = presence_history(
        presence_log, event_log, config.PRESENCE_LOG_GAP_S, config.PRESENCE_FACE_MAX_AGE_S,
        event_clock(),
        name=request.args.get('name') or None,
        start=start.timestamp() if start else None,
        end=end.timestamp() if end else None,
        limit=limit,
    )
    return sessions, None


@app.route('/api/history')
def api_history():
    """
    Sessions de présence, les plus récentes d'abord.

    Réponse : 200 { sessions: [{ id, name, arrived, departed|null, last_seen,
                                 ongoing, duration_s }], names: [...] } | 400
    """
    sessions, error = _history_query()
    if error:
        return error
    return jsonify({'sessions': sessions, 'names': history_names(presence_log, event_log)})


@app.route('/api/history.csv')
def api_history_csv():
    """Mêmes filtres que /api/history, en CSV à télécharger."""
    sessions, error = _history_query()
    if error:
        return error
    filename = f"presences-{datetime.now():%Y%m%d-%H%M}.csv"
    return Response(PresenceLog.to_csv(sessions), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


# ══════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════
if __name__ == '__main__':
    # Deux threads par caméra : lecture, et traitement (roadmap 2.2).
    threads = []
    for camera in cameras:
        threads += [threading.Thread(target=camera_loop, args=(camera,), daemon=True,
                                     name=f"camera-{camera.id}"),
                    threading.Thread(target=processing_loop, args=(camera,), daemon=True,
                                     name=f"processing-{camera.id}")]
    for thread in threads:
        thread.start()
    _persist_secret_key()
    if not auth_enabled():
        logger.warning("Accès NON protégé : définir ADMIN_PASSWORD_HASH ou ADMIN_PASSWORD "
                       "dans .env pour exiger une connexion.")
    logger.info("Démarrage sur http://%s:%d (waitress, %d threads)",
                config.FLASK_HOST, config.FLASK_PORT, config.SERVER_THREADS)

    # waitress plutôt que le serveur de développement de Flask, qui n'est pas
    # fait pour tourner en continu. Il fonctionne sous Linux comme sous Windows.
    from waitress import serve

    # `scripts/visioncam.sh stop` envoie SIGTERM : sans handler, Python quitte
    # sans passer par le finally et les sessions de présence restent ouvertes.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        serve(app, host=config.FLASK_HOST, port=config.FLASK_PORT,
              threads=config.SERVER_THREADS, ident="VisionCam")
    finally:
        state.running = False
        # Attendre la fin des threads : quitter pendant une inférence GPU
        # interrompait le processus (« terminate called without an active
        # exception »), et une frame traitée après close_all rouvrirait une session.
        for thread in threads:
            thread.join(timeout=10)
        # Ferme les sessions et les pistes en cours à leur dernière heure vue.
        presence_log.close_all()
        for camera in cameras:
            event_log.write(camera.track_events.finish())
        # Vide la file d'écriture avant de quitter : rien n'est perdu à l'arrêt.
        db_writer.close()
