import os

# OpenBLAS ne lit cette variable qu'au chargement de numpy : elle doit précéder
# tous les imports. Ses threads se disputaient le CPU avec torch et
# onnxruntime pour des matrices trop petites pour en profiter (distances
# cosinus de DeepSORT) : sur MOT17-04, l'association passait de 2 à 24 ms et le
# prétraitement des crops de 3 à 23 ms. setdefault laisse la main à un réglage
# explicite de l'environnement.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hmac
import json
import logging
import queue
import secrets
import statistics
import threading
import time
from collections import defaultdict, deque
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
from core.events import EventBus, StageTimer, UnknownWatcher
from core.face_body_tracker import FaceBodyTracker
from core.face_recognition import FaceRecognizer
from core.pose_estimation import PoseEstimator
from core.pose_from_keypoints import KeypointPoseEstimator
from core.presence_log import PresenceLog

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
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        # Réveille les flux MJPEG à chaque nouvelle frame (partage self.lock).
        self.frame_ready = threading.Condition(self.lock)
        self.frame_seq = 0
        self.stream_clients = 0
        self.current_frame = None
        self.currently_present = []
        self.fps = 0.0
        self.total_known = len(face_recognizer.known_names)
        self.running = True
        # PRESENCE_TIMEOUT : { track_id: (person_dict, last_seen_timestamp) }
        self.last_seen: dict = {}
        # Dernière frame BRUTE (sans overlay) et TrackedPerson associés.
        # Alimentent /bench/pose, qui a besoin des crops et des keypoints,
        # pas du JPEG annoté envoyé au navigateur.
        self.bench_frame = None
        self.bench_persons: list = []

state = AppState()
presence_log = PresenceLog(config.PRESENCE_DB_PATH, gap=config.PRESENCE_LOG_GAP_S,
                           retention=config.PRESENCE_RETENTION_DAYS * 86400 or None)
events = EventBus()
timings = StageTimer()
unknown_watcher = UnknownWatcher(config.UNKNOWN_ALERT_S)
STARTED_AT = time.time()


def _publish(event_type, **fields):
    """Diffuse un événement aux pages ouvertes (/api/events)."""
    events.publish({'type': event_type, 'time': datetime.now().isoformat(timespec='seconds'),
                    **fields})

# Queue inter-thread caméra → AI. Taille 2 : on garde toujours la frame la plus fraîche.
_frame_queue: queue.Queue = queue.Queue(maxsize=2)


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
        logger.warning("Échec pose sur track #%d : %s: %s",
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


def _open_camera():
    """
    Ouvre la caméra avec un timeout court pour ne pas bloquer le thread.
    Retourne un objet VideoCapture ouvert, ou None si l'ouverture échoue.
    """
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)

    # Timeout de connexion et de lecture (ms). Sans ça, VideoCapture sur HTTP
    # peut bloquer jusqu'à 30s sans aucun log — invisible pour l'utilisateur.
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.CAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.CAM_READ_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)

    if not cap.isOpened():
        cap.release()
        return None
    logger.info("Caméra ouverte : %dx%d", int(cap.get(3)), int(cap.get(4)))
    return cap


def _reconnect_camera(cap):
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

        new_cap = _open_camera()
        if new_cap is not None:
            logger.info("Reconnexion réussie après %d tentative(s)", attempt)
            return new_cap

        delay = min(delay * 2, 30.0)

    return None


# ══════════════════════════════════════════
# THREAD A — LECTURE CAMÉRA
# ══════════════════════════════════════════
def camera_loop():
    """
    Lit les frames depuis la caméra et les pousse dans _frame_queue.
    Entièrement découplé du pipeline AI : cap.read() ne bloque jamais
    le calcul YOLO/InsightFace. La queue conserve toujours la frame
    la plus récente (drop oldest si pleine).
    """
    logger.info("Thread caméra démarré.")
    cap = _open_camera()
    if cap is None:
        logger.error("Impossible d'ouvrir la caméra au démarrage : %s", config.CAMERA_SOURCE)
        cap = _reconnect_camera(None)
        if cap is None:
            return

    consecutive_failures = 0

    while state.running:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures == 1:
                logger.warning("Frame perdue...")
            elif consecutive_failures >= config.CAM_MAX_FAILURES:
                logger.error("%d échecs consécutifs — déconnexion détectée.", consecutive_failures)
                cap = _reconnect_camera(cap)
                if cap is None:
                    break
                consecutive_failures = 0
            else:
                time.sleep(0.05)
            continue

        consecutive_failures = 0

        # Toujours garder la frame la plus fraîche : drop oldest si queue pleine.
        try:
            _frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                _frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                _frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    cap.release()
    logger.info("Thread caméra arrêté.")


# ══════════════════════════════════════════
# THREAD B — PIPELINE AI
# ══════════════════════════════════════════
def processing_loop():
    """
    Consomme les frames de _frame_queue et applique le pipeline AI.
    Ne touche plus à la caméra — entièrement découplé de camera_loop.
    """
    logger.info("Thread de traitement démarré.")

    frame_count = 0
    fps_time = time.time()
    fps_counter = 0
    persons_cache = []
    pose_cache = {}

    while state.running:
        try:
            frame = _frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_count += 1
        fps_counter += 1
        frame_start = time.perf_counter()
        # Sans annotations, la frame brute part telle quelle dans le flux : ni
        # copie ni dessin (l'interface web dessine ses boîtes elle-même).
        display_frame = frame.copy() if config.MJPEG_ANNOTATE else frame

        # ── Pipeline Body-First (YOLO + DeepSORT + InsightFace) ──────────────
        try:
            persons_cache = tracker.update(frame, frame_count)
            for stage, seconds in tracker.last_timings.items():
                timings.record(stage, seconds)
        except Exception as e:
            logger.error("Erreur traitement : %s: %s", type(e).__name__, e)

        # ── Pose estimation (toutes les FRAME_SKIP frames, sur le crop corps) ─
        if frame_count % config.FRAME_SKIP == 0:
            with timings.measure('pose'):
                pose_cache = {
                    p.track_id: _estimate_pose_safe(frame, p)
                    for p in persons_cache
                }

        # ── Dessin + construction de la liste de présence ────────────────────
        now = time.time()
        present_list = []
        present_ids = set()

        for person in persons_cache:
            pose = pose_cache.get(person.track_id)
            if config.MJPEG_ANNOTATE:
                _draw_person(display_frame, person, pose, frame_count)
            entry = {
                'name': person.name,
                'track_id': person.track_id,
                'confidence': person.confidence,
                'pose': pose,
                'last_face_frame': person.last_face_frame,
            }
            present_list.append(entry)
            present_ids.add(person.track_id)
            state.last_seen[person.track_id] = (entry, now)

        # ── PRESENCE_TIMEOUT : réinjecter les personnes récemment vues ───────
        for tid, (person_entry, last_time) in list(state.last_seen.items()):
            if tid not in present_ids:
                if now - last_time <= config.PRESENCE_TIMEOUT:
                    present_list.append(person_entry)
                else:
                    del state.last_seen[tid]

        # ── Historique des présences ─────────────────────────────────────────
        # Une erreur de base de données ne doit pas arrêter le pipeline vidéo.
        try:
            for event in presence_log.update({p.name for p in persons_cache}, now):
                logger.info("Présence : %s %s", event['name'],
                            "arrivé(e)" if event['type'] == 'arrival' else "parti(e)")
                events.publish(event)
        except Exception as e:
            logger.warning("Historique de présence indisponible : %s: %s", type(e).__name__, e)

        # ── Alerte : personne visible restée inconnue ────────────────────────
        for track_id in unknown_watcher.update([(p.track_id, p.name) for p in persons_cache], now):
            _publish('unknown', track_id=track_id)

        # ── FPS ──────────────────────────────────────────────────────────────
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()
        else:
            current_fps = state.fps

        if config.MJPEG_ANNOTATE:
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
            cv2.putText(display_frame, f"Personnes: {len(present_list)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 245, 160), 2)

        # ── Mise à jour de l'état global ─────────────────────────────────────
        with timings.measure('encode'):
            _publish_display_frame(display_frame)
        timings.record('frame', time.perf_counter() - frame_start)
        with state.lock:
            state.currently_present = present_list
            state.fps = current_fps
            state.bench_frame = frame
            state.bench_persons = persons_cache

    tracker.release()
    logger.info("Thread de traitement arrêté.")


# ══════════════════════════════════════════
# ROUTES FLASK
# ══════════════════════════════════════════
def _publish_display_frame(display_frame):
    """
    Encode la frame annotée pour le flux MJPEG et réveille les clients.

    L'encodage JPEG (plusieurs ms en 1080p) se fait hors de state.lock, que
    /status et les autres routes attendaient sinon à chaque frame. Sans
    client connecté, rien n'est encodé : /capture et /bench/pose lisent la
    frame brute, pas ce JPEG.
    """
    with state.lock:
        if state.stream_clients == 0:
            return
    _, buffer = cv2.imencode(
        '.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
    )
    with state.frame_ready:
        state.current_frame = buffer.tobytes()
        state.frame_seq += 1
        state.frame_ready.notify_all()


def generate_frames():
    """
    Flux MJPEG : n'envoie une frame que lorsqu'elle est nouvelle.

    Renvoyer la dernière frame à cadence fixe dupliquait l'image dès que le
    pipeline tournait moins vite que MJPEG_FPS_LIMIT. MJPEG_FPS_LIMIT reste un
    plafond, pour les clients lents.
    """
    interval = 1.0 / max(1, config.MJPEG_FPS_LIMIT)
    with state.lock:
        state.stream_clients += 1
        last_seq = state.frame_seq
    try:
        while True:
            with state.frame_ready:
                state.frame_ready.wait_for(lambda: state.frame_seq != last_seq, timeout=1.0)
                if state.frame_seq == last_seq:
                    continue
                last_seq = state.frame_seq
                frame = state.current_frame
            sent_at = time.monotonic()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(max(0.0, interval - (time.monotonic() - sent_at)))
    finally:
        # Déconnexion du navigateur : Flask ferme le générateur.
        with state.lock:
            state.stream_clients -= 1


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


@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/enroll', methods=['POST'])
def enroll():
    """
    Enrôle une personne à chaud sans redémarrer l'application.

    Entrée (multipart/form-data) :
        name   : str — identifiant de la personne (ex: "Armand_Lauener")
        method : 'average' (défaut) | 'multitemplate'
        images : fichier(s) image JPG/PNG

    Réponse :
        200 { success: true,  message: str, total_known: int }
        400 { success: false, message: str }
        422 { success: false, message: str }  ← aucun visage trouvé
    """
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': 'Champ "name" manquant ou vide.'}), 400

    method = request.form.get('method', 'average').strip().lower()
    if method not in ('average', 'multitemplate'):
        return jsonify({'success': False,
                        'message': 'Champ "method" invalide : attendu "average" ou "multitemplate".'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'success': False, 'message': 'Aucune image fournie (champ "images").'}), 400

    decoded = []
    for f in files:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        label = os.path.splitext(f.filename)[0] if f.filename else f"img{len(decoded)}"
        decoded.append((label, img))

    if not decoded:
        return jsonify({'success': False, 'message': 'Impossible de décoder les images reçues.'}), 400

    if method == 'average':
        images = [img for _, img in decoded]
        success, message = face_recognizer.enroll_person_average(name, images)
    else:
        frames_dict = {label: img for label, img in decoded}
        success, message = face_recognizer.enroll_person_multitemplate(name, frames_dict)

    if success:
        _publish('enrolled', name=name)
        with state.lock:
            state.total_known = len(face_recognizer.known_names)
        return jsonify({'success': True, 'message': message,
                        'total_known': state.total_known}), 200

    return jsonify({'success': False, 'message': message}), 422


@app.route('/capture', methods=['POST'])
def capture():
    """
    Enrôle une personne depuis la frame courante du flux caméra.

    Entrée (multipart/form-data) :
        name : str — identifiant de la personne

    Réponse :
        200 { success: true,  message: str, total_known: int }
        400 { success: false, message: str }
        503 { success: false, message: str }  ← pas de frame disponible
        409 { success: false, message: str }  ← plusieurs personnes à l'écran
        422 { success: false, message: str }  ← aucun visage détecté
    """
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': 'Champ "name" manquant ou vide.'}), 400

    # Frame brute, pas le JPEG streamé : celui-ci porte les boîtes et les
    # textes dessinés, et a perdu du détail à la compression.
    with state.lock:
        frame = state.bench_frame
        person_count = len(state.bench_persons)

    if frame is None:
        return jsonify({'success': False, 'message': 'Aucune frame disponible (caméra déconnectée ?)'}), 503

    # L'enrôlement garde le plus grand visage : avec plusieurs personnes, rien
    # ne garantit que c'est celle qui a donné son nom.
    if person_count > 1:
        return jsonify({'success': False,
                        'message': f'{person_count} personnes à l\'écran : '
                                   'une seule doit être visible pour la capture.'}), 409

    success, message = face_recognizer.enroll_person_average(name, [frame])

    if success:
        _publish('enrolled', name=name)
        with state.lock:
            state.total_known = len(face_recognizer.known_names)
        return jsonify({'success': True, 'message': message,
                        'total_known': state.total_known}), 200

    return jsonify({'success': False, 'message': message}), 422


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
        _publish('renamed', name=new_name, old_name=name)
    return jsonify({'success': status == 'ok', 'message': message}), _PEOPLE_STATUS[status]


@app.route('/api/people/<name>', methods=['DELETE'])
def api_delete_person(name):
    """Supprime photos, entrées et historique. Réponse : 200 | 400 | 404 — { success, message }"""
    status, message = face_recognizer.delete_person(name)
    if status == 'ok':
        # Droit à l'effacement : l'historique de présence part avec la personne.
        presence_log.forget(name)
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

    # Mediapipe n'est pas réentrant : une seule mesure à la fois.
    if not _bench_lock.acquire(blocking=False):
        return jsonify({'success': False, 'message': 'Une mesure est déjà en cours.'}), 409

    try:
        mediapipe_est, keypoint_est = _bench_get_estimators()
        mp_times, kp_times, spans = [], [], []
        confusion: dict = {}
        agree = compared = mp_silent = kp_silent = 0

        for _ in range(samples):
            with state.lock:
                frame = state.bench_frame
                persons = list(state.bench_persons)

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
    État courant.

    Réponse : { currently_present[], fps, total_known,
                tracks: [{ track_id, name, bbox: [x1, y1, x2, y2], pose }],
                frame_size: [largeur, hauteur] | null }

    `tracks` ne contient que les personnes visibles sur la dernière image, en
    pixels de cette image : la page les superpose au flux pour cliquer dessus.
    """
    return jsonify(_status_payload())


def _status_payload():
    with state.lock:
        poses = {p['track_id']: p['pose'] for p in state.currently_present}
        tracks = [{'track_id': str(p.track_id), 'name': p.name,
                   'bbox': [int(v) for v in p.body_bbox], 'pose': poses.get(p.track_id)}
                  for p in state.bench_persons]
        frame = state.bench_frame
        return {
            'currently_present': list(state.currently_present),
            'fps': state.fps,
            'total_known': state.total_known,
            'tracks': tracks,
            'frame_size': [frame.shape[1], frame.shape[0]] if frame is not None else None,
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
    def stream():
        subscription = events.subscribe()
        try:
            # Reconnexion automatique du navigateur après 2 s si le flux coupe.
            yield 'retry: 2000\n\n'
            next_status = 0.0
            while state.running:
                now = time.monotonic()
                if now >= next_status:
                    yield f"data: {json.dumps({'type': 'status', **_status_payload()})}\n\n"
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
    with state.lock:
        fps, video_clients = state.fps, state.stream_clients
        frame = state.bench_frame
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
            'source': 'webcam' if config.USE_LOCAL_CAM else config.REMOTE_SOURCE,
            'frame_size': [frame.shape[1], frame.shape[0]] if frame is not None else None,
        },
        'clients': {'video': video_clients, 'events': events.subscriber_count},
    })


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

    with state.lock:
        frame = state.bench_frame
        person = next((p for p in state.bench_persons if str(p.track_id) == track_id), None)
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
    sessions = presence_log.sessions(
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
    return jsonify({'sessions': sessions, 'names': presence_log.names()})


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
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    process_thread = threading.Thread(target=processing_loop, daemon=True)
    camera_thread.start()
    process_thread.start()
    _persist_secret_key()
    if not auth_enabled():
        logger.warning("Accès NON protégé : définir ADMIN_PASSWORD_HASH ou ADMIN_PASSWORD "
                       "dans .env pour exiger une connexion.")
    logger.info("Démarrage sur http://%s:%d (waitress, %d threads)",
                config.FLASK_HOST, config.FLASK_PORT, config.SERVER_THREADS)

    # waitress plutôt que le serveur de développement de Flask, qui n'est pas
    # fait pour tourner en continu. Il fonctionne sous Linux comme sous Windows.
    from waitress import serve

    try:
        serve(app, host=config.FLASK_HOST, port=config.FLASK_PORT,
              threads=config.SERVER_THREADS, ident="VisionCam")
    finally:
        state.running = False
        # Ferme les sessions en cours à leur dernière heure vue.
        presence_log.close_all()
