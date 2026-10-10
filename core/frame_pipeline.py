"""
frame_pipeline.py — Traitement des images d'une caméra (roadmap 2.2).

Image → FrameResult : tracker de corps et reconnaissance (FaceBodyTracker),
orientation, présents (avec PRESENCE_TIMEOUT), FPS et dessin. Sans état
partagé avec l'application : tourne dans un thread de app.py ou dans un
processus de caméras (core/camera_worker.py).
"""

from __future__ import annotations

import csv
import logging
import time
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def build_pose_estimator():
    """Estimateur d'orientation désigné par POSE_SOURCE."""
    if config.POSE_SOURCE == "yolo":
        from core.pose_from_keypoints import KeypointPoseEstimator
        return KeypointPoseEstimator()
    if config.POSE_SOURCE == "mediapipe":
        from core.pose_estimation import PoseEstimator
        return PoseEstimator()
    raise ValueError(
        f"POSE_SOURCE invalide : {config.POSE_SOURCE!r} — attendu 'mediapipe' ou 'yolo'."
    )


def estimate_pose_safe(pose_estimator, frame, person):
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


# Style « Person of Interest » (la Machine) : crochets aux coins, bords en
# pointillés, onglet d'état et panneau d'informations en majuscules. Codes de la
# série : blanc pour une personne quelconque, jaune pour ceux que la Machine
# connaît, noir à coins jaunes pour l'« interface analogique ».
POI_YELLOW = (0, 215, 255)
POI_WHITE = (235, 235, 235)
POI_BLACK = (20, 20, 20)
POI_FONT = cv2.FONT_HERSHEY_SIMPLEX
POI_DASH = 6


def _ascii_upper(text: str) -> str:
    """Majuscules sans accents : les polices Hershey d'OpenCV n'ont que l'ASCII."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().upper()


def _dashed_line(img, p1, p2, color) -> None:
    length = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])))
    for start in range(0, length, 2 * POI_DASH):
        a, b = start / length, min(start + POI_DASH, length) / length
        cv2.line(img, (int(p1[0] + a * (p2[0] - p1[0])), int(p1[1] + a * (p2[1] - p1[1]))),
                 (int(p1[0] + b * (p2[0] - p1[0])), int(p1[1] + b * (p2[1] - p1[1]))),
                 color, 1, cv2.LINE_AA)


def _poi_frame(img, box, corner_color, edge_color) -> None:
    """Bords fins en pointillés et crochets épais aux quatre coins, comme la Machine."""
    x1, y1, x2, y2 = box
    corner = max(8, int(0.2 * min(x2 - x1, y2 - y1)))
    for p1, p2 in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                   ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        _dashed_line(img, p1, p2, edge_color)
    for (cx, cy), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                               ((x2, y2), (-1, -1)), ((x1, y2), (1, -1))):
        cv2.line(img, (cx, cy), (cx + dx * corner, cy), corner_color, 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * corner), corner_color, 3, cv2.LINE_AA)


def _poi_panel(img, x, y, tab: str, lines: list[str], color) -> None:
    """Onglet plein (texte noir) puis lignes sur fond sombre translucide."""
    scale, pad, line_h = 0.45, 5, 16
    widths = [cv2.getTextSize(t, POI_FONT, scale, 1)[0][0] for t in [tab, *lines]]
    width = max(widths) + 2 * pad
    height = line_h + pad + len(lines) * line_h + pad
    h, w = img.shape[:2]
    x = int(min(max(0, x), max(0, w - width)))
    y = int(min(max(0, y), max(0, h - height)))
    roi = img[y + line_h + pad:y + height, x:x + width]
    if roi.size:
        roi[:] = (roi * 0.35).astype(roi.dtype)
    cv2.rectangle(img, (x, y), (x + widths[0] + 2 * pad, y + line_h + pad - 1), color, -1)
    cv2.putText(img, tab, (x + pad, y + line_h - 2), POI_FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x + pad, y + line_h + pad + (i + 1) * line_h - 4),
                    POI_FONT, scale, color, 1, cv2.LINE_AA)


def draw_person(display_frame, person, pose, frame_count, fps: float = 0.0):
    """
    Dessine une TrackedPerson dans le style « Person of Interest ».

    États d'identification :
      - jaune : visage reconnu récemment (< FACE_FRESHNESS_FRAMES) ;
      - noir à coins jaunes : nom connu, visage perdu — suivi par le corps
        (avec le temps écoulé depuis le dernier visage) ;
      - blanc : personne inconnue.
    """
    x1, y1, x2, y2 = person.body_bbox
    frames_since_face = frame_count - person.last_face_frame

    if person.name == 'Inconnu':
        color, edge, tab = POI_WHITE, POI_WHITE, "UNKNOWN"
        status = f"#{person.track_id}"
        name = "UNIDENTIFIED"
    elif frames_since_face <= config.FACE_FRESHNESS_FRAMES:
        color, edge, tab = POI_YELLOW, POI_YELLOW, "IDENTIFIED"
        status = f"#{person.track_id}  FACE {person.confidence * 100:.0f}%"
        name = _ascii_upper(person.name)
    else:
        color, edge, tab = POI_YELLOW, POI_BLACK, "TRACKING"
        since = f"{frames_since_face / fps:.0f}S" if fps > 0 else f"{frames_since_face}F"
        status = f"#{person.track_id}  BODY  LAST FACE {since}"
        name = _ascii_upper(person.name)

    _poi_frame(display_frame, (x1, y1, x2, y2), color, edge)
    lines = [name, status]
    if pose:
        lines.append(f"POSE: {_ascii_upper(str(pose))}")
    # À droite de la boîte, aligné sur le haut ; à gauche si l'image s'arrête.
    panel_x = x2 + 6
    if panel_x + 170 > display_frame.shape[1]:
        panel_x = x1 - 176
    _poi_panel(display_frame, panel_x, y1, tab, lines, color)


def log_tracks(writer, frame_count, persons) -> None:
    """Une ligne par piste visible : image,track_id,x,y,largeur,hauteur,nom."""
    for p in persons:
        x1, y1, x2, y2 = p.body_bbox
        writer.writerow([frame_count, p.track_id, x1, y1, x2 - x1, y2 - y1, p.name])


@dataclass
class FrameResult:
    """Ce que le pipeline d'une caméra produit pour une image.

    Frontière entre le traitement d'image (tracker, pose, dessin) et ce qui en
    découle pour toute l'application (historique, journal d'événements,
    alertes, flux web) : la première partie pourra tourner dans un autre
    processus que la seconde.
    """
    frame: np.ndarray | None          # image brute, pour /api/capture et /bench/pose
    display_frame: np.ndarray | None  # image envoyée au flux MJPEG
    persons: list                # TrackedPerson visibles sur cette image
    present_list: list           # présents, PRESENCE_TIMEOUT compris
    fps: float
    received: float              # réception par le thread caméra (perf_counter)
    frame_start: float           # début du traitement (perf_counter)
    frame_size: tuple[int, int] | None = None  # (largeur, hauteur)
    # Venu d'un processus de caméras (core/camera_worker.py) : les images ne
    # traversent pas, seulement le JPEG du flux s'il a des clients, et les
    # mesures que le thread aurait enregistrées lui-même.
    jpeg: bytes | None = None
    stats: dict | None = None
    camera_id: str = ""


@dataclass
class FrameSnapshot:
    """Dernière image brute d'une caméra et ses personnes : réponse d'un processus
    de caméras à ("frame", …), pour /api/capture et /bench/pose."""
    request_id: int
    camera_id: str
    frame: np.ndarray | None     # None avant la première image
    persons: list


class FramePipeline:
    """Traitement des images d'une caméra : tracker, pose, présents, FPS, dessin.

    Ne touche à aucun état partagé de l'application : les durées des étapes
    passent par `record`, le reste sort dans le FrameResult.
    """

    def __init__(self, camera_id, camera_tracker, record, pose_estimator, tracks_path="",
                 tuning=None):
        self.camera_id = camera_id
        self.tracker = camera_tracker
        self.record = record
        self.pose_estimator = pose_estimator
        # Réglages ajustés en marche (core/adaptive.py), relus à chaque image.
        self.tuning = tuning
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
        if self.tuning is not None:
            self.tracker.recognition_skip = self.tuning.recognition_skip
        try:
            self.persons = self.tracker.update(frame, self.frame_count)
            for stage, seconds in self.tracker.last_timings.items():
                self.record(stage, seconds)
            if self._tracks_writer:
                log_tracks(self._tracks_writer, self.frame_count, self.persons)
        except Exception as e:
            logger.error("[%s] Erreur traitement : %s: %s", self.camera_id, type(e).__name__, e)

        # ── Pose estimation (toutes les FRAME_SKIP frames, sur le crop corps) ─
        if self.frame_count % config.FRAME_SKIP == 0:
            pose_start = time.perf_counter()
            self._pose_cache = {
                p.track_id: estimate_pose_safe(self.pose_estimator, frame, p)
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
                draw_person(display_frame, person, pose, self.frame_count, self.fps)
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
                           frame_start=frame_start, frame_size=(frame.shape[1], frame.shape[0]),
                           camera_id=self.camera_id)

    def close(self):
        self.tracker.release()
        if self._tracks_log:
            self._tracks_log.close()
