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


def draw_person(display_frame, person, pose, frame_count):
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
                draw_person(display_frame, person, pose, self.frame_count)
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
