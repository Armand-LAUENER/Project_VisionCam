"""
face_body_tracker.py — Pipeline Body-First avec keypoints YOLO-Pose.

Architecture :
  1. YOLOv8-Pose → détection corps + keypoints squelette COCO (17 pts)
  2. DeepSORT    → tracking corps + ReID apparence (chaque frame)
                   Les keypoints sont transmis via `others` du tuple DeepSORT.
  3. InsightFace → crop dynamique centré sur le Nez (keypoint 0)
                   Skip automatique si nez non visible (dos tourné → gain FPS).
  4. Association → face_center ↔ nose_keypoint (proximité euclidienne)
  5. Persistance → identity_map[track_id] conserve le nom même sans visage visible
"""

from __future__ import annotations

import logging
import numpy as np
from collections import deque
from typing import Optional

from ultralytics import YOLO

import config
from core.face_recognition import FaceRecognizer
from core.tracker_backends import build_body_tracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

class TrackedPerson:
    """
    Résultat final pour une personne trackée à un instant donné.
    Contient les infos de position (corps) et d'identité (visage).
    """
    __slots__ = ('track_id', 'body_bbox', 'name', 'confidence', 'last_face_frame',
                 'pose_kps')

    def __init__(self, track_id: int, body_bbox: list[int],
                 name: str = "Inconnu", confidence: float = 0.0,
                 last_face_frame: int = -1, pose_kps: list | None = None) -> None:
        self.track_id = track_id
        self.body_bbox = body_bbox
        self.name = name
        self.confidence = confidence
        self.last_face_frame = last_face_frame
        # Keypoints COCO 0-6 du dernier match YOLO, pour POSE_SOURCE="yolo".
        self.pose_kps = pose_kps


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

class FaceBodyTracker:
    """
    Orchestre le pipeline Body-First avec keypoints YOLO-Pose.

    Usage:
        tracker = FaceBodyTracker(FaceRecognizer(...))
        for frame_count, frame in enumerate(camera):
            persons = tracker.update(frame, frame_count)
            for p in persons:
                draw_box(frame, p.body_bbox, p.name)
    """

    # Lissage temporel : consensus requis avant de confirmer une identité.
    # VOTE_WINDOW=3, majorité = 2/3 → ≈ 100ms à 30 FPS.
    VOTE_WINDOW = 3

    def __init__(self, face_recognizer: FaceRecognizer) -> None:
        logger.info("Chargement YOLO-Pose (%s)...", config.YOLO_MODEL)
        self.yolo = YOLO(config.YOLO_MODEL)
        logger.info("YOLO-Pose chargé")

        # deep_sort_realtime ou deepsort-rs selon config.TRACKER_BACKEND.
        self.body_tracker = build_body_tracker()

        self.face_recognizer = face_recognizer

        # { track_id: { 'name': str, 'confidence': float, 'last_face_frame': int } }
        self._identity_map: dict[int, dict] = {}

        # { track_id: deque([(name, confidence), ...], maxlen=VOTE_WINDOW) }
        self._vote_buffer: dict[int, deque] = {}

        # { track_id: (nose_x, nose_y, nose_conf) }
        # Mis à jour par IoU matching YOLO↔DeepSORT chaque frame.
        # Contourne le fait que track.others n'est pas fiable dans DeepSORT 1.3.x.
        self._nose_map: dict[int, tuple] = {}

        # { track_id: list[(x, y, conf) × 5] } — keypoints COCO 0-4 (nose, eyes, ears)
        self._face_kps_map: dict[int, list] = {}

        # { track_id: list[(x, y, conf) × 7] } — COCO 0-6, ajoute les épaules.
        # Alimente l'estimation d'orientation quand POSE_SOURCE="yolo".
        self._pose_kps_map: dict[int, list] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Point d'entrée public
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, frame_count: int) -> list[TrackedPerson]:
        """Pipeline complet pour une frame."""

        # Étape 1 : Détection YOLO-Pose (corps + keypoints)
        body_detections = self._detect_bodies(frame)

        # Étape 2 : Tracking DeepSORT
        raw_tracks = self.body_tracker.update(body_detections, frame)
        active_tracks = [t for t in raw_tracks if t.is_confirmed()]

        # Étape 2b : Mise à jour du nose_map par IoU matching YOLO↔tracks
        # track.others n'est pas fiable dans DeepSORT 1.3.x → on maintient
        # notre propre dict { track_id: (nx, ny, nc) }.
        self._update_nose_map(active_tracks, body_detections)

        # Étapes 3 & 4 : Reconnaissance faciale intelligente (cadencée)
        if frame_count % config.FACE_RECOGNITION_SKIP == 0 and active_tracks:

            tracks_to_recognize = []
            for track in active_tracks:
                identity = self._identity_map.get(track.track_id, {})
                name = identity.get('name', 'Inconnu')
                last_face_frame = identity.get('last_face_frame', -1)

                if name == 'Inconnu' or (frame_count - last_face_frame > config.FACE_FRESHNESS_FRAMES):
                    tracks_to_recognize.append(track)

            # Max 2 InsightFace/frame pour garantir les FPS
            tracks_to_recognize = tracks_to_recognize[:2]

            if tracks_to_recognize:
                faces = self._recognize_faces_for_tracks(frame, tracks_to_recognize)
                self._associate_faces_to_tracks(active_tracks, faces, frame_count)

        # Étape 5 : Construire les résultats
        results = self._build_results(active_tracks)

        # Nettoyage : purger les entrées des tracks disparus
        active_ids = {t.track_id for t in active_tracks}
        self._identity_map = {tid: v for tid, v in self._identity_map.items() if tid in active_ids}
        self._vote_buffer  = {tid: v for tid, v in self._vote_buffer.items()  if tid in active_ids}
        # _nose_map et _face_kps_map sont déjà purgés dans _update_nose_map

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 2b — Mise à jour du nose_map (bypass track.others)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_nose_map(self, active_tracks: list, body_detections: list) -> None:
        """
        Associe chaque track confirmé à la détection YOLO la plus proche (IoU)
        et met à jour self._nose_map[track_id] avec le keypoint nez correspondant.

        Nécessaire car track.others n'est pas propagé de façon fiable par
        deep_sort_realtime 1.3.x pour les tracks confirmés.
        """
        for track in active_tracks:
            tx1, ty1, tx2, ty2 = track.to_ltrb()
            best_iou = 0.0
            best_nose = None
            best_face_kps = None
            best_pose_kps = None

            for det_bbox, _conf, _cls, det_others in body_detections:
                dx, dy, dw, dh = det_bbox
                dx2, dy2 = dx + dw, dy + dh

                ix1 = max(tx1, dx);  iy1 = max(ty1, dy)
                ix2 = min(tx2, dx2); iy2 = min(ty2, dy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0.0:
                    continue
                union = (tx2 - tx1) * (ty2 - ty1) + dw * dh - inter
                iou = inter / union if union > 0.0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_nose = det_others.get('nose')
                    best_face_kps = det_others.get('face_kps')
                    best_pose_kps = det_others.get('pose_kps')

            if best_iou > 0.3:
                if best_nose is not None:
                    self._nose_map[track.track_id] = best_nose
                if best_face_kps is not None:
                    self._face_kps_map[track.track_id] = best_face_kps
                if best_pose_kps is not None:
                    self._pose_kps_map[track.track_id] = best_pose_kps
            # Si pas de match : on conserve les valeurs précédentes (track coasté)

        # Purger les tracks disparus
        active_ids = {t.track_id for t in active_tracks}
        self._nose_map    = {tid: v for tid, v in self._nose_map.items()    if tid in active_ids}
        self._face_kps_map = {tid: v for tid, v in self._face_kps_map.items() if tid in active_ids}
        self._pose_kps_map = {tid: v for tid, v in self._pose_kps_map.items() if tid in active_ids}

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 1 — Détection YOLO-Pose
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_bodies(self, frame: np.ndarray) -> list[tuple]:
        """
        Détecte les personnes avec YOLOv8-Pose et formate les résultats pour DeepSORT.

        Chaque détection est un 4-tuple : ([x, y, w, h], conf, 'person', others).
        `others` contient le keypoint Nez sous la forme (x, y, conf) ou None.
        DeepSORT stocke `others` dans track.others → disponible dans les étapes suivantes.

        Returns:
            Liste de 4-tuples au format DeepSORT avec keypoints embarqués.
        """
        yolo_results = self.yolo.predict(
            frame,
            classes=[0],
            conf=config.YOLO_CONF_THRESHOLD,
            device=0,
            verbose=False,
        )

        detections = []
        for result in yolo_results:
            kps_data = result.keypoints.data if result.keypoints is not None else None

            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Extraire les keypoints faciaux COCO 0-4 : nose, left_eye, right_eye, left_ear, right_ear
                nose = None
                face_kps = None
                pose_kps = None
                if kps_data is not None and i < len(kps_data):
                    kp = kps_data[i]        # Tensor (17, 3) : (x, y, conf) par keypoint
                    nx, ny, nc = float(kp[0][0]), float(kp[0][1]), float(kp[0][2])
                    nose = (nx, ny, nc)
                    face_kps = [(float(kp[j][0]), float(kp[j][1]), float(kp[j][2])) for j in range(5)]
                    # 0-6 : les 5 précédents + épaules gauche/droite, pour l'orientation.
                    pose_kps = [(float(kp[j][0]), float(kp[j][1]), float(kp[j][2])) for j in range(7)]

                others = {'nose': nose, 'face_kps': face_kps, 'pose_kps': pose_kps}
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person', others))

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 3 — Reconnaissance faciale par crop dynamique centré sur le Nez
    # ─────────────────────────────────────────────────────────────────────────

    def _recognize_faces_for_tracks(
            self,
            frame: np.ndarray,
            tracks_to_recognize: list,
    ) -> list[dict]:
        """
        Pour chaque track, extrait le keypoint Nez depuis track.others et découpe
        un carré dynamique centré sur ce point pour InsightFace.

        Avantages vs. crop "60% supérieur" :
          - Fonctionne quelle que soit la distance à la caméra.
          - Skip automatique si nez absent (dos tourné) → gain FPS immédiat.
          - Crop plus petit et plus précis → InsightFace plus rapide.

        Returns:
            Liste de visages détectés avec bbox remises aux coordonnées globales
            et `source_track_id` indiquant le track d'origine.
        """
        all_faces = []
        h_img, w_img = frame.shape[:2]

        for track in tracks_to_recognize:
            bx1, by1, bx2, by2 = [int(v) for v in track.to_ltrb()]
            body_height = max(1, by2 - by1)
            half = max(config.POSE_CROP_HALF_SIZE, int(body_height * 0.25))

            # Centroïde des keypoints faciaux COCO 0-4 visibles (nez, yeux, oreilles).
            # Résistant aux lunettes et aux occlusions partielles : 1 keypoint suffit.
            crop_center = None
            face_kps = self._face_kps_map.get(track.track_id)
            if face_kps:
                visible = [(x, y) for x, y, c in face_kps if c >= config.POSE_NOSE_CONF_THRESHOLD]
                if len(visible) >= config.POSE_FACE_KP_MIN_VISIBLE:
                    cx = int(sum(x for x, y in visible) / len(visible))
                    cy = int(sum(y for x, y in visible) / len(visible))
                    crop_center = (cx, cy)
                    if len(visible) < 5:
                        logger.debug("track #%s : %d/5 keypoints faciaux visibles → centroïde",
                                     track.track_id, len(visible))

            # Fallback : aucun keypoint visible (personne de dos, très loin, très occulté)
            # → on estime la position de la tête depuis le haut de la bbox corps.
            if crop_center is None:
                cx = (bx1 + bx2) // 2
                cy = by1 + int(body_height * 0.15)
                crop_center = (cx, cy)
                logger.debug("track #%s : aucun keypoint facial → fallback body-top", track.track_id)

            cx, cy = crop_center
            crop_x1 = max(0, cx - half)
            crop_y1 = max(0, cy - half)
            crop_x2 = min(w_img, cx + half)
            crop_y2 = min(h_img, cy + half)

            if crop_x2 - crop_x1 < 20 or crop_y2 - crop_y1 < 20:
                continue

            head_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            try:
                crop_faces = self.face_recognizer.detect_and_recognize(head_crop)
            except Exception as e:
                logger.warning("InsightFace erreur track #%s : %s: %s",
                               track.track_id, type(e).__name__, e)
                continue

            for face in crop_faces:
                fx1, fy1, fx2, fy2 = face['bbox']
                # Remettre les coordonnées du crop à l'échelle de l'image globale
                face['bbox'] = [fx1 + crop_x1, fy1 + crop_y1, fx2 + crop_x1, fy2 + crop_y1]
                # Association directe : on sait déjà à quel track ce visage appartient
                face['source_track_id'] = track.track_id
                all_faces.append(face)

        return all_faces

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 4 — Association Visage → Corps (basée sur le nez)
    # ─────────────────────────────────────────────────────────────────────────

    def _associate_faces_to_tracks(
            self,
            active_tracks: list,
            faces: list[dict],
            frame_count: int,
    ) -> None:
        """
        Associe chaque visage reconnu à son track et met à jour _identity_map.

        Priorité d'association :
          1. source_track_id (défini au crop → association directe, sans calcul).
          2. Proximité euclidienne face_center ↔ nose_keypoint (fallback).

        Règles absolues :
          - 'Inconnu' ne remplace JAMAIS une identité établie.
          - Confiance < RECOGNITION_THRESHOLD → rejeté.
          - Anti-clonage : une identité ne peut appartenir qu'à un seul corps.
          - Vote buffer : consensus requis avant confirmation.
        """
        active_track_map = {t.track_id: t for t in active_tracks}

        for face in faces:
            if face['name'] == 'Inconnu':
                continue
            if face['confidence'] < config.RECOGNITION_THRESHOLD:
                continue

            # Association directe via source_track_id (chemin nominal)
            best_track_id = face.get('source_track_id')
            if best_track_id not in active_track_map:
                # Fallback géométrique si le track a disparu entre le crop et l'association
                best_track_id = self._find_containing_track(face['bbox'], active_tracks)

            if best_track_id is None:
                continue

            # ── Vote buffer ────────────────────────────────────────────────────
            if best_track_id not in self._vote_buffer:
                self._vote_buffer[best_track_id] = deque(maxlen=self.VOTE_WINDOW)
            self._vote_buffer[best_track_id].append((face['name'], face['confidence']))

            votes: dict[str, list[float]] = {}
            for name, conf in self._vote_buffer[best_track_id]:
                votes.setdefault(name, []).append(conf)

            top_name = max(votes, key=lambda n: len(votes[n]))
            top_count = len(votes[top_name])
            if top_count < (self.VOTE_WINDOW + 1) // 2:
                continue  # pas encore de consensus

            avg_confidence = sum(votes[top_name]) / top_count
            new_name = top_name

            # ── Anti-clonage ────────────────────────────────────────────────
            for old_tid, identity in list(self._identity_map.items()):
                if old_tid != best_track_id and identity.get('name') == new_name:
                    self._identity_map[old_tid] = {
                        'name': 'Inconnu', 'confidence': 0.0, 'last_face_frame': -1
                    }
                    logger.warning(
                        "Correction usurpation : '%s' passe du corps #%d au corps #%d",
                        new_name, old_tid, best_track_id,
                    )

            self._identity_map[best_track_id] = {
                'name': new_name,
                'confidence': avg_confidence,
                'last_face_frame': frame_count,
            }

    def _find_containing_track(
            self,
            face_bbox: list[int],
            active_tracks: list,
    ) -> Optional[int]:
        """
        Fallback d'association : retourne le track dont le nez est le plus proche
        du centre de la bbox du visage.

        Seuil de proximité : 80% de la largeur du corps (tolérant aux imprécisions
        YOLO vs InsightFace sur la localisation du nez).
        """
        face_cx = (face_bbox[0] + face_bbox[2]) / 2
        face_cy = (face_bbox[1] + face_bbox[3]) / 2

        best_track_id = None
        best_distance = float('inf')

        for track in active_tracks:
            nose = self._nose_map.get(track.track_id)
            if nose is None or nose[2] < config.POSE_NOSE_CONF_THRESHOLD:
                continue

            nose_x, nose_y, _ = nose
            dist = ((face_cx - nose_x) ** 2 + (face_cy - nose_y) ** 2) ** 0.5

            bx1, by1, bx2, by2 = [int(v) for v in track.to_ltrb()]
            body_width = max(1, bx2 - bx1)
            max_dist = body_width * 0.8

            if dist < max_dist and dist < best_distance:
                best_distance = dist
                best_track_id = track.track_id

        return best_track_id

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 5 — Construction du résultat
    # ─────────────────────────────────────────────────────────────────────────

    def _build_results(self, active_tracks: list) -> list[TrackedPerson]:
        """Assemble les TrackedPerson depuis les tracks actifs + l'identity_map."""
        results = []
        for track in active_tracks:
            tid = track.track_id
            x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
            identity = self._identity_map.get(tid, {})

            results.append(TrackedPerson(
                track_id=tid,
                body_bbox=[x1, y1, x2, y2],
                name=identity.get('name', 'Inconnu'),
                confidence=identity.get('confidence', 0.0),
                last_face_frame=identity.get('last_face_frame', -1),
                pose_kps=self._pose_kps_map.get(tid),
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Libère les ressources."""
        self._identity_map.clear()
        self._vote_buffer.clear()
        self._face_kps_map.clear()
        logger.info("Ressources libérées")
