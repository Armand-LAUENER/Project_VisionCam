"""
face_body_tracker.py — Pipeline Body-First avec association Visage/Corps.

Architecture :
  1. YOLOv8     → détection corps (classe 'person', chaque frame)
  2. DeepSORT   → tracking corps + ReID apparence (chaque frame)
  3. InsightFace → reconnaissance visages (toutes les FACE_RECOGNITION_SKIP frames)
  4. Association géométrique → face_center ∈ moitié supérieure du body_bbox
  5. Persistance → identity_map[track_id] conserve le nom même sans visage visible
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
from core.face_recognition import FaceRecognizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackedPerson:
    """
    Résultat final pour une personne trackée à un instant donné.
    Contient à la fois les infos de position (corps) et d'identité (visage).
    """
    track_id: int
    body_bbox: list[int]          # [x1, y1, x2, y2] du corps (YOLO + DeepSORT)
    name: str = "Inconnu"
    confidence: float = 0.0
    last_face_frame: int = -1     # Frame où le visage a été vu pour la dernière fois


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

class FaceBodyTracker:
    """
    Orchestre le pipeline Body-First.

    Usage:
        face_recognizer = FaceRecognizer(...)
        tracker = FaceBodyTracker(face_recognizer)

        for frame_count, frame in enumerate(camera):
            persons = tracker.update(frame, frame_count)
            for p in persons:
                draw_box(frame, p.body_bbox, p.name)
    """

    # Fraction maximale de la hauteur du corps depuis le haut dans laquelle
    # on accepte un visage. 0.6 = les 60% supérieurs (tête + épaules).
    # Au-delà = torse ou jambes, ce n'est pas un visage.
    FACE_IN_BODY_VERTICAL_RATIO = 0.6

    def __init__(self, face_recognizer: FaceRecognizer) -> None:
        # ── Détecteur de corps ──────────────────────────────────────────────
        logger.info("Chargement YOLO (%s)...", config.YOLO_MODEL)
        self.yolo = YOLO(config.YOLO_MODEL)
        logger.info("YOLO chargé")

        # ── Tracker de corps avec ReID apparence ────────────────────────────
        self.body_tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            embedder=config.DEEPSORT_EMBEDDER,
            embedder_gpu=config.DEEPSORT_EMBEDDER_GPU,
        )

        # ── Reconnaissance faciale (réutilisée depuis l'architecture existante)
        self.face_recognizer = face_recognizer

        # ── Carte d'identité persistante ────────────────────────────────────
        # { track_id: { 'name': str, 'confidence': float, 'last_face_frame': int } }
        self._identity_map: dict[int, dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Point d'entrée public
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, frame_count: int) -> list[TrackedPerson]:
        """
        Pipeline complet pour une frame avec optimisation drastique des FPS.
        """
        # ── Étape 1 : Détection des corps (YOLO) ────────────────────────────
        body_detections = self._detect_bodies(frame)

        # ── Étape 2 : Tracking des corps (DeepSORT) ─────────────────────────
        raw_tracks = self.body_tracker.update_tracks(body_detections, frame=frame)
        active_tracks = [t for t in raw_tracks if t.is_confirmed()]

        # ── Étape 3 & 4 : Reco faciale intelligente ─────────────────────────
        if frame_count % config.FACE_RECOGNITION_SKIP == 0 and active_tracks:

            # OPTIMISATION : On ne sélectionne QUE les corps qui ont besoin d'être identifiés
            tracks_to_recognize = []
            for track in active_tracks:
                identity = self._identity_map.get(track.track_id, {})
                name = identity.get('name', 'Inconnu')
                last_face_frame = identity.get('last_face_frame', -1)

                # S'il est inconnu, OU si on n'a pas revérifié son visage depuis FACE_FRESHNESS_FRAMES
                if name == 'Inconnu' or (frame_count - last_face_frame > config.FACE_FRESHNESS_FRAMES):
                    tracks_to_recognize.append(track)

            # OPTIMISATION 2 : On limite à 1 ou 2 analyses max par frame pour garantir les FPS
            # Si on a 5 inconnus, les autres seront analysés aux frames suivantes !
            tracks_to_recognize = tracks_to_recognize[:2]

            if tracks_to_recognize:
                body_rois = [t.to_ltrb() for t in tracks_to_recognize]
                faces = self._recognize_faces_in_rois(frame, body_rois)
                # On passe toujours active_tracks complet pour l'association géométrique
                self._associate_faces_to_tracks(active_tracks, faces, frame_count)

        # ── Étape 5 : Construire la liste de résultats ───────────────────────
        results = self._build_results(active_tracks)

        # Nettoyage : retirer les identités des tracks qui ne sont plus actifs
        active_ids = {t.track_id for t in active_tracks}
        self._identity_map = {
            tid: v for tid, v in self._identity_map.items()
            if tid in active_ids
        }

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 1 — Détection YOLO
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_bodies(self, frame: np.ndarray) -> list[tuple]:
        """
        Détecte les personnes et formate les résultats pour DeepSORT.

        Returns:
            Liste de tuples ([x1, y1, w, h], confidence, 'person').
            DeepSORT attend le format ltwh (left-top-width-height).
        """
        yolo_results = self.yolo.predict(
            frame,
            classes=[0],                        # Classe 0 = 'person' (COCO)
            conf=config.YOLO_CONF_THRESHOLD,
            device=0,
            verbose=False,
        )

        detections = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # DeepSORT attend [x_left, y_top, width, height]
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 3 — Reconnaissance faciale par "Zoom & Crop"
    # ─────────────────────────────────────────────────────────────────────────

    def _recognize_faces_in_rois(
            self,
            frame: np.ndarray,
            body_rois: list[list[float]],
    ) -> list[dict]:
        """
        Découpe la partie supérieure de chaque corps (Crop) et l'envoie à InsightFace.
        Permet un "zoom" artificiel qui améliore drastiquement la reconnaissance à distance.

        Returns:
            Liste de tous les visages détectés dans ces zones, remis aux coordonnées globales.
        """
        all_faces = []
        h_img, w_img = frame.shape[:2]

        for body_bbox in body_rois:
            bx1, by1, bx2, by2 = [int(v) for v in body_bbox]

            # 1. Calculer la limite basse du crop (les 60% supérieurs du corps)
            crop_y_bottom = int(by1 + (by2 - by1) * self.FACE_IN_BODY_VERTICAL_RATIO)

            # 2. Sécuriser les bordures (au cas où YOLO déborde de l'image)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, crop_y_bottom = min(w_img, bx2), min(h_img, crop_y_bottom)

            # Si la boîte est invalide ou microscopique, on ignore
            if bx2 - bx1 < 20 or crop_y_bottom - by1 < 20:
                continue

            # 3. Découper (Cropper) l'image : on ne garde que la tête/épaules
            head_crop = frame[by1:crop_y_bottom, bx1:bx2]

            # 4. Lancer InsightFace sur ce "zoom"
            crop_faces = self.face_recognizer.detect_and_recognize(head_crop)

            # 5. Ajuster les coordonnées (bbox) pour les ramener à l'échelle de l'image entière
            for face in crop_faces:
                fx1, fy1, fx2, fy2 = face['bbox']

                # Le visage a été trouvé dans le crop, il faut rajouter l'offset du crop (bx1, by1)
                # pour que les rectangles s'affichent au bon endroit sur l'écran final !
                face['bbox'] = [fx1 + bx1, fy1 + by1, fx2 + bx1, fy2 + by1]

                all_faces.append(face)

        # Plus besoin de vérifier si le visage appartient au corps,
        # puisqu'on a forcé l'IA à chercher DEDANS !
        return all_faces

    def _face_belongs_to_any_body(

        self,
        face_bbox: list[int],
        body_rois: list[list[float]],
        ) -> bool:
        """
        Retourne True si le centre du visage appartient à la zone
        supérieure d'au moins un corps connu.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2

        for body_bbox in body_rois:
            bx1, by1, bx2, by2 = body_bbox
            # Limite verticale : on accepte le visage dans les 60% supérieurs du corps.
            upper_limit = by1 + (by2 - by1) * self.FACE_IN_BODY_VERTICAL_RATIO

            if bx1 <= face_cx <= bx2 and by1 <= face_cy <= upper_limit:
                return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 4 — Association géométrique Visage → Corps
    # ─────────────────────────────────────────────────────────────────────────

    def _associate_faces_to_tracks(
            self,
            active_tracks: list,
            faces: list[dict],
            frame_count: int,
    ) -> None:
        """
        Pour chaque visage reconnu (nom != 'Inconnu'), trouve le track corps
        qui le contient et met à jour _identity_map.

        Règles absolues :
            - 'Inconnu' ne remplace JAMAIS une identité établie (filtre en amont).
            - Une confiance < RECOGNITION_THRESHOLD ne met jamais à jour la map.
            - ANTI-CLONAGE : Une identité ne peut appartenir qu'à un seul corps à la fois.
        """
        for face in faces:
            # Inconnu ne doit jamais écraser une identité établie.
            if face['name'] == 'Inconnu':
                continue

            # Gate de confiance : rejeter les détections trop faibles.
            if face['confidence'] < config.RECOGNITION_THRESHOLD:
                continue

            best_track_id = self._find_containing_track(face['bbox'], active_tracks)

            if best_track_id is not None:
                new_name = face['name']

                # --- LOGIQUE ANTI-CLONAGE ---
                # Si ce nom est déjà attribué à un autre corps actif, on le retire.
                # La détection actuelle (qui vient d'être vue de face à l'instant) fait autorité.
                for old_tid, identity in list(self._identity_map.items()):
                    if old_tid != best_track_id and identity.get('name') == new_name:
                        # Rétrograder l'ancien corps à "Inconnu"
                        self._identity_map[old_tid] = {
                            'name': 'Inconnu',
                            'confidence': 0.0,
                            'last_face_frame': -1
                        }
                        logger.warning(
                            "Correction usurpation : '%s' passe du corps #%d au corps #%d",
                            new_name, old_tid, best_track_id,
                        )

                # Mise à jour du nouveau corps
                self._identity_map[best_track_id] = {
                    'name': new_name,
                    'confidence': face['confidence'],
                    'last_face_frame': frame_count,
                }

    def _find_containing_track(
        self,
        face_bbox: list[int],
        active_tracks: list,
    ) -> Optional[int]:
        """
        Retourne le track_id du corps qui contient géométriquement ce visage.
        Si plusieurs corps candidats, retourne le plus proche (distance des centres).
        Retourne None si aucun corps ne contient ce visage.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2

        best_track_id = None
        best_distance = float('inf')

        for track in active_tracks:
            bx1, by1, bx2, by2 = [int(v) for v in track.to_ltrb()]
            upper_limit = by1 + (by2 - by1) * self.FACE_IN_BODY_VERTICAL_RATIO

            # Le centre du visage doit être dans la zone supérieure du corps.
            if not (bx1 <= face_cx <= bx2 and by1 <= face_cy <= upper_limit):
                continue

            body_cx = (bx1 + bx2) / 2
            body_cy = (by1 + by2) / 2
            distance = ((face_cx - body_cx) ** 2 + (face_cy - body_cy) ** 2) ** 0.5

            if distance < best_distance:
                best_distance = distance
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
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Libère les ressources (appeler à l'arrêt de l'application)."""
        self._identity_map.clear()
        logger.info("Ressources libérées")
