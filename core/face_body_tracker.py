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

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
from core.face_recognition import FaceRecognizer


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
        print(f"[BODY] Chargement YOLO ({config.YOLO_MODEL})...")
        self.yolo = YOLO(config.YOLO_MODEL)
        print("[BODY] YOLO chargé ✅")

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
        Pipeline complet pour une frame.

        Args:
            frame       : image BGR (numpy array)
            frame_count : numéro de la frame depuis le démarrage

        Returns:
            Liste des TrackedPerson actifs (tracks confirmés uniquement).
        """
        # ── Étape 1 : Détection des corps (YOLO) ────────────────────────────
        body_detections = self._detect_bodies(frame)

        # ── Étape 2 : Tracking des corps (DeepSORT) ─────────────────────────
        # DeepSORT retourne des Track objects avec track_id stable entre frames.
        raw_tracks = self.body_tracker.update_tracks(body_detections, frame=frame)
        active_tracks = [t for t in raw_tracks if t.is_confirmed()]

        # ── Étape 3 & 4 : Reco faciale + association (toutes les N frames) ──
        if frame_count % config.FACE_RECOGNITION_SKIP == 0 and active_tracks:
            body_rois = [t.to_ltrb() for t in active_tracks]
            faces = self._recognize_faces_in_rois(frame, body_rois)
            self._associate_faces_to_tracks(active_tracks, faces, frame_count)

        # ── Étape 5 : Construire la liste de résultats ───────────────────────
        results = self._build_results(active_tracks)

        # Nettoyage : retirer les identités des tracks qui ne sont plus actifs.
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
    # Étape 3 — Reconnaissance faciale filtrée
    # ─────────────────────────────────────────────────────────────────────────

    def _recognize_faces_in_rois(
        self,
        frame: np.ndarray,
        body_rois: list[list[float]],
    ) -> list[dict]:
        """
        Lance InsightFace sur la frame entière (une seule passe GPU),
        puis filtre les visages dont le centre n'appartient à aucun corps connu.

        Pourquoi une passe globale plutôt que des crops par corps ?
        Cropper introduit du padding, de l'upscaling, et N passes GPU distinctes.
        Une passe + filtre géométrique est plus rapide dès que N > 2 corps.

        Returns:
            Sous-ensemble des visages détectés, filtrés par appartenance corps.
        """
        all_faces = self.face_recognizer.detect_and_recognize(frame)

        if not body_rois or not all_faces:
            return all_faces

        filtered = []
        for face in all_faces:
            if self._face_belongs_to_any_body(face['bbox'], body_rois):
                filtered.append(face)

        return filtered

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

        Stratégie : "gate de confiance + dernière observation gagne"
            1. On rejette d'abord les détections trop faibles (< RECOGNITION_THRESHOLD) :
               un visage de dos ou flou ne doit pas écraser une bonne identification.
            2. Parmi les détections valides, la plus récente remplace toujours l'ancienne :
               buffalo_l est fiable ; la dernière frame a plus de chances d'être bien éclairée
               que la première observation (qui pouvait être prise à un mauvais angle).

        Règles absolues :
            - 'Inconnu' ne remplace JAMAIS une identité établie (filtre en amont).
            - Une confiance < RECOGNITION_THRESHOLD ne met jamais à jour la map.
        """
        for face in faces:
            # Inconnu ne doit jamais écraser une identité établie.
            if face['name'] == 'Inconnu':
                continue

            # Gate de confiance : rejeter les détections trop faibles.
            # Le seuil est le même que celui de FaceRecognizer._identify.
            if face['confidence'] < config.RECOGNITION_THRESHOLD:
                continue

            best_track_id = self._find_containing_track(face['bbox'], active_tracks)

            if best_track_id is not None:
                self._identity_map[best_track_id] = {
                    'name': face['name'],
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
        print("[BODY] Ressources libérées")
