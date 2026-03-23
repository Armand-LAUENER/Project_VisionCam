"""
core/tracker.py — Moteur de tracking intelligent.

Architecture du pipeline (par frame) :
  1. InsightFace détecte les visages → reconnaissance via FaceDatabase
  2. Mediapipe Pose détecte les corps → extraction de bboxes du torse
  3. DeepSort assigne des track_id persistants aux corps détectés
  4. Algorithme Hongrois associe chaque visage reconnu à un corps tracké
  5. Persistance : si un visage disparaît mais que le corps reste tracké,
     le nom est maintenu sur la bbox du corps

Règle stricte : les personnes inconnues sont IGNORÉES (pas de bbox, pas de tracking).
"""

import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from scipy.optimize import linear_sum_assignment
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
from core.database import FaceDatabase


# =============================================================================
# UTILITAIRES GÉOMÉTRIQUES
# =============================================================================

def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calcule le centre d'une bounding box.

    Args:
        bbox: (x1, y1, x2, y2) — coins supérieur-gauche et inférieur-droit

    Returns:
        (cx, cy) — centre de la bbox
    """
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def euclidean_distance(
    p1: Tuple[int, int], p2: Tuple[int, int]
) -> float:
    """
    Distance euclidienne entre deux points 2D.

    Args:
        p1: Premier point (x, y)
        p2: Second point (x, y)

    Returns:
        Distance en pixels
    """
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class SmartTracker:
    """
    Moteur d'IA combinant reconnaissance faciale, détection corporelle
    et suivi persistant des personnes connues uniquement.

    Attributs:
        face_db (FaceDatabase): Base d'embeddings faciaux
        deepsort (DeepSort): Tracker multi-objets pour les corps
        mp_pose (mediapipe.solutions.pose.Pose): Détecteur de pose corporelle
        track_identity (dict): Mapping {track_id: nom_personne} persistant
        present_users (dict): {nom_personne: timestamp_dernière_détection}
        all_seen_today (set): Ensemble des noms uniques vus depuis le lancement
    """

    def __init__(self):
        """Initialise tous les composants du pipeline d'IA."""

        print("[TRACKER] ══════════════════════════════════════")
        print("[TRACKER]   Initialisation du moteur d'IA...")
        print("[TRACKER] ══════════════════════════════════════")

        # --- 1. Base de données faciale (InsightFace inclus) ---
        self.face_db = FaceDatabase()

        # --- 2. Référence au modèle InsightFace (réutilisé depuis la DB) ---
        self.face_analyzer = self.face_db.analyzer

        # --- 3. DeepSort — Tracker multi-objets sur GPU ---
        print("[TRACKER] Chargement de DeepSort...")
        self.deepsort = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            embedder=config.DEEPSORT_EMBEDDER,
            half=True,          # Demi-précision pour GPU Nvidia
            bgr=True,           # OpenCV fournit du BGR
            embedder_gpu=True   # Embedder sur GPU
        )
        print("[TRACKER] DeepSort chargé (GPU, half precision).")

        # --- 4. Mediapipe Pose — Détection du torse ---
        print("[TRACKER] Chargement de Mediapipe Pose...")
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,        # 0=light, 1=full, 2=heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[TRACKER] Mediapipe Pose chargé.")

        # --- 5. Dictionnaires de suivi ---

        # Mapping persistant : {track_id (int) : nom_personne (str)}
        # Maintient l'identité même quand le visage disparaît
        self.track_identity: Dict[int, str] = {}

        # Dernière fois qu'une personne a été vue : {nom: timestamp}
        self.present_users: Dict[str, float] = {}

        # Ensemble de tous les noms uniques détectés depuis le lancement
        self.all_seen_today: set = set()

        # --- 6. Paramètres internes ---

        # Distance max (pixels) pour associer un visage à un corps
        self._max_association_distance = 200.0

        # Compteur de frames pour le scheduling de la détection faciale
        self._frame_count = 0

        # Fréquence de détection faciale (1 frame sur N) pour économiser le GPU
        self._face_detection_interval = 3

        print("[TRACKER] ══════════════════════════════════════")
        print("[TRACKER]   Moteur d'IA prêt.")
        print("[TRACKER] ══════════════════════════════════════\n")

    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Traite une frame complète du pipeline.

        Pipeline :
          1. Détection faciale + reconnaissance (toutes les N frames)
          2. Détection corporelle via Mediapipe Pose
          3. Tracking DeepSort sur les corps détectés
          4. Association Hongroise visages↔corps
          5. Persistance des identités + annotation visuelle

        Args:
            frame: Image BGR (numpy array) provenant d'OpenCV

        Returns:
            annotated_frame: Frame avec les annotations visuelles
            current_names: Liste des noms actuellement identifiés
        """
        self._frame_count += 1
        now = time.time()
        h, w = frame.shape[:2]

        # -----------------------------------------------------------------
        # ÉTAPE 1 : Détection et reconnaissance faciale (toutes les N frames)
        # -----------------------------------------------------------------
        recognized_faces = []  # Liste de tuples (nom, bbox_visage, centre_visage)

        if self._frame_count % self._face_detection_interval == 0:
            recognized_faces = self._detect_and_recognize_faces(frame)

        # -----------------------------------------------------------------
        # ÉTAPE 2 : Détection des corps via Mediapipe Pose
        # -----------------------------------------------------------------
        body_bboxes = self._detect_bodies(frame, h, w)

        # -----------------------------------------------------------------
        # ÉTAPE 3 : Tracking DeepSort sur les corps
        # -----------------------------------------------------------------
        tracks = self._update_deepsort(frame, body_bboxes)

        # -----------------------------------------------------------------
        # ÉTAPE 4 : Association Hongroise — visages reconnus ↔ corps trackés
        # -----------------------------------------------------------------
        if recognized_faces and tracks:
            self._associate_faces_to_tracks(recognized_faces, tracks)

        # -----------------------------------------------------------------
        # ÉTAPE 5 : Mise à jour de la présence + nettoyage
        # -----------------------------------------------------------------
        current_names = self._update_presence(tracks, now)

        # -----------------------------------------------------------------
        # ÉTAPE 6 : Annotation visuelle de la frame
        # -----------------------------------------------------------------
        annotated = self._annotate_frame(frame, tracks, now)

        return annotated, current_names

    # =========================================================================
    # ÉTAPE 1 : DÉTECTION & RECONNAISSANCE FACIALE
    # =========================================================================

    def _detect_and_recognize_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int]]]:
        """
        Détecte les visages via InsightFace et tente de les reconnaître.
        Seuls les visages RECONNUS sont retournés (les inconnus sont ignorés).

        Args:
            frame: Image BGR

        Returns:
            Liste de (nom, bbox_visage, centre_visage)
            - bbox_visage : (x1, y1, x2, y2)
            - centre_visage : (cx, cy) — calculé mathématiquement via la bbox
        """
        faces = self.face_analyzer.get(frame)
        recognized = []

        for face in faces:
            # Récupération de l'embedding
            if face.embedding is None:
                continue

            # Tentative de reconnaissance
            name = self.face_db.recognize(face.embedding)

            # RÈGLE STRICTE : ignorer les inconnus
            if name is None:
                continue

            # Extraction de la bbox du visage
            # InsightFace retourne bbox en float [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_bbox = (x1, y1, x2, y2)

            # Centre calculé MATHÉMATIQUEMENT via la bbox (pas de FaceMesh)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            face_center = (cx, cy)

            recognized.append((name, face_bbox, face_center))

        return recognized

    # =========================================================================
    # ÉTAPE 2 : DÉTECTION CORPORELLE VIA MEDIAPIPE POSE
    # =========================================================================

    def _detect_bodies(
        self, frame: np.ndarray, h: int, w: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Utilise Mediapipe Pose pour détecter le torse (épaules + hanches)
        et en déduire une bounding box corporelle.

        Landmarks utilisés :
          - 11 : Épaule gauche
          - 12 : Épaule droite
          - 23 : Hanche gauche
          - 24 : Hanche droite

        On élargit la bbox du torse pour englober le corps entier
        (tête au-dessus, jambes en dessous).

        Args:
            frame: Image BGR
            h, w: Hauteur et largeur de la frame

        Returns:
            Liste de bboxes (x1, y1, x2, y2)
        """
        # Mediapipe attend du RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_frame)

        body_bboxes = []

        if results.pose_landmarks is None:
            return body_bboxes

        landmarks = results.pose_landmarks.landmark

        # Indices des points du torse
        TORSO_INDICES = [11, 12, 23, 24]

        # Vérification que tous les landmarks du torse sont visibles
        torso_visible = all(
            landmarks[i].visibility > 0.5 for i in TORSO_INDICES
        )

        if not torso_visible:
            return body_bboxes

        # Extraction des coordonnées du torse en pixels
        torso_x = [int(landmarks[i].x * w) for i in TORSO_INDICES]
        torso_y = [int(landmarks[i].y * h) for i in TORSO_INDICES]

        # Bbox du torse seul
        tx1, tx2 = min(torso_x), max(torso_x)
        ty1, ty2 = min(torso_y), max(torso_y)

        # Calcul des dimensions du torse
        torso_w = tx2 - tx1
        torso_h = ty2 - ty1

        # Extension de la bbox pour englober le corps complet
        # - Horizontalement : +40% de chaque côté
        # - Vers le haut : +80% de la hauteur du torse (pour la tête)
        # - Vers le bas : +100% de la hauteur du torse (pour les jambes)
        margin_x = int(torso_w * 0.4)
        margin_top = int(torso_h * 0.8)
        margin_bottom = int(torso_h * 1.0)

        x1 = max(0, tx1 - margin_x)
        y1 = max(0, ty1 - margin_top)
        x2 = min(w, tx2 + margin_x)
        y2 = min(h, ty2 + margin_bottom)

        body_bboxes.append((x1, y1, x2, y2))

        return body_bboxes

    # =========================================================================
    # ÉTAPE 3 : MISE À JOUR DEEPSORT
    # =========================================================================

    def _update_deepsort(
        self, frame: np.ndarray, body_bboxes: List[Tuple[int, int, int, int]]
    ) -> list:
        """
        Transmet les détections corporelles à DeepSort pour le tracking.

        DeepSort attend des détections au format :
            [[x1, y1, largeur, hauteur], confidence, class_name]

        Args:
            frame: Image BGR originale (pour l'extraction d'embeddings DeepSort)
            body_bboxes: Liste de (x1, y1, x2, y2)

        Returns:
            Liste de tracks actifs de DeepSort
        """
        # Conversion du format (x1,y1,x2,y2) → (x1,y1,w,h) + confiance
        detections = []
        for (x1, y1, x2, y2) in body_bboxes:
            w = x2 - x1
            h = y2 - y1
            # Format attendu par deep_sort_realtime : ([x1, y1, w, h], confidence, class)
            detections.append(([x1, y1, w, h], 0.95, "person"))

        # Mise à jour du tracker
        tracks = self.deepsort.update_tracks(detections, frame=frame)

        # Filtrage : ne garder que les tracks confirmés
        active_tracks = [t for t in tracks if t.is_confirmed()]

        return active_tracks

    # =========================================================================
    # ÉTAPE 4 : ASSOCIATION HONGROISE (visages ↔ corps)
    # =========================================================================

    def _associate_faces_to_tracks(
        self,
        recognized_faces: List[Tuple[str, Tuple, Tuple[int, int]]],
        tracks: list
    ) -> None:
        """
        Associe chaque visage reconnu à un corps tracké via l'Algorithme Hongrois.

        Matrice de coût : distance euclidienne entre le centre du visage
        et le centre de la bbox du corps tracké.

        Les associations dont la distance dépasse le seuil max sont rejetées.

        Args:
            recognized_faces: Liste de (nom, bbox_visage, centre_visage)
            tracks: Liste de tracks DeepSort confirmés
        """
        n_faces = len(recognized_faces)
        n_tracks = len(tracks)

        if n_faces == 0 or n_tracks == 0:
            return

        # --- Construction de la matrice de coût (distances) ---
        cost_matrix = np.zeros((n_faces, n_tracks), dtype=np.float64)

        # Pré-calcul des centres des tracks
        track_centers = []
        for track in tracks:
            # DeepSort : to_ltrb() retourne [x1, y1, x2, y2]
            ltrb = track.to_ltrb()
            tc = bbox_center((int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])))
            track_centers.append(tc)

        for i, (name, face_bbox, face_center) in enumerate(recognized_faces):
            for j, tc in enumerate(track_centers):
                cost_matrix[i, j] = euclidean_distance(face_center, tc)

        # --- Résolution par l'Algorithme Hongrois ---
        face_indices, track_indices = linear_sum_assignment(cost_matrix)

        # --- Validation des associations ---
        for fi, ti in zip(face_indices, track_indices):
            distance = cost_matrix[fi, ti]

            # Rejet si la distance est trop grande (pas le même individu)
            if distance > self._max_association_distance:
                continue

            name = recognized_faces[fi][0]
            track_id = tracks[ti].track_id

            # Sauvegarde de l'association persistante
            self.track_identity[track_id] = name

    # =========================================================================
    # ÉTAPE 5 : GESTION DE LA PRÉSENCE
    # =========================================================================

    def _update_presence(self, tracks: list, now: float) -> List[str]:
        """
        Met à jour les dictionnaires de présence en fonction des tracks actifs.

        - Si un track a une identité connue → la personne est présente
        - Si un track n'a pas d'identité → ignoré (personne inconnue)

        Args:
            tracks: Liste des tracks DeepSort confirmés
            now: Timestamp actuel

        Returns:
            Liste des noms actuellement présents
        """
        current_names = []

        for track in tracks:
            track_id = track.track_id
            name = self.track_identity.get(track_id)

            if name is not None:
                # Mise à jour du timestamp de dernière détection
                self.present_users[name] = now
                self.all_seen_today.add(name)
                current_names.append(name)

        # Nettoyage des tracks disparus du mapping d'identité
        active_track_ids = {t.track_id for t in tracks}
        stale_ids = [
            tid for tid in self.track_identity
            if tid not in active_track_ids
        ]
        for tid in stale_ids:
            del self.track_identity[tid]

        return current_names

    # =========================================================================
    # ÉTAPE 6 : ANNOTATION VISUELLE
    # =========================================================================

    def _annotate_frame(
        self, frame: np.ndarray, tracks: list, now: float
    ) -> np.ndarray:
        """
        Dessine les bboxes et noms sur la frame pour les personnes connues uniquement.

        Style visuel :
          - Rectangle vert avec coins arrondis simulés
          - Nom affiché sur fond vert au-dessus de la bbox
          - Indicateur "TRACKING" si la personne est suivie sans visage visible

        Args:
            frame: Image BGR originale
            tracks: Liste des tracks DeepSort confirmés
            now: Timestamp actuel

        Returns:
            Frame annotée
        """
        annotated = frame.copy()

        for track in tracks:
            track_id = track.track_id
            name = self.track_identity.get(track_id)

            # RÈGLE STRICTE : pas de dessin pour les inconnus
            if name is None:
                continue

            # Récupération de la bbox du track
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # --- Couleur dynamique ---
            # Vert si vu récemment (visage visible), orange si tracking seul
            last_seen = self.present_users.get(name, 0)
            is_recent_face = (now - last_seen) < 1.0  # Visage vu dans la dernière seconde

            if is_recent_face:
                color = (0, 220, 100)       # Vert — visage confirmé
                status = "IDENTIFIE"
            else:
                color = (0, 165, 255)       # Orange — tracking persistant
                status = "TRACKING"

            # --- Dessin de la bbox ---
            thickness = 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Petits coins décoratifs (style caméra de sécurité)
            corner_len = 20
            ct = 3  # Épaisseur des coins
            # Coin supérieur gauche
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, ct)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, ct)
            # Coin supérieur droit
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, ct)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, ct)
            # Coin inférieur gauche
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, ct)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, ct)
            # Coin inférieur droit
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, ct)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, ct)

            # --- Étiquette avec le nom ---
            # Formatage du nom pour l'affichage (remplace les underscores)
            display_name = name.replace("_", " ")
            label = f"{display_name} [{status}]"

            # Taille du texte pour le fond
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (tw, th), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Fond du label
            label_y1 = max(0, y1 - th - 14)
            label_y2 = y1 - 2
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + tw + 10, label_y2),
                color,
                cv2.FILLED
            )

            # Texte blanc sur fond coloré
            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

        # --- Bandeau informatif en haut de l'écran ---
        self._draw_info_bar(annotated, now)

        return annotated

    def _draw_info_bar(self, frame: np.ndarray, now: float) -> None:
        """
        Dessine un bandeau semi-transparent en haut de la frame
        avec les statistiques en temps réel.

        Args:
            frame: Image BGR (modifiée en place)
            now: Timestamp actuel
        """
        h, w = frame.shape[:2]
        bar_height = 40

        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Nombre de personnes actuellement trackées (identifiées)
        n_present = len([
            name for name, ts in self.present_users.items()
            if (now - ts) < config.PRESENCE_TIMEOUT
        ])

        # Nombre total de personnes uniques vues
        n_total = len(self.all_seen_today)

        # Texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = (
            f"SMART TRACKER | "
            f"Presents: {n_present} | "
            f"Total aujourd'hui: {n_total} | "
            f"Base: {len(self.face_db)} personne(s)"
        )
        cv2.putText(
            frame, info_text, (10, 27),
            font, 0.55, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Timestamp
        time_str = time.strftime("%H:%M:%S")
        cv2.putText(
            frame, time_str, (w - 100, 27),
            font, 0.55, (100, 255, 100), 1, cv2.LINE_AA
        )

    # =========================================================================
    # API PUBLIQUE POUR LE SERVEUR FLASK
    # =========================================================================

    def get_currently_present(self) -> List[str]:
        """
        Retourne la liste des personnes actuellement présentes
        (vues dans les dernières PRESENCE_TIMEOUT secondes).

        Returns:
            Liste de noms
        """
        now = time.time()
        return [
            name for name, ts in self.present_users.items()
            if (now - ts) < config.PRESENCE_TIMEOUT
        ]

    def get_total_unique_today(self) -> int:
        """
        Retourne le nombre total de personnes uniques vues aujourd'hui.

        Returns:
            Entier >= 0
        """
        return len(self.all_seen_today)

    def get_all_seen_names(self) -> List[str]:
        """
        Retourne la liste de tous les noms uniques vus depuis le lancement.

        Returns:
            Liste triée de noms
        """
        return sorted(self.all_seen_today)

    # =========================================================================
    # NETTOYAGE
    # =========================================================================

    def release(self) -> None:
        """Libère les ressources Mediapipe."""
        if self.mp_pose:
            self.mp_pose.close()
            print("[TRACKER] Mediapipe Pose libéré.")

    def __del__(self):
        """Destructeur — libération automatique."""
        self.release()


# =============================================================================
# POINT D'ENTRÉE POUR TEST AUTONOME (webcam directe, sans Flask)
# =============================================================================

if __name__ == "__main__":
    """
    Test autonome du tracker avec affichage OpenCV direct.
    Lancer avec : python -m core.tracker
    ou           : python core/tracker.py

    Commandes :
      - 'q' : Quitter
      - 's' : Afficher les statistiques dans le terminal
    """
    import sys
    sys.path.insert(0, __import__("os").path.dirname(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__))
    ))

    print("=" * 55)
    print("  TEST AUTONOME DU SMART TRACKER")
    print("  Appuyez sur 'q' pour quitter, 's' pour les stats")
    print("=" * 55)

    # Initialisation
    tracker = SmartTracker()
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra.")
        sys.exit(1)

    # Réglage de la résolution de capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)

    # Variables pour le calcul du FPS
    fps_start = time.time()
    fps_count = 0
    display_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame non lue. Fin du flux.")
                break

            # Traitement par le tracker
            annotated, current_names = tracker.process_frame(frame)

            # Calcul du FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # Affichage du FPS sur la frame
            cv2.putText(
                annotated,
                f"FPS: {display_fps:.1f}",
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA
            )

            # Affichage
            cv2.imshow("Smart Tracker - Test", annotated)

            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                print(f"\n[STATS] Présents : {tracker.get_currently_present()}")
                print(f"[STATS] Total unique : {tracker.get_total_unique_today()}")
                print(f"[STATS] Tous vus : {tracker.get_all_seen_names()}\n")

    except KeyboardInterrupt:
        print("\n[INFO] Interruption clavier.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
        print("[INFO] Ressources libérées. Au revoir.")
