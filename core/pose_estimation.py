import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """
    Estime la posture d'une personne (face/profil/dos) via Mediapipe Pose.
    """

    def __init__(self):
        print("[POSE] Chargement Mediapipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,  # 0 = léger et rapide
            min_detection_confidence=0.5
        )
        print("[POSE] Mediapipe Pose chargé ✅")

    def estimate(self, person_crop):
        """
        Estime la pose d'un crop de personne.

        Args:
            person_crop: image BGR (crop d'une personne)

        Returns:
            str: "Face", "Profil gauche", "Profil droit", "Dos", ou None si pas détecté
        """
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        if h < 50 or w < 30:
            return None

        # Convertir en RGB pour Mediapipe
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        # Points clés
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]

        # Vérifier la visibilité minimale
        if nose.visibility < 0.3 and left_ear.visibility < 0.3 and right_ear.visibility < 0.3:
            return "Dos"

        # ── Calcul de l'orientation ──
        # Distance entre les épaules (référence)
        shoulder_dist = abs(left_shoulder.x - right_shoulder.x)

        if shoulder_dist < 0.01:
            return None

        # Centre des épaules
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2

        # Offset du nez par rapport au centre des épaules
        nose_offset = nose.x - shoulder_center_x

        # Ratio d'asymétrie des oreilles
        left_ear_vis = left_ear.visibility
        right_ear_vis = right_ear.visibility

        # ── Classification ──
        # Dos : nez peu visible
        if nose.visibility < 0.4:
            return "Dos"

        # Profil : une oreille beaucoup moins visible que l'autre
        ear_diff = abs(left_ear_vis - right_ear_vis)

        if ear_diff > 0.4:
            if left_ear_vis > right_ear_vis:
                return "Profil droit"  # On voit l'oreille gauche → tourné vers la droite
            else:
                return "Profil gauche"

        # Légère rotation basée sur l'offset du nez
        offset_ratio = nose_offset / shoulder_dist

        if abs(offset_ratio) < 0.15:
            return "Face"
        elif offset_ratio > 0.15:
            return "Profil gauche"
        else:
            return "Profil droit"

    def release(self):
        """Libère les ressources."""
        if self.pose:
            self.pose.close()
        print("[POSE] Ressources libérées")
