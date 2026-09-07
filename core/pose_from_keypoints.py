"""
pose_from_keypoints.py — Estimation d'orientation depuis les keypoints YOLOv8-Pose.

Alternative GPU à PoseEstimator (Mediapipe) : YOLOv8-Pose produit déjà les 17
keypoints COCO sur GPU à chaque frame, dont les 7 dont la classification
d'orientation a besoin. Les relire coûte zéro inférence supplémentaire, là où
Mediapipe ajoute un passage TFLite sur CPU par personne et par frame.

La logique de classification reproduit celle de PoseEstimator.estimate(), avec
deux adaptations imposées par la nature des données :

  - Mediapipe fournit une `visibility` par landmark, YOLO une confiance de
    keypoint. Les deux vivent dans [0, 1] et jouent le même rôle ici, mais
    leurs distributions diffèrent : les seuils sont donc exposés dans config.
  - YOLO renvoie (0, 0, 0) pour un keypoint non détecté. Un contrôle de
    confiance sur les épaules est indispensable, sinon l'écart d'épaules est
    calculé sur des coordonnées nulles et produit un verdict arbitraire.
    Mediapipe n'a pas besoin de ce garde-fou, ses landmarks sont toujours
    estimés.

Les coordonnées sont en pixels (repère frame) au lieu de normalisées : sans
importance, la classification ne repose que sur le ratio offset/écart d'épaules,
qui est invariant d'échelle.
"""

import logging

import config

logger = logging.getLogger(__name__)

# Indices COCO utilisés (ordre des keypoints YOLOv8-Pose)
NOSE = 0
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

REQUIRED_KEYPOINTS = RIGHT_SHOULDER + 1


class KeypointPoseEstimator:
    """
    Estime la posture (face/profil/dos) à partir des keypoints COCO d'une personne.

    Interface volontairement parallèle à PoseEstimator : `estimate()` puis
    `release()`, mêmes valeurs de retour, pour que app.py puisse basculer de
    l'un à l'autre sans traitement particulier.
    """

    def __init__(self):
        logger.info("Estimation de pose depuis les keypoints YOLO (aucun modèle à charger)")

    def estimate(self, keypoints):
        """
        Classe l'orientation d'une personne.

        Args:
            keypoints: séquence de (x, y, conf) en ordre COCO, au moins 7 éléments.
                       None ou trop courte → None.

        Returns:
            str: "Face", "Profil gauche", "Profil droit", "Dos", ou None si
                 les keypoints ne permettent pas de conclure.
        """
        if keypoints is None or len(keypoints) < REQUIRED_KEYPOINTS:
            return None

        nose_x, _, nose_conf = keypoints[NOSE]
        _, _, left_ear_conf = keypoints[LEFT_EAR]
        _, _, right_ear_conf = keypoints[RIGHT_EAR]
        left_sh_x, _, left_sh_conf = keypoints[LEFT_SHOULDER]
        right_sh_x, _, right_sh_conf = keypoints[RIGHT_SHOULDER]

        # Les épaules d'abord : ce sont les keypoints les plus robustes du haut
        # du corps, ils restent détectés de dos comme de profil. Si YOLO n'est
        # même pas sûr d'elles, c'est que la personne est trop petite ou trop
        # occultée pour conclure quoi que ce soit — et surtout pas "Dos".
        #
        # Ce contrôle précède délibérément le test de la tête. Dans l'ordre
        # inverse, une personne lointaine dont TOUS les keypoints sont
        # incertains ressortait "Dos" au lieu de None : mesuré sur MOT17-04,
        # 122 verdicts "Dos" sur 240 observations là où Mediapipe déclarait
        # forfait.
        if (left_sh_conf < config.POSE_KP_VISIBLE_CONF
                or right_sh_conf < config.POSE_KP_VISIBLE_CONF):
            return None

        # Épaules vues mais tête invisible sous tous les angles → de dos.
        if (nose_conf < config.POSE_KP_VISIBLE_CONF
                and left_ear_conf < config.POSE_KP_VISIBLE_CONF
                and right_ear_conf < config.POSE_KP_VISIBLE_CONF):
            return "Dos"

        shoulder_dist = abs(left_sh_x - right_sh_x)
        if shoulder_dist < config.POSE_MIN_SHOULDER_DIST_PX:
            return None

        # Nez trop incertain alors que les oreilles portent le signal → dos.
        if nose_conf < config.POSE_KP_NOSE_CONF:
            return "Dos"

        # Une oreille nettement plus sûre que l'autre → profil marqué.
        ear_diff = abs(left_ear_conf - right_ear_conf)
        if ear_diff > config.POSE_EAR_DIFF_THRESHOLD:
            # Oreille gauche (anatomique) visible → la personne est tournée vers sa droite.
            return "Profil droit" if left_ear_conf > right_ear_conf else "Profil gauche"

        # Sinon, décalage du nez par rapport au centre des épaules.
        shoulder_center_x = (left_sh_x + right_sh_x) / 2
        offset_ratio = (nose_x - shoulder_center_x) / shoulder_dist

        if abs(offset_ratio) < config.POSE_OFFSET_RATIO_THRESHOLD:
            return "Face"
        return "Profil gauche" if offset_ratio > 0 else "Profil droit"

    def release(self):
        """Aucune ressource à libérer — présent pour la parité d'interface."""
