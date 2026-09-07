"""
tests/test_pose_from_keypoints.py

Teste KeypointPoseEstimator : classification d'orientation depuis les keypoints
COCO produits par YOLOv8-Pose, alternative GPU à Mediapipe (config.POSE_SOURCE).

Aucun modèle n'est chargé : la classification est purement arithmétique sur des
triplets (x, y, conf). Les cas couvrent les quatre verdicts possibles, les deux
chemins qui mènent à "Dos", et toutes les sorties None (données inexploitables).

Lancer : pytest tests/test_pose_from_keypoints.py -v
"""

import pytest

import config
from core.pose_from_keypoints import KeypointPoseEstimator

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_kps(nose=(100.0, 50.0, 0.9),
             left_ear=(115.0, 45.0, 0.8),
             right_ear=(85.0, 45.0, 0.8),
             left_shoulder=(160.0, 100.0, 0.9),
             right_shoulder=(40.0, 100.0, 0.9)):
    """
    Construit une séquence de keypoints COCO 0-6.

    Défauts : personne de face, nez exactement au centre des épaules
    (centre = 100, écart = 120 px, au-dessus de POSE_MIN_SHOULDER_DIST_PX),
    oreilles également visibles.
    Les yeux (indices 1-2) ne sont pas lus par l'estimateur.
    """
    eye = (0.0, 0.0, 0.0)
    return [nose, eye, eye, left_ear, right_ear, left_shoulder, right_shoulder]


@pytest.fixture
def estimator():
    return KeypointPoseEstimator()


# ─────────────────────────────────────────────────────────────────────────────
# Tests : verdicts nominaux
# ─────────────────────────────────────────────────────────────────────────────

class TestClassification:

    def test_nez_centre_donne_face(self, estimator):
        assert estimator.estimate(make_kps()) == "Face"

    def test_oreille_gauche_seule_visible_donne_profil_droit(self, estimator):
        """Voir l'oreille gauche (anatomique) = personne tournée vers sa droite."""
        kps = make_kps(left_ear=(115.0, 45.0, 0.9), right_ear=(85.0, 45.0, 0.1))
        assert estimator.estimate(kps) == "Profil droit"

    def test_oreille_droite_seule_visible_donne_profil_gauche(self, estimator):
        kps = make_kps(left_ear=(115.0, 45.0, 0.1), right_ear=(85.0, 45.0, 0.9))
        assert estimator.estimate(kps) == "Profil gauche"

    def test_nez_decale_vers_les_x_croissants_donne_profil_gauche(self, estimator):
        """Oreilles à égalité : c'est l'offset du nez qui tranche. ratio = +0.25."""
        kps = make_kps(nose=(120.0, 50.0, 0.9))
        assert estimator.estimate(kps) == "Profil gauche"

    def test_nez_decale_vers_les_x_decroissants_donne_profil_droit(self, estimator):
        kps = make_kps(nose=(80.0, 50.0, 0.9))
        assert estimator.estimate(kps) == "Profil droit"

    def test_offset_sous_le_seuil_reste_face(self, estimator):
        """ratio = 8/80 = 0.10, sous POSE_OFFSET_RATIO_THRESHOLD (0.15)."""
        kps = make_kps(nose=(108.0, 50.0, 0.9))
        assert estimator.estimate(kps) == "Face"


class TestDos:
    """Deux chemins distincts mènent à 'Dos', les deux doivent être couverts."""

    def test_tete_entierement_invisible(self, estimator):
        kps = make_kps(nose=(100.0, 50.0, 0.1),
                       left_ear=(115.0, 45.0, 0.1),
                       right_ear=(85.0, 45.0, 0.1))
        assert estimator.estimate(kps) == "Dos"

    def test_nez_incertain_mais_oreilles_visibles(self, estimator):
        """Le nez passe le premier filtre (0.35 > 0.3) mais échoue au second (< 0.4)."""
        kps = make_kps(nose=(100.0, 50.0, 0.35),
                       left_ear=(115.0, 45.0, 0.9),
                       right_ear=(85.0, 45.0, 0.9))
        assert estimator.estimate(kps) == "Dos"


# ─────────────────────────────────────────────────────────────────────────────
# Tests : données inexploitables → None
# ─────────────────────────────────────────────────────────────────────────────

class TestDonneesInexploitables:

    def test_keypoints_absents(self, estimator):
        assert estimator.estimate(None) is None

    def test_keypoints_trop_courts(self, estimator):
        """Moins de 7 keypoints : les épaules manquent, rien n'est décidable."""
        assert estimator.estimate(make_kps()[:5]) is None

    def test_personne_lointaine_tous_keypoints_incertains(self, estimator):
        """
        Non-régression : une personne trop petite pour que YOLO soit sûr de quoi
        que ce soit doit donner None, pas "Dos". Le contrôle des épaules passe
        donc avant celui de la tête.
        """
        kps = make_kps(nose=(100.0, 50.0, 0.1),
                       left_ear=(115.0, 45.0, 0.1),
                       right_ear=(85.0, 45.0, 0.1),
                       left_shoulder=(140.0, 100.0, 0.15),
                       right_shoulder=(60.0, 100.0, 0.15))
        assert estimator.estimate(kps) is None

    def test_epaule_non_detectee(self, estimator):
        """
        YOLO renvoie (0, 0, 0) pour un keypoint absent. Sans ce garde-fou,
        l'écart d'épaules serait calculé sur des coordonnées nulles.
        """
        kps = make_kps(left_shoulder=(0.0, 0.0, 0.0))
        assert estimator.estimate(kps) is None

    def test_epaules_trop_proches(self, estimator):
        """Écart de 4 px : personne minuscule ou de profil strict, ratio non fiable."""
        kps = make_kps(left_shoulder=(102.0, 100.0, 0.9),
                       right_shoulder=(98.0, 100.0, 0.9))
        assert estimator.estimate(kps) is None

    def test_personne_lointaine_sous_le_seuil_de_taille(self, estimator):
        """
        Écart de 80 px, typique d'une personne à distance moyenne : sous les
        100 px mesurés comme limite de fiabilité, on refuse de conclure.
        """
        kps = make_kps(left_shoulder=(140.0, 100.0, 0.9),
                       right_shoulder=(60.0, 100.0, 0.9))
        assert estimator.estimate(kps) is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests : les seuils viennent bien de config (réglables sans toucher au code)
# ─────────────────────────────────────────────────────────────────────────────

class TestSeuilsConfigurables:

    def test_seuil_offset_pilote_le_verdict(self, estimator, monkeypatch):
        kps = make_kps(nose=(120.0, 50.0, 0.9))     # ratio = 0.25
        assert estimator.estimate(kps) == "Profil gauche"

        monkeypatch.setattr(config, "POSE_OFFSET_RATIO_THRESHOLD", 0.30)
        assert estimator.estimate(kps) == "Face"

    def test_seuil_ecart_oreilles_pilote_le_verdict(self, estimator, monkeypatch):
        kps = make_kps(left_ear=(115.0, 45.0, 0.9), right_ear=(85.0, 45.0, 0.4))
        assert estimator.estimate(kps) == "Profil droit"   # écart 0.5 > 0.4

        monkeypatch.setattr(config, "POSE_EAR_DIFF_THRESHOLD", 0.6)
        assert estimator.estimate(kps) == "Face"           # écart passe sous le seuil
