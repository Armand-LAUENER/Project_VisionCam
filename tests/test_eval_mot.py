"""
tests/test_eval_mot.py

Couvre la lecture des détections publiques MOT17 par tools/eval_mot.py.

C'est l'endroit où une erreur passerait inaperçue : un décalage de repère ne
fait pas planter l'évaluation, il déplace juste toutes les boîtes d'un pixel et
fausse silencieusement le MOTA. MOT indexe ses pixels à la manière de Matlab,
motmetrics ramène la vérité terrain en base 0 au chargement — les détections
doivent suivre le même repère.

Lancer : pytest tests/test_eval_mot.py -v
"""

import pytest

from tools.eval_mot import load_public_detections


@pytest.fixture
def seq_dir(tmp_path):
    """Séquence MOT17 minimale : uniquement le det.txt qui nous intéresse."""
    det_dir = tmp_path / "det"
    det_dir.mkdir()
    (det_dir / "det.txt").write_text(
        # frame, id, left, top, width, height, conf, x, y, z
        "1,-1,100,200,50,120,0.9,-1,-1,-1\n"
        "1,-1,300,150,40,100,0.2,-1,-1,-1\n"   # sous le seuil de confiance
        "2,-1,110,205,50,120,0.8,-1,-1,-1\n"
        "2,-1,400,100,0,90,0.95,-1,-1,-1\n"    # largeur nulle
        "2,-1,500,100,30,0,0.95,-1,-1,-1\n"    # hauteur nulle
    )
    return str(tmp_path)


def test_matlab_indexing_is_removed(seq_dir):
    """Le coin haut-gauche perd 1 px sur chaque axe, la taille ne bouge pas."""
    detections = load_public_detections(seq_dir, min_conf=0.5)

    ltwh, conf, label, others = detections[1][0]
    assert ltwh == [99.0, 199.0, 50.0, 120.0]
    assert conf == 0.9
    assert label == "person"
    assert others is None


def test_detections_below_the_threshold_are_dropped(seq_dir):
    assert len(load_public_detections(seq_dir, min_conf=0.5)[1]) == 1
    assert len(load_public_detections(seq_dir, min_conf=0.0)[1]) == 2


def test_degenerate_boxes_are_dropped(seq_dir):
    """Une boîte plate ferait planter le redimensionnement de l'embedder."""
    frame_2 = load_public_detections(seq_dir, min_conf=0.5)[2]

    assert len(frame_2) == 1
    assert frame_2[0][0] == [109.0, 204.0, 50.0, 120.0]


def test_frames_without_detection_are_absent(seq_dir):
    """L'appelant utilise .get(frame, []) : une frame vide ne doit pas exister
    plutôt que d'exister vide, sinon on ne distingue plus les deux cas."""
    detections = load_public_detections(seq_dir, min_conf=0.5)

    assert set(detections) == {1, 2}
