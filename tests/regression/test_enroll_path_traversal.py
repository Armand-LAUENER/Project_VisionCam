"""
Regression test — un nom d'enrôlement ne peut pas écrire hors de known_faces/.

Bug d'origine : `name` (formulaire de /enroll et /capture) et `label` (nom du
fichier envoyé, en multitemplate) entraient tels quels dans os.path.join.
`name=../../x` écrivait des JPEG hors de known_faces/, et un chemin absolu
remplaçait carrément le dossier de base.
Fix : liste blanche de caractères sur le nom et le label, vérifiée avant toute
écriture, puis contrôle que le chemin résolu reste dans known_faces/.

InsightFace est remplacé par un double qui trouve un visage dans chaque image.

Lancer : pytest tests/regression/test_enroll_path_traversal.py -v
"""

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.face_recognition import FaceRecognizer

IMAGE = np.zeros((32, 32, 3), dtype=np.uint8)


def make_recognizer(tmp_path):
    recognizer = object.__new__(FaceRecognizer)
    recognizer.cache_path = str(tmp_path / "data" / "embeddings.pkl")
    recognizer.known_faces_dir = str(tmp_path / "known_faces")
    os.makedirs(recognizer.known_faces_dir)
    os.makedirs(os.path.dirname(recognizer.cache_path))
    recognizer.known_embeddings = []
    recognizer.known_names = []
    recognizer._lock = threading.Lock()
    face = SimpleNamespace(bbox=[0, 0, 10, 10], embedding=np.ones(512, dtype=np.float32))
    recognizer.app = SimpleNamespace(get=lambda img: [face])
    return recognizer


def files_outside_known_faces(tmp_path):
    return sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.jpg")
                  if "known_faces" not in p.parts)


class TestEnrollPathTraversal:

    @pytest.mark.parametrize("name", ["../evil", "../../evil", "a/../../evil",
                                      "..", ".hidden", "a\\b", "Bob#Face", ""])
    def test_average_rejects_unsafe_name(self, tmp_path, name):
        recognizer = make_recognizer(tmp_path)

        success, _ = recognizer.enroll_person_average(name, [IMAGE])

        assert success is False
        assert files_outside_known_faces(tmp_path) == []
        assert recognizer.known_names == []

    def test_average_rejects_absolute_path(self, tmp_path):
        recognizer = make_recognizer(tmp_path)

        success, _ = recognizer.enroll_person_average(str(tmp_path / "outside"), [IMAGE])

        assert success is False
        assert not (tmp_path / "outside").exists()

    def test_multitemplate_rejects_unsafe_label(self, tmp_path):
        """Un seul label dangereux : rien n'est écrit, pas même les labels sains."""
        recognizer = make_recognizer(tmp_path)

        success, _ = recognizer.enroll_person_multitemplate(
            "Bob", {"Face": IMAGE, "../../evil": IMAGE})

        assert success is False
        assert list(tmp_path.rglob("*.jpg")) == []
        assert recognizer.known_names == []

    def test_save_refuses_path_outside_known_faces(self, tmp_path):
        """Seconde barrière, indépendante de la liste blanche."""
        recognizer = make_recognizer(tmp_path)

        with pytest.raises(ValueError):
            recognizer._save_images_to_disk("../evil", [IMAGE])

        assert files_outside_known_faces(tmp_path) == []

    def test_ordinary_names_still_enroll(self, tmp_path):
        """Garde-fou : accents, espaces, tirets et points au milieu restent acceptés."""
        recognizer = make_recognizer(tmp_path)

        assert recognizer.enroll_person_average("Hélène Dupont-Durand", [IMAGE])[0]
        assert recognizer.enroll_person_multitemplate("Armand", {"photo.v2": IMAGE})[0]

        assert (tmp_path / "known_faces" / "Hélène Dupont-Durand" / "avg_000.jpg").exists()
        assert (tmp_path / "known_faces" / "Armand#photo.v2" / "photo.v2_000.jpg").exists()
