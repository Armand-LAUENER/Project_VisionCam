"""
tests/test_people_management.py

Couvre la gestion des personnes de FaceRecognizer : enrôlement cumulatif,
liste, renommage et suppression. InsightFace est remplacé par un double dont
l'embedding dépend de la couleur de l'image : deux photos de la même couleur
représentent la même personne.

Lancer : pytest tests/test_people_management.py -v
"""

import os
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.face_recognition import CONFLICT, INVALID, NOT_FOUND, OK, FaceRecognizer


def photo(value):
    return np.full((32, 32, 3), value, dtype=np.uint8)


def embedding_for(img):
    """Vecteur unitaire propre à la valeur des pixels ; image noire = pas de visage."""
    value = int(img[0, 0, 0])
    if value == 0:
        return None
    v = np.zeros(512, dtype=np.float32)
    v[value % 512] = 1.0
    v[(value * 7) % 512] += 0.5
    return v


def make_recognizer(tmp_path):
    recognizer = object.__new__(FaceRecognizer)
    recognizer.threshold = 0.45
    recognizer.known_faces_dir = str(tmp_path / "known_faces")
    recognizer.cache_path = str(tmp_path / "data" / "embeddings.npz")
    os.makedirs(recognizer.known_faces_dir)
    os.makedirs(os.path.dirname(recognizer.cache_path))
    recognizer.known_embeddings = []
    recognizer.known_names = []
    recognizer._lock = threading.Lock()

    def fake_get(img):
        emb = embedding_for(img)
        return [] if emb is None else [SimpleNamespace(bbox=[0, 0, 32, 32], embedding=emb)]

    recognizer.app = SimpleNamespace(get=fake_get)
    return recognizer


def files(recognizer, entry):
    return sorted(os.listdir(os.path.join(recognizer.known_faces_dir, entry)))


def cached_names(recognizer):
    with np.load(recognizer.cache_path, allow_pickle=False) as data:
        return sorted(str(n) for n in data["names"])


class TestEnrollmentAccumulates:

    def test_enrolling_again_adds_photos_instead_of_overwriting(self, tmp_path):
        """Non-régression : le second enrôlement réécrivait avg_000.jpg."""
        recognizer = make_recognizer(tmp_path)

        recognizer.enroll_person_average("Alice", [photo(10), photo(10)])
        recognizer.enroll_person_average("Alice", [photo(12)])

        assert files(recognizer, "Alice") == ["avg_000.jpg", "avg_001.jpg", "avg_002.jpg"]

    def test_embedding_matches_what_a_rebuild_would_compute(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_average("Alice", [photo(10)])
        recognizer.enroll_person_average("Alice", [photo(12)])
        after_enroll = recognizer.known_embeddings[recognizer.known_names.index("Alice")]

        recognizer.rebuild_database()

        rebuilt = recognizer.known_embeddings[recognizer.known_names.index("Alice")]
        np.testing.assert_allclose(after_enroll, rebuilt, atol=1e-6)

    def test_images_without_a_face_are_not_saved(self, tmp_path):
        recognizer = make_recognizer(tmp_path)

        success, _ = recognizer.enroll_person_average("Alice", [photo(10), photo(0)])

        assert success
        assert files(recognizer, "Alice") == ["avg_000.jpg"]


class TestListPeople:

    def test_groups_templates_and_counts_photos(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_average("Bob", [photo(20), photo(20)])
        recognizer.enroll_person_multitemplate("Alice", {"Face": photo(10), "ProfilG": photo(11)})

        people = recognizer.list_people()

        assert [p["name"] for p in people] == ["Alice", "Bob"]
        alice, bob = people
        assert alice["templates"] == ["Face", "ProfilG"] and alice["photos"] == 2
        assert bob["templates"] == [] and bob["photos"] == 2
        assert all(p["enrolled"] for p in people)
        assert bob["thumbnail"].endswith(os.path.join("Bob", "avg_000.jpg"))

    def test_folder_without_a_usable_face_is_listed_as_not_enrolled(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        os.makedirs(os.path.join(recognizer.known_faces_dir, "Carol"))
        cv2.imwrite(os.path.join(recognizer.known_faces_dir, "Carol", "a.jpg"), photo(0))

        (carol,) = recognizer.list_people()

        assert carol["name"] == "Carol" and carol["photos"] == 1 and not carol["enrolled"]


class TestRenamePerson:

    def test_moves_folders_and_entries_together(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_multitemplate("Alice", {"Face": photo(10), "ProfilG": photo(11)})

        status, _ = recognizer.rename_person("Alice", "Alice Martin")

        assert status == OK
        assert sorted(os.listdir(recognizer.known_faces_dir)) == ["Alice Martin#Face",
                                                                  "Alice Martin#ProfilG"]
        assert sorted(recognizer.known_names) == ["Alice Martin#Face", "Alice Martin#ProfilG"]
        assert cached_names(recognizer) == ["Alice Martin#Face", "Alice Martin#ProfilG"]
        assert recognizer._identify(embedding_for(photo(10)))[0] == "Alice Martin"

    def test_refuses_a_name_already_taken(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_average("Alice", [photo(10)])
        recognizer.enroll_person_average("Bob", [photo(20)])

        status, _ = recognizer.rename_person("Alice", "Bob")

        assert status == CONFLICT
        assert sorted(os.listdir(recognizer.known_faces_dir)) == ["Alice", "Bob"]

    def test_unknown_person(self, tmp_path):
        assert make_recognizer(tmp_path).rename_person("Nobody", "Somebody")[0] == NOT_FOUND

    def test_does_not_touch_a_person_whose_name_merely_starts_the_same(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_average("Al", [photo(10)])
        recognizer.enroll_person_average("Alice", [photo(20)])

        recognizer.rename_person("Al", "Albert")

        assert sorted(recognizer.known_names) == ["Albert", "Alice"]

    @pytest.mark.parametrize("old, new", [("../x", "Bob"), ("Alice", "../../evil"), ("Alice", "")])
    def test_refuses_unsafe_names(self, tmp_path, old, new):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_average("Alice", [photo(10)])

        assert recognizer.rename_person(old, new)[0] == INVALID
        assert os.listdir(recognizer.known_faces_dir) == ["Alice"]


class TestDeletePerson:

    def test_removes_photos_entries_and_cache(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_multitemplate("Alice", {"Face": photo(10), "ProfilG": photo(11)})
        recognizer.enroll_person_average("Bob", [photo(20)])

        status, _ = recognizer.delete_person("Alice")

        assert status == OK
        assert os.listdir(recognizer.known_faces_dir) == ["Bob"]
        assert recognizer.known_names == ["Bob"]
        assert cached_names(recognizer) == ["Bob"]
        assert recognizer._identify(embedding_for(photo(10)))[0] == "Inconnu"

    def test_unknown_person(self, tmp_path):
        assert make_recognizer(tmp_path).delete_person("Nobody")[0] == NOT_FOUND

    def test_refuses_unsafe_names_without_deleting_anything(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        recognizer.enroll_person_average("Alice", [photo(10)])
        outside = tmp_path / "outside"
        outside.mkdir()

        assert recognizer.delete_person("../outside")[0] == INVALID
        assert outside.exists()
        assert os.listdir(recognizer.known_faces_dir) == ["Alice"]
