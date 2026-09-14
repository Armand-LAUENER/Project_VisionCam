"""
tests/test_identify.py

Couvre FaceRecognizer._identify : meilleur score cosinus, seuil, suffixe
multitemplate. InsightFace n'est pas chargé, l'instance est construite sans
passer par `__init__`.

Lancer : pytest tests/test_identify.py -v
"""

import threading

import numpy as np
import pytest

from core.face_recognition import FaceRecognizer

AXES = np.eye(512, dtype=np.float32)


def make_recognizer(entries, threshold=0.45):
    recognizer = object.__new__(FaceRecognizer)
    recognizer.threshold = threshold
    recognizer.known_names = [name for name, _ in entries]
    recognizer.known_embeddings = [emb for _, emb in entries]
    recognizer._lock = threading.Lock()
    return recognizer


def unit(*weights):
    v = sum(w * AXES[i] for i, w in enumerate(weights))
    return v / np.linalg.norm(v)


class TestIdentify:

    def test_empty_database(self):
        assert make_recognizer([])._identify(AXES[0]) == ("Inconnu", 0.0)

    def test_picks_the_closest_entry(self):
        recognizer = make_recognizer([("Alice", unit(1, 0)), ("Bob", unit(0, 1))])

        name, score = recognizer._identify(unit(0.2, 1))

        assert name == "Bob"
        assert score == pytest.approx(float(unit(0.2, 1) @ unit(0, 1)))

    def test_embedding_need_not_be_normalized(self):
        recognizer = make_recognizer([("Alice", unit(1, 0))])

        name, score = recognizer._identify(AXES[0] * 37.0)

        assert name == "Alice"
        assert score == pytest.approx(1.0)

    def test_below_threshold_is_unknown_but_keeps_its_score(self):
        recognizer = make_recognizer([("Alice", unit(1, 0))], threshold=0.9)

        name, score = recognizer._identify(unit(1, 1))

        assert name == "Inconnu"
        assert score == pytest.approx(0.7071, abs=1e-4)

    def test_only_negative_scores_give_zero(self):
        recognizer = make_recognizer([("Alice", unit(1, 0))])

        assert recognizer._identify(-AXES[0]) == ("Inconnu", 0.0)

    def test_multitemplate_suffix_is_stripped(self):
        recognizer = make_recognizer([("Armand#Face", unit(1, 0)), ("Armand#Profil", unit(0, 1))])

        assert recognizer._identify(unit(0, 1))[0] == "Armand"

    def test_tie_keeps_the_first_entry(self):
        recognizer = make_recognizer([("Alice", unit(1, 0)), ("Bob", unit(1, 0))])

        assert recognizer._identify(unit(1, 0))[0] == "Alice"
