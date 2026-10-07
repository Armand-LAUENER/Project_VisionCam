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

from core.face_recognition import FaceRecognizer, is_frontal

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


class TestStackedMatrix:
    """La matrice (N, 512) est empilée une fois, puis à chaque changement de la base.

    Avant : np.stack à chaque appel de _identify, soit à chaque visage reconnu.
    """

    def count_stacks(self, monkeypatch):
        calls = []
        real_stack = np.stack

        def counting_stack(*args, **kwargs):
            calls.append(1)
            return real_stack(*args, **kwargs)

        monkeypatch.setattr("core.face_recognition.np.stack", counting_stack)
        return calls

    def test_matrix_is_not_restacked_between_calls(self, monkeypatch):
        recognizer = make_recognizer([("Alice", unit(1, 0)), ("Bob", unit(0, 1))])
        stacks = self.count_stacks(monkeypatch)

        for _ in range(5):
            recognizer._identify(unit(1, 0))

        assert len(stacks) == 1

    def test_enrolled_person_is_recognized_right_away(self):
        recognizer = make_recognizer([("Alice", unit(1, 0))])
        recognizer._identify(unit(1, 0))

        with recognizer._lock:
            recognizer._upsert_embedding("Bob", unit(0, 1))

        assert recognizer._identify(unit(0, 1))[0] == "Bob"

    def test_updated_embedding_replaces_the_old_one(self):
        recognizer = make_recognizer([("Alice", unit(1, 0)), ("Bob", unit(0, 1))])
        recognizer._identify(unit(1, 0))

        with recognizer._lock:
            recognizer._upsert_embedding("Alice", unit(0, 0, 1))

        assert recognizer._identify(unit(1, 0))[0] == "Inconnu"
        assert recognizer._identify(unit(0, 0, 1))[0] == "Alice"

    def test_replaced_database_is_used(self):
        """Rebuild, chargement du cache et suppression remplacent les listes entières."""
        recognizer = make_recognizer([("Alice", unit(1, 0))])
        recognizer._identify(unit(1, 0))

        with recognizer._lock:
            recognizer.known_embeddings, recognizer.known_names = [unit(0, 1)], ["Bob"]

        assert recognizer._identify(unit(0, 1))[0] == "Bob"
        assert recognizer._identify(unit(1, 0))[0] == "Inconnu"


# ─────────────────────────────────────────────────────────────────────────────
# Visage de face ou tourné (is_frontal), d'après les 5 points SCRFD
# ─────────────────────────────────────────────────────────────────────────────

def _kps(left_eye_x, right_eye_x, nose_x):
    return [[left_eye_x, 50], [right_eye_x, 50], [nose_x, 70], [left_eye_x, 90], [right_eye_x, 90]]


@pytest.mark.parametrize("nose_x, frontal", [
    (50, True),     # milieu des yeux
    (45, True),     # légèrement tourné (0,375 de l'écart)
    (38, False),    # trois-quarts : nez à 0,2 de l'écart, près de l'œil
    (20, False),    # profil : nez au-delà de l'œil
    (85, False),
])
def test_frontal_from_nose_position(nose_x, frontal):
    assert is_frontal(_kps(30, 70, nose_x), margin=0.25) is frontal


def test_eye_order_does_not_matter():
    assert is_frontal(_kps(70, 30, 50), margin=0.25)


def test_merged_eyes_are_not_frontal():
    assert not is_frontal(_kps(50, 50, 50), margin=0.25)
