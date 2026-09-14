"""
tests/test_face_recognition_cache.py

Couvre le cache d'embeddings de FaceRecognizer : `data/embeddings.npz` est
entièrement reconstructible depuis `known_faces/`, un cache abîmé ne doit donc
jamais empêcher le démarrage.

Régression : un cache de 0 octet — laissé par un processus tué pendant
l'écriture — faisait remonter `EOFError` depuis `__init__`, et l'application
ne démarrait plus du tout.

Le cache était un pickle, dont le chargement exécute du code si le fichier a
été altéré. Il est désormais en .npz, lu avec allow_pickle=False.

InsightFace n'est pas chargé : les méthodes testées ne touchent jamais
`self.app`, l'instance est donc construite sans passer par `__init__`.

Lancer : pytest tests/test_face_recognition_cache.py -v
"""

import os
import pickle
import threading

import numpy as np
import pytest

from core.face_recognition import FaceRecognizer


def make_recognizer(tmp_path):
    """FaceRecognizer réduit à ce que le cache met en jeu."""
    recognizer = object.__new__(FaceRecognizer)
    recognizer.cache_path = str(tmp_path / "data" / "embeddings.npz")
    recognizer.known_faces_dir = str(tmp_path / "known_faces")
    os.makedirs(recognizer.known_faces_dir, exist_ok=True)
    os.makedirs(os.path.dirname(recognizer.cache_path), exist_ok=True)
    recognizer.known_embeddings = []
    recognizer.known_names = []
    recognizer._lock = threading.Lock()
    return recognizer


def write_cache(recognizer, names, embeddings=None):
    if embeddings is None:
        embeddings = np.ones((len(names), 512), dtype=np.float32)
    with open(recognizer.cache_path, "wb") as f:
        np.savez(f, names=np.array(names, dtype=str), embeddings=embeddings)


def read_cached_names(recognizer):
    with np.load(recognizer.cache_path, allow_pickle=False) as data:
        return [str(name) for name in data["names"]]


def tmp_leftovers(recognizer):
    directory = os.path.dirname(recognizer.cache_path)
    return [name for name in os.listdir(directory) if name.endswith(".tmp")]


# ─────────────────────────────────────────────────────────────────────────────
# Lecture d'un cache abîmé
# ─────────────────────────────────────────────────────────────────────────────

class TestDamagedCacheIsRebuilt:

    def test_empty_cache_does_not_block_startup(self, tmp_path):
        """Le cas rencontré : fichier de 0 octet."""
        recognizer = make_recognizer(tmp_path)
        open(recognizer.cache_path, "wb").close()

        recognizer._load_or_build_database()

        assert recognizer.known_names == []
        # Reconstruit ET réécrit : le fichier vide ne doit pas rester en place
        # pour refaire le même effet au démarrage suivant.
        assert read_cached_names(recognizer) == []

    def test_truncated_cache_does_not_block_startup(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        write_cache(recognizer, ["Armand"])
        with open(recognizer.cache_path, "rb+") as f:
            f.truncate(12)

        recognizer._load_or_build_database()

        assert recognizer.known_names == []

    def test_cache_missing_a_key_does_not_block_startup(self, tmp_path):
        """Cache d'une version antérieure, ou écrit par autre chose."""
        recognizer = make_recognizer(tmp_path)
        with open(recognizer.cache_path, "wb") as f:
            np.savez(f, embeddings=np.ones((1, 512), dtype=np.float32))

        recognizer._load_or_build_database()

        assert recognizer.known_names == []

    def test_mismatched_lengths_do_not_block_startup(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        write_cache(recognizer, ["Armand", "Invité"], np.ones((1, 512), dtype=np.float32))

        recognizer._load_or_build_database()

        assert recognizer.known_names == []

    def test_pickled_content_is_refused_not_executed(self, tmp_path):
        """Un pickle à la place du cache est rejeté sans être désérialisé."""
        recognizer = make_recognizer(tmp_path)
        marker = tmp_path / "executed"
        with open(recognizer.cache_path, "wb") as f:
            pickle.dump(_Payload(str(marker)), f)

        recognizer._load_or_build_database()

        assert not marker.exists()
        assert recognizer.known_names == []

    def test_valid_cache_is_used_as_is(self, tmp_path):
        """Le chemin nominal ne doit pas devenir une reconstruction silencieuse :
        reconstruire coûte une passe InsightFace sur toutes les photos."""
        recognizer = make_recognizer(tmp_path)
        write_cache(recognizer, ["Armand", "Invité"])

        recognizer._load_or_build_database()

        assert recognizer.known_names == ["Armand", "Invité"]
        assert len(recognizer.known_embeddings) == 2
        assert recognizer.known_embeddings[0].shape == (512,)

    def test_absent_cache_is_built(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        assert not os.path.exists(recognizer.cache_path)

        recognizer._load_or_build_database()

        assert recognizer.known_names == []
        assert os.path.exists(recognizer.cache_path)


class _Payload:
    """Objet dont la désérialisation crée un fichier témoin."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


# ─────────────────────────────────────────────────────────────────────────────
# Écriture : c'est elle qui a produit le fichier vide
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheWriteIsAtomic:

    def test_failed_write_leaves_the_previous_cache_intact(self, tmp_path, monkeypatch):
        recognizer = make_recognizer(tmp_path)
        write_cache(recognizer, ["Armand"])
        recognizer.known_names = ["Nouveau"]
        recognizer.known_embeddings = [np.ones(512, dtype=np.float32)]

        def boom(*_args, **_kwargs):
            raise OSError("disque plein")

        monkeypatch.setattr("core.face_recognition.np.savez", boom)
        with pytest.raises(OSError):
            recognizer._save_cache()

        assert read_cached_names(recognizer) == ["Armand"]
        assert tmp_leftovers(recognizer) == []

    def test_successful_write_replaces_the_cache(self, tmp_path):
        recognizer = make_recognizer(tmp_path)
        write_cache(recognizer, ["Armand"])
        recognizer.known_names = ["Nouveau"]
        recognizer.known_embeddings = [np.ones(512, dtype=np.float32)]

        recognizer._save_cache()

        assert read_cached_names(recognizer) == ["Nouveau"]
        assert tmp_leftovers(recognizer) == []

    def test_round_trip_keeps_names_and_embeddings(self, tmp_path):
        """Accents et suffixe multitemplate compris."""
        recognizer = make_recognizer(tmp_path)
        recognizer.known_names = ["Hélène", "Armand#Profil"]
        recognizer.known_embeddings = [np.eye(512, dtype=np.float32)[i] for i in (0, 1)]
        recognizer._save_cache()

        reloaded = make_recognizer(tmp_path)
        reloaded._load_or_build_database()

        assert reloaded.known_names == ["Hélène", "Armand#Profil"]
        np.testing.assert_array_equal(np.stack(reloaded.known_embeddings),
                                      np.stack(recognizer.known_embeddings))
