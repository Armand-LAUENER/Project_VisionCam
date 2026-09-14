"""
Regression test — la reconnaissance garde la base complète pendant un rebuild.

Bug d'origine : _build_database vidait known_names / known_embeddings, puis
les remplissait personne par personne, sans verrou, pendant que le thread de
traitement continuait d'appeler _identify. Tant que la reconstruction durait
(une inférence InsightFace par image), les personnes connues ressortaient
« Inconnu ».
Fix : la nouvelle base est construite à part, puis remplace l'ancienne d'un
seul coup sous _lock.

Le double d'InsightFace interroge _identify à chaque image traitée : c'est
exactement ce que voit le thread de traitement au milieu du rebuild, sans
dépendre d'un ordonnancement de threads.

Lancer : pytest tests/regression/test_rebuild_keeps_database_readable.py -v
"""

import os
import threading
from types import SimpleNamespace

import cv2
import numpy as np

from core.face_recognition import FaceRecognizer

BOB = np.eye(512, dtype=np.float32)[0]
ALICE = np.eye(512, dtype=np.float32)[1]


def make_recognizer(tmp_path, people):
    """Recognizer avec Bob en base, et `people` ({nom: embedding}) sur disque."""
    recognizer = object.__new__(FaceRecognizer)
    recognizer.threshold = 0.45
    recognizer.cache_path = str(tmp_path / "data" / "embeddings.npz")
    recognizer.known_faces_dir = str(tmp_path / "known_faces")
    os.makedirs(os.path.dirname(recognizer.cache_path))
    recognizer.known_embeddings = [BOB]
    recognizer.known_names = ['Bob']
    recognizer._lock = threading.Lock()

    # Une image par personne, marquée par sa valeur de pixel pour que le
    # double sache quel embedding renvoyer.
    embeddings_by_pixel = {}
    for pixel, (name, embedding) in enumerate(people.items(), start=1):
        os.makedirs(os.path.join(recognizer.known_faces_dir, name))
        cv2.imwrite(os.path.join(recognizer.known_faces_dir, name, "a.png"),
                    np.full((8, 8, 3), pixel, dtype=np.uint8))
        embeddings_by_pixel[pixel] = embedding

    seen_mid_rebuild = []

    def fake_get(img):
        seen_mid_rebuild.append(recognizer._identify(BOB)[0])
        embedding = embeddings_by_pixel[int(img[0, 0, 0])]
        return [SimpleNamespace(bbox=[0, 0, 8, 8], embedding=embedding)]

    recognizer.app = SimpleNamespace(get=fake_get)
    return recognizer, seen_mid_rebuild


class TestRebuildKeepsDatabaseReadable:

    def test_known_person_stays_recognized_during_rebuild(self, tmp_path):
        recognizer, seen = make_recognizer(tmp_path, {'Alice': ALICE, 'Bob': BOB})

        recognizer.rebuild_database()

        assert seen == ['Bob', 'Bob']

    def test_rebuild_replaces_the_whole_database(self, tmp_path):
        """Garde-fou : la nouvelle base remplace bien l'ancienne à la fin."""
        recognizer, _ = make_recognizer(tmp_path, {'Alice': ALICE})

        recognizer.rebuild_database()

        assert recognizer.known_names == ['Alice']
        assert recognizer._identify(ALICE)[0] == 'Alice'
        assert recognizer._identify(BOB)[0] == 'Inconnu'
