"""
Regression test — un visage voisin dans le crop de tête ne prête pas son identité.

Bug d'origine : _recognize_faces_for_tracks reconnaissait TOUS les visages
trouvés dans le crop de tête d'un track et leur donnait à tous le
source_track_id de ce track. Dans une scène dense, le crop attrape les visages
des voisins (jusqu'à 5 par paire de crops sur MOT17) : un voisin connu
finissait par donner son nom au mauvais corps, et chaque voisin coûtait une
inférence de reconnaissance (~10 ms).
Fix : FaceRecognizer.recognize_center_face ne reconnaît que le visage le plus
proche du centre du crop, qui est centré sur les keypoints faciaux du track.

Le vrai FaceRecognizer est utilisé, seuls ses modèles InsightFace sont
remplacés par des doubles : le test couvre donc bien le chemin tracker →
reconnaissance → association.
"""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('deep_sort_realtime', MagicMock())
sys.modules.setdefault('deep_sort_realtime.deepsort_tracker', MagicMock())
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

from core.face_body_tracker import FaceBodyTracker  # noqa: E402
from core.face_recognition import FaceRecognizer  # noqa: E402

EMBEDDINGS = {
    'Bob': np.eye(512, dtype=np.float32)[0],
    'inconnu': np.eye(512, dtype=np.float32)[1],
}

# Cinq keypoints faciaux étalés sur 180 px, loin des bords d'une frame 720p :
# le crop fait 576 px de côté, son centre tombe à (288, 288).
FACE_KPS = [(560.0, 360.0, 0.99), (740.0, 360.0, 0.99), (650.0, 380.0, 0.99),
            (600.0, 400.0, 0.99), (700.0, 400.0, 0.99)]
CENTER_BBOX = [238.0, 238.0, 338.0, 338.0]
NEIGHBOR_BBOX = [10.0, 10.0, 90.0, 110.0]


class FakeFaceAnalysis:
    """Double d'InsightFace : visages placés à la main, embeddings par identité."""

    def __init__(self, faces):
        # faces : [(bbox, identité)]
        self.faces = faces
        self.recognition_calls = 0
        self.det_model = SimpleNamespace(detect=self._detect)
        self.models = {'recognition': SimpleNamespace(get=self._recognize)}

    @staticmethod
    def _kps(bbox):
        """Cinq keypoints déduits de la bbox ; le premier encode la position."""
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return np.array([[cx, cy]] * 5, dtype=np.float32)

    def _identity_at(self, kps):
        return next(identity for bbox, identity in self.faces
                    if np.allclose(self._kps(bbox), kps))

    def _detect(self, img, max_num=0, metric='default'):
        bboxes = np.array([[*bbox, 0.9] for bbox, _ in self.faces], dtype=np.float32).reshape(-1, 5)
        kpss = np.array([self._kps(bbox) for bbox, _ in self.faces], dtype=np.float32).reshape(-1, 5, 2)
        return bboxes, kpss

    def _recognize(self, img, face):
        self.recognition_calls += 1
        face.embedding = EMBEDDINGS[self._identity_at(face.kps)]
        return face.embedding

    def get(self, img, max_num=0):
        """Chemin d'avant le correctif : tous les visages, tous reconnus."""
        faces = []
        for bbox, _ in self.faces:
            face = SimpleNamespace(bbox=np.array(bbox), kps=self._kps(bbox), embedding=None)
            self._recognize(img, face)
            faces.append(face)
        return faces


def make_recognizer(faces):
    recognizer = object.__new__(FaceRecognizer)
    recognizer.threshold = 0.45
    recognizer.known_embeddings = [EMBEDDINGS['Bob']]
    recognizer.known_names = ['Bob']
    recognizer._lock = threading.Lock()
    recognizer.app = FakeFaceAnalysis(faces)
    return recognizer


def make_track():
    track = MagicMock()
    track.track_id = 1
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = [366, 210, 899, 709]
    return track


def run_recognition_rounds(recognizer, rounds=3):
    """Plusieurs passes de reconnaissance, de quoi atteindre le consensus du vote."""
    tracker = FaceBodyTracker(recognizer)
    tracker._face_kps_map[1] = FACE_KPS
    track = make_track()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for frame_count in range(1, rounds + 1):
        faces = tracker._recognize_faces_for_tracks(frame, [track])
        tracker._associate_faces_to_tracks([track], faces, frame_count)
    return tracker


class TestNeighborFaceInHeadCrop:

    def test_known_neighbor_does_not_lend_its_identity(self):
        recognizer = make_recognizer([(NEIGHBOR_BBOX, 'Bob'), (CENTER_BBOX, 'inconnu')])

        tracker = run_recognition_rounds(recognizer)

        assert tracker._identity_map.get(1, {}).get('name') != 'Bob'

    def test_only_the_center_face_is_recognized(self):
        recognizer = make_recognizer([(NEIGHBOR_BBOX, 'Bob'), (CENTER_BBOX, 'inconnu')])

        run_recognition_rounds(recognizer, rounds=1)

        assert recognizer.app.recognition_calls == 1

    def test_center_face_still_identifies_the_track(self):
        """Garde-fou : le correctif ne doit pas empêcher toute reconnaissance."""
        recognizer = make_recognizer([(NEIGHBOR_BBOX, 'inconnu'), (CENTER_BBOX, 'Bob')])

        tracker = run_recognition_rounds(recognizer)

        assert tracker._identity_map[1]['name'] == 'Bob'


class TestRecognizeCenterFace:

    def test_no_face_returns_none(self):
        recognizer = make_recognizer([])

        assert recognizer.recognize_center_face(np.zeros((576, 576, 3), dtype=np.uint8)) is None

    def test_bbox_is_clipped_to_the_crop(self):
        recognizer = make_recognizer([([-20.0, 250.0, 700.0, 330.0], 'Bob')])

        face = recognizer.recognize_center_face(np.zeros((576, 576, 3), dtype=np.uint8))

        assert face['bbox'] == [0, 250, 576, 330]
        assert face['name'] == 'Bob'
        assert face['confidence'] == pytest.approx(1.0)
