"""
tests/test_pose_keypoint_propagation.py

Teste la remontée des keypoints d'orientation (COCO 0-6) depuis la détection
YOLO jusqu'à TrackedPerson.pose_kps, qui alimente POSE_SOURCE="yolo".

Le chemin testé : _detect_bodies pose 'pose_kps' dans others → _update_nose_map
l'associe au bon track par IoU → _build_results l'expose sur TrackedPerson.

Contient aussi un garde-fou de non-régression : 'face_kps' doit rester à
exactement 5 éléments. Cette liste sert à compter les keypoints faciaux visibles
(POSE_FACE_KP_MIN_VISIBLE) ; y ajouter les épaules fausserait ce comptage.

Mêmes mocks sys.modules que test_face_body_association.py : aucun GPU chargé.

Lancer : pytest tests/test_pose_keypoint_propagation.py -v
"""

import sys
from unittest.mock import MagicMock

import pytest

_mock_ultralytics = MagicMock()
_mock_ultralytics.YOLO = MagicMock(return_value=MagicMock())
sys.modules.setdefault('ultralytics', _mock_ultralytics)

_mock_deepsort_module = MagicMock()
_mock_deepsort_module.DeepSort = MagicMock(return_value=MagicMock())
sys.modules.setdefault('deep_sort_realtime', MagicMock())
sys.modules.setdefault('deep_sort_realtime.deepsort_tracker', _mock_deepsort_module)

sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

from core.face_body_tracker import FaceBodyTracker  # noqa: E402

# 7 keypoints COCO : nez, 2 yeux, 2 oreilles, 2 épaules
KPS_7 = [(100.0, 50.0, 0.9), (105.0, 45.0, 0.8), (95.0, 45.0, 0.8),
         (115.0, 45.0, 0.7), (85.0, 45.0, 0.7),
         (140.0, 100.0, 0.9), (60.0, 100.0, 0.9)]


def make_track(track_id, ltrb):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = ltrb
    return track


def make_detection(bbox_ltwh, pose_kps=KPS_7):
    """Détection au format DeepSORT : ([x, y, w, h], conf, 'person', others)."""
    others = {
        'nose': pose_kps[0],
        'face_kps': pose_kps[:5],
        'pose_kps': pose_kps,
    }
    return (bbox_ltwh, 0.9, 'person', others)


@pytest.fixture
def tracker():
    face_recognizer = MagicMock()
    face_recognizer.known_names = ['Armand']
    return FaceBodyTracker(face_recognizer)


class TestPropagationDesKeypoints:

    def test_pose_kps_atteint_tracked_person(self, tracker):
        track = make_track(1, [0.0, 0.0, 200.0, 400.0])
        detection = make_detection([0, 0, 200, 400])

        tracker._update_nose_map([track], [detection])
        results = tracker._build_results([track])

        assert len(results) == 1
        assert results[0].pose_kps == KPS_7

    def test_les_epaules_sont_incluses(self, tracker):
        """Sans les indices 5 et 6, l'estimateur ne peut pas calculer l'écart d'épaules."""
        track = make_track(1, [0.0, 0.0, 200.0, 400.0])
        tracker._update_nose_map([track], [make_detection([0, 0, 200, 400])])

        assert len(tracker._pose_kps_map[1]) == 7

    def test_iou_insuffisant_ne_pollue_pas_la_map(self, tracker):
        """Détection trop éloignée (IoU <= 0.3) : aucun keypoint ne doit être associé."""
        track = make_track(1, [0.0, 0.0, 100.0, 100.0])
        detection = make_detection([500, 500, 100, 100])

        tracker._update_nose_map([track], [detection])

        assert 1 not in tracker._pose_kps_map

    def test_track_disparu_est_purge(self, tracker):
        track = make_track(1, [0.0, 0.0, 200.0, 400.0])
        tracker._update_nose_map([track], [make_detection([0, 0, 200, 400])])
        assert 1 in tracker._pose_kps_map

        # Frame suivante : le track 1 n'est plus actif.
        tracker._update_nose_map([], [])
        assert 1 not in tracker._pose_kps_map

    def test_sans_keypoints_pose_kps_reste_none(self, tracker):
        """Une détection sans keypoints ne doit pas faire échouer _build_results."""
        track = make_track(1, [0.0, 0.0, 200.0, 400.0])
        detection = ([0, 0, 200, 400], 0.9, 'person', {'nose': None, 'face_kps': None})

        tracker._update_nose_map([track], [detection])
        results = tracker._build_results([track])

        assert results[0].pose_kps is None


class TestNonRegressionFaceKps:

    def test_face_kps_reste_a_cinq_elements(self, tracker):
        """
        face_kps alimente le comptage POSE_FACE_KP_MIN_VISIBLE : il doit rester
        limité aux keypoints du visage (COCO 0-4), épaules exclues.
        """
        track = make_track(1, [0.0, 0.0, 200.0, 400.0])
        tracker._update_nose_map([track], [make_detection([0, 0, 200, 400])])

        assert len(tracker._face_kps_map[1]) == 5
