"""
Regression test — une piste en roue libre n'est ni affichée ni reconnue.

Bug d'origine : FaceBodyTracker traitait toutes les pistes confirmées comme
présentes. Une personne occultée ou sortie du champ gardait une boîte prédite
par Kalman pendant max_age images (70) : boîte fantôme à l'écran, et crop
visage soumis à InsightFace alors qu'il montrait l'obstacle. Sur MOT17
(séquences de validation), ces fantômes représentaient l'essentiel des faux
positifs : MOTA 21,5 % contre 44,2 % une fois masqués.
De plus, l'association piste → détection prenait la meilleure IoU pour chaque
piste indépendamment : une piste en roue libre dont la boîte chevauchait une
autre personne récupérait ses keypoints.
Fix : appariement un-pour-un ; seules les pistes appariées à une détection de
l'image sont affichées et reconnues. Les autres gardent leur identité en
mémoire jusqu'à max_age.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

import config  # noqa: E402
from core.face_body_tracker import FaceBodyTracker, match_tracks_to_detections  # noqa: E402


@pytest.fixture(autouse=True)
def no_body_tracker(monkeypatch):
    """Ni DeepSORT ni son embedder : les pistes sont fournies à la main."""
    monkeypatch.setattr("core.face_body_tracker.build_body_tracker", MagicMock)


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
ALICE_BOX = [100, 100, 180, 400]
BOB_BOX = [600, 100, 680, 400]


def make_track(track_id, ltrb):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = ltrb
    return track


def detection(ltrb, nose_x=None):
    x1, y1, x2, y2 = ltrb
    kps = [(nose_x if nose_x is not None else (x1 + x2) / 2, y1 + 20, 0.9)] * 7
    return ([x1, y1, x2 - x1, y2 - y1], 0.9, 'person',
            {'nose': kps[0], 'face_kps': kps[:5], 'pose_kps': kps})


def make_tracker(tracks, detections, known):
    tracker = FaceBodyTracker(MagicMock())
    tracker.body_tracker = MagicMock()
    tracker.body_tracker.update.return_value = tracks
    tracker.frame_detections = detections
    tracker._detect_bodies = lambda frame: tracker.frame_detections
    tracker.submitted = []

    def fake_recognize(frame, tracks_to_recognize):
        tracker.submitted.append([t.track_id for t in tracks_to_recognize])
        return [{'name': known[t.track_id], 'confidence': 0.9, 'bbox': [0, 0, 1, 1],
                 'source_track_id': t.track_id}
                for t in tracks_to_recognize if t.track_id in known]

    tracker._recognize_faces_for_tracks = fake_recognize
    return tracker


def recognition_frame(i):
    return i * config.FACE_RECOGNITION_SKIP


class TestCoastingTracks:

    def test_undetected_track_is_not_displayed(self):
        alice, bob = make_track(1, ALICE_BOX), make_track(2, BOB_BOX)
        tracker = make_tracker([alice, bob], [detection(ALICE_BOX)], known={})

        results = tracker.update(FRAME, 1)

        assert [p.track_id for p in results] == [1]

    def test_identity_survives_the_occlusion(self):
        alice = make_track(1, ALICE_BOX)
        tracker = make_tracker([alice], [detection(ALICE_BOX)], known={1: 'Alice'})
        for i in range(1, 4):
            tracker.update(FRAME, recognition_frame(i))

        tracker.frame_detections = []
        hidden = tracker.update(FRAME, recognition_frame(4))
        tracker.frame_detections = [detection(ALICE_BOX)]
        back = tracker.update(FRAME, recognition_frame(4) + 1)

        assert hidden == []
        assert [(p.track_id, p.name) for p in back] == [(1, 'Alice')]

    def test_undetected_track_is_not_sent_to_recognition(self):
        alice, bob = make_track(1, ALICE_BOX), make_track(2, BOB_BOX)
        tracker = make_tracker([alice, bob], [detection(ALICE_BOX)], known={})

        for i in range(1, 4):
            tracker.update(FRAME, recognition_frame(i))

        assert all(ids == [1] for ids in tracker.submitted)

    def test_coasting_track_does_not_take_a_neighbours_keypoints(self):
        # La boîte prédite de Bob (roue libre) chevauche Alice, détectée.
        alice = make_track(1, ALICE_BOX)
        ghost_bob = make_track(2, [110, 110, 190, 410])
        tracker = make_tracker([alice, ghost_bob], [detection(ALICE_BOX, nose_x=140)], known={})

        results = tracker.update(FRAME, 1)

        assert [p.track_id for p in results] == [1]
        assert 2 not in tracker._nose_map


class TestMatchTracksToDetections:

    def test_each_detection_serves_one_track_best_iou_first(self):
        tracks = [[110, 110, 190, 410], [100, 100, 180, 400]]
        detections = [[100, 100, 80, 300]]

        assert match_tracks_to_detections(tracks, detections) == {1: 0}

    def test_low_overlap_is_not_a_match(self):
        assert match_tracks_to_detections([[0, 0, 100, 100]], [[90, 90, 100, 100]]) == {}

    def test_every_track_gets_its_own_detection(self):
        tracks = [ALICE_BOX, BOB_BOX]
        detections = [[600, 100, 80, 300], [100, 100, 80, 300]]

        assert match_tracks_to_detections(tracks, detections) == {0: 1, 1: 0}
