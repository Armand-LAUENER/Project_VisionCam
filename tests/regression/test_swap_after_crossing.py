"""
Regression test — après un croisement, une piste nommée qui saute sur une
personne non enrôlée vue de dos perd son nom.

Cas réel (CHIRLA seq_026_camera_3, 2026-10-08) : les personnes 2 (enrôlée,
id_2) et 9 (non enrôlée) se croisent ; la piste de 9 disparaît, celle de 2
saute sur 9 et garde « id_2 » pendant 529 images (17,6 s). Vue de dos, aucun
visage de face ne fait jouer RECOGNITION_UNKNOWN_STREAK. Fix : quand une
piste qui en recouvrait une autre disparaît, la survivante perd son nom.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

import config  # noqa: E402
from core import face_body_tracker  # noqa: E402
from core.face_body_tracker import FaceBodyTracker  # noqa: E402

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    # Le module importé ici, pas sys.modules : test_web_routes y met un double.
    monkeypatch.setattr(face_body_tracker, "build_body_tracker", MagicMock)
    monkeypatch.setattr(face_body_tracker, "load_yolo", MagicMock())


def track(track_id, x):
    t = MagicMock()
    t.track_id = track_id
    t.is_confirmed.return_value = True
    t.to_ltrb.return_value = [x, 100, x + 80, 400]
    return t


def make_tracker():
    tracker = FaceBodyTracker(MagicMock())
    tracker.body_tracker = MagicMock()
    tracker._recognize_faces_for_tracks = lambda frame, tracks: []   # aucun visage exploitable
    tracker._identity_map[1] = {'name': 'Alice', 'confidence': 0.9, 'last_face_frame': 0}
    return tracker


def run(tracker, frames, *positions):
    """Chaque image, les pistes visibles : (track_id, x) ; détections identiques."""
    for frame_count, tracks in frames:
        visible = [track(tid, x) for tid, x in tracks]
        tracker.body_tracker.update.return_value = visible
        tracker._detect_bodies = (lambda boxes: (lambda f: boxes))(
            [([x, 100, 80, 300], 0.9, 'person', {}) for _, x in tracks])
        tracker.update(FRAME, frame_count)


def name_of(tracker, tid):
    return tracker._identity_map.get(tid, {}).get('name', 'Inconnu')


def test_survivor_of_a_crossing_loses_its_name():
    tracker = make_tracker()
    # Alice (piste 1) et un inconnu (piste 2) se rapprochent puis se recouvrent.
    run(tracker, [(f, [(1, 300 + 2 * f), (2, 420 - 2 * f)]) for f in range(1, 31)])
    # La piste 2 disparaît : la piste 1 a pu sauter sur l'inconnu.
    run(tracker, [(f, [(1, 360)]) for f in range(31, 40)])

    assert name_of(tracker, 1) == 'Inconnu'


def test_a_track_that_vanishes_far_from_anyone_changes_nothing():
    tracker = make_tracker()
    run(tracker, [(f, [(1, 100), (2, 900)]) for f in range(1, 20)])
    run(tracker, [(f, [(1, 100)]) for f in range(20, 30)])

    assert name_of(tracker, 1) == 'Alice'


def test_two_tracks_that_overlap_and_both_stay_keep_their_names():
    tracker = make_tracker()
    run(tracker, [(f, [(1, 300), (2, 330)]) for f in range(1, 40)])

    assert name_of(tracker, 1) == 'Alice'


def test_an_old_crossing_no_longer_counts():
    tracker = make_tracker()
    run(tracker, [(f, [(1, 300), (2, 330)]) for f in range(1, 10)])
    run(tracker, [(f, [(1, 300), (2, 900)]) for f in range(10, 10 + config.CROSSING_WINDOW_FRAMES + 5)])
    run(tracker, [(f, [(1, 300)]) for f in range(60, 70)])

    assert name_of(tracker, 1) == 'Alice'
