"""
Regression test — une identité établie ne se renomme pas sur 2 votes sur 3.

Cas réel (CHIRLA seq_026_camera_3, moteur YOLO dynamique, 2026-10-08) : la
personne 6, nommée correctement depuis des minutes, devient « id_5 » pendant
275 images après 2 reconnaissances erronées. Fix : renommer une piste déjà
nommée exige RENAME_MIN_VOTES voix sur ses RENAME_WINDOW dernières
reconnaissances ; nommer une piste « Inconnu » reste à 2 votes sur 3.
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
    monkeypatch.setattr(face_body_tracker, "build_body_tracker", MagicMock)
    monkeypatch.setattr(face_body_tracker, "load_yolo", MagicMock())


def make_tracker(names):
    """Une piste visible ; chaque passe de reconnaissance renvoie le nom suivant."""
    track = MagicMock()
    track.track_id = 1
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = [100, 100, 180, 400]
    tracker = FaceBodyTracker(MagicMock())
    tracker.body_tracker = MagicMock()
    tracker.body_tracker.update.return_value = [track]
    tracker._detect_bodies = lambda frame: [([100, 100, 80, 300], 0.9, 'person', {})]
    answers = iter(names)
    tracker._recognize_faces_for_tracks = lambda frame, tracks: [
        {'name': next(answers), 'confidence': 0.9, 'bbox': [0, 0, 1, 1],
         'source_track_id': 1, 'frontal': True}]
    return tracker


def drop_name(tracker):
    """Comme une perte réelle (série d'« Inconnu », croisement) : nom et votes."""
    tracker._remember_dropped(1)
    tracker._identity_map[1] = {'name': 'Inconnu', 'confidence': 0.0, 'last_face_frame': -1}
    tracker._vote_buffer.pop(1, None)


def name_after(tracker, passes):
    # Une passe toutes les FACE_RECOGNITION_SKIP images ; le visage reste frais
    # pour que chaque passe soumette la piste.
    for i in range(passes):
        tracker.update(FRAME, (i + 1) * config.FACE_RECOGNITION_SKIP)
        tracker._identity_map.get(1, {})['last_face_frame'] = -10**6
    return tracker._identity_map.get(1, {}).get('name', 'Inconnu')


def test_an_unknown_track_is_named_on_two_votes_out_of_three():
    assert name_after(make_tracker(['Alice', 'Alice']), 2) == 'Alice'


def test_two_wrong_recognitions_do_not_rename_an_established_name():
    tracker = make_tracker(['Alice'] * 6 + ['Bob', 'Bob'] + ['Alice'] * 4)

    assert name_after(tracker, 8) == 'Alice'
    assert name_after(tracker, 4) == 'Alice'


def test_a_name_backed_by_enough_votes_does_rename():
    tracker = make_tracker(['Alice'] * 6 + ['Bob'] * config.RENAME_MIN_VOTES)

    assert name_after(tracker, 6 + config.RENAME_MIN_VOTES) == 'Bob'


def test_a_name_just_lost_still_counts_as_established():
    """Cas réel : id_6 retombe (série d'« Inconnu »), puis 2 erreurs « id_5 »
    nommaient la même personne id_5. Le nom perdu compte encore un moment."""
    tracker = make_tracker(['Alice'] * 4 + ['Bob', 'Bob'])
    name_after(tracker, 4)
    drop_name(tracker)

    assert name_after(tracker, 2) == 'Inconnu'


def test_the_lost_name_comes_back_on_the_usual_two_votes():
    tracker = make_tracker(['Alice'] * 4 + ['Alice', 'Alice'])
    name_after(tracker, 4)
    drop_name(tracker)

    assert name_after(tracker, 2) == 'Alice'
