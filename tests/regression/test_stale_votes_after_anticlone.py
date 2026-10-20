"""
Regression test — l'anti-clonage vide le tampon de votes de la piste dépouillée.

Bug d'origine : _associate_faces_to_tracks retirait son nom à une piste (anti
-clonage : une identité ne peut appartenir qu'à un corps) sans vider son
_vote_buffer, contrairement à _suspend et _count_unknown_face. Les anciennes
voix restaient : un visage reconnu plus tard — même sous un autre nom —
complétait une majorité avec elles et renommait la piste, désormais sur une
autre personne. Sur seq_026_camera_3 (une image sur deux), la piste #132
reprenait ainsi id_2 sur un non-enrôlé.

Fix : purger _vote_buffer (et _unknown_streak) comme _suspend quand l'anti
-clonage dépouille une piste.

InsightFace et le détecteur ne servent pas : on appelle _associate_faces_to_tracks
directement, comme tests/regression/test_swap_to_unenrolled.py (dernier test).
"""

import sys
from collections import deque
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

from core.face_body_tracker import FaceBodyTracker  # noqa: E402


def make_track(track_id):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = [100, 100, 180, 400]
    return track


def face(name, source_track_id, confidence=0.9):
    return {'name': name, 'confidence': confidence, 'bbox': [0, 0, 1, 1],
            'source_track_id': source_track_id, 'frontal': True}


@pytest.fixture
def tracker_with_stolen_name():
    """La piste '1' tient 'Alice' (vieux visage, non protégée), avec des voix.

    La piste '2' reconnue 'Alice' déclenche l'anti-clonage : 'Alice' passe à
    '2', '1' repasse « Inconnu ». Retourne le tracker après ce croisement.
    """
    tracker = FaceBodyTracker(MagicMock())
    tracks = [make_track('1'), make_track('2')]
    # '1' établie bien avant (hors IDENTITY_PROTECT_FRAMES), tampon plein.
    tracker._identity_map['1'] = {'name': 'Alice', 'confidence': 0.9,
                                  'last_face_frame': -1000}
    tracker._vote_buffer['1'] = deque([('Alice', 0.9), ('Alice', 0.9)],
                                      maxlen=tracker.VOTE_WINDOW)
    tracker._associate_faces_to_tracks(tracks, [face('Alice', '2')] * 2, frame_count=10)
    assert tracker._identity_map['2']['name'] == 'Alice'
    assert tracker._identity_map['1']['name'] == 'Inconnu'
    return tracker, tracks


def test_anticlone_clears_the_vote_buffer(tracker_with_stolen_name):
    tracker, _ = tracker_with_stolen_name
    assert not tracker._vote_buffer.get('1')


def test_stale_votes_do_not_rename_after_anticlone(tracker_with_stolen_name):
    """Une piste dépouillée ne reprend pas son ancien nom sur une seule voix.

    La piste '2' a disparu : plus personne ne protège 'Alice'. Un unique visage
    « Bob » arrive sur '1'. Avec le bug, ses deux 'Alice' périmés font encore la
    majorité et '1' redevient 'Alice'. Tampon vidé, « Bob » est seul : pas de
    consensus, '1' reste « Inconnu ».
    """
    tracker, _ = tracker_with_stolen_name
    tracker._associate_faces_to_tracks([make_track('1')], [face('Bob', '1')],
                                       frame_count=11)
    assert tracker._identity_map['1']['name'] == 'Inconnu'
