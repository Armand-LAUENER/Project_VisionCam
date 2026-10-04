"""
Regression test — un nom ne saute plus vers une piste voisine pendant que sa
porteuse est visible et vient d'être reconnue.

Bug d'origine (roadmap 1.8, cause 3) : l'anti-clonage retirait son nom à
toute piste dès qu'une autre l'obtenait par consensus. Sur CHIRLA seq_026, le
visage d'une voisine (ou le crop d'une boîte partielle, ou d'une piste
dédoublée) était reconnu par moments sous le nom de la personne assise : le
nom passait d'une piste à l'autre puis revenait, 59 fois sur une séquence.
Fix : une piste visible dont le visage a été reconnu sous ce nom depuis moins
de IDENTITY_PROTECT_FRAMES le garde, sauf reconnaissance nettement plus sûre
(IDENTITY_STEAL_MARGIN). En roue libre ou sans visage récent, elle le perd
comme avant.
"""

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

import config  # noqa: E402
from core.face_body_tracker import FaceBodyTracker  # noqa: E402

NOW = 1000


@pytest.fixture(autouse=True)
def no_body_tracker(monkeypatch):
    """Ni DeepSORT ni son embedder : aucun test de ce fichier ne s'en sert."""
    monkeypatch.setattr("core.face_body_tracker.build_body_tracker", MagicMock)


def make_track(track_id, x):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = [x, 100, x + 150, 400]
    return track


def claim(tracker, tracks, track_id, name, confidence):
    """Deux reconnaissances d'affilée : le consensus du vote est atteint."""
    faces = [{'name': name, 'confidence': confidence, 'bbox': [0, 0, 1, 1],
              'source_track_id': track_id} for _ in range(2)]
    tracker._associate_faces_to_tracks(tracks, faces, NOW)


def names(tracker):
    return {tid: identity['name'] for tid, identity in tracker._identity_map.items()}


@pytest.fixture
def scene():
    """Piste 1 : Alice, reconnue il y a `age` images. Piste 2 : la voisine."""
    def build(age, owner_visible=True, owner_confidence=0.70):
        tracker = FaceBodyTracker(MagicMock())
        tracker._identity_map[1] = {'name': 'Alice', 'confidence': owner_confidence,
                                    'last_face_frame': NOW - age}
        owner, neighbour = make_track(1, 100), make_track(2, 400)
        visible = [owner, neighbour] if owner_visible else [neighbour]
        return tracker, visible
    return build


class TestIdentityPingPong:

    def test_fresh_visible_owner_keeps_its_name(self, scene):
        tracker, visible = scene(age=10)

        claim(tracker, visible, 2, 'Alice', 0.66)

        assert names(tracker) == {1: 'Alice'}

    def test_much_surer_recognition_takes_the_name(self, scene):
        tracker, visible = scene(age=10)

        claim(tracker, visible, 2, 'Alice', 0.70 + config.IDENTITY_STEAL_MARGIN + 0.05)

        assert names(tracker) == {1: 'Inconnu', 2: 'Alice'}

    def test_owner_without_recent_face_loses_it(self, scene):
        tracker, visible = scene(age=config.IDENTITY_PROTECT_FRAMES + 1)

        claim(tracker, visible, 2, 'Alice', 0.60)

        assert names(tracker) == {1: 'Inconnu', 2: 'Alice'}

    def test_coasting_owner_loses_it(self, scene):
        """Porteuse en roue libre : la personne est sans doute passée sur la nouvelle piste."""
        tracker, visible = scene(age=10, owner_visible=False)

        claim(tracker, visible, 2, 'Alice', 0.60)

        assert names(tracker) == {1: 'Inconnu', 2: 'Alice'}

    def test_owner_reconfirmation_is_untouched(self, scene):
        tracker, visible = scene(age=10)

        claim(tracker, visible, 1, 'Alice', 0.72)

        assert names(tracker) == {1: 'Alice'}
        assert tracker._identity_map[1]['last_face_frame'] == NOW
