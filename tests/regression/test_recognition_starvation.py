"""
Regression test — au-delà de deux tracks, chacun finit par passer en reconnaissance.

Bug d'origine : update() plafonne la reconnaissance à 2 tracks par passe en
gardant les deux premiers de la liste (tracks_to_recognize[:2]). Deux tracks
qui restent « Inconnu » (inconnus, ou de dos) repassaient donc à chaque fois
en tête, et le troisième n'était jamais soumis à InsightFace.
Fix : les candidats sont triés par dernière tentative, la plus ancienne
d'abord ; un track jamais tenté passe avant tous les autres.

Le détecteur, le tracker et InsightFace sont remplacés par des doubles : seule
la sélection des tracks faite par update() est testée.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

import config  # noqa: E402
from core.face_body_tracker import FaceBodyTracker  # noqa: E402


@pytest.fixture(autouse=True)
def no_body_tracker(monkeypatch):
    """Ni DeepSORT ni son embedder : aucun test de ce fichier ne s'en sert.

    Remplacé le temps du test plutôt que dans sys.modules : un faux
    deep_sort_realtime y resterait pour les fichiers de test suivants
    (cf. commit cadc2a7).
    """
    monkeypatch.setattr("core.face_body_tracker.build_body_tracker", MagicMock)


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def make_track(track_id):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = [100 * track_id, 100, 100 * track_id + 80, 400]
    return track


def make_tracker(tracks, known):
    """Tracker dont InsightFace reconnaît `known` ({track_id: nom}) et rien d'autre."""
    tracker = FaceBodyTracker(MagicMock())
    # Une détection sur chaque piste : seules les pistes détectées sur l'image
    # sont soumises à la reconnaissance.
    tracker._detect_bodies = lambda frame: [
        ([x1, y1, x2 - x1, y2 - y1], 0.9, 'person', {})
        for x1, y1, x2, y2 in (t.to_ltrb() for t in tracks)
    ]
    tracker.body_tracker = MagicMock()
    tracker.body_tracker.update.return_value = tracks
    tracker.submitted = []

    def fake_recognize(frame, tracks_to_recognize):
        tracker.submitted.append([t.track_id for t in tracks_to_recognize])
        return [{'name': known[t.track_id], 'confidence': 0.9, 'bbox': [0, 0, 1, 1],
                 'source_track_id': t.track_id}
                for t in tracks_to_recognize if t.track_id in known]

    tracker._recognize_faces_for_tracks = fake_recognize
    return tracker


def run_recognition_passes(tracker, passes):
    for i in range(passes):
        results = tracker.update(FRAME, i * config.FACE_RECOGNITION_SKIP)
    return results


class TestRecognitionStarvation:

    def test_third_track_is_eventually_recognized(self):
        tracks = [make_track(1), make_track(2), make_track(3)]
        tracker = make_tracker(tracks, known={3: 'Bob'})

        results = run_recognition_passes(tracker, passes=10)

        assert {p.track_id: p.name for p in results}[3] == 'Bob'

    def test_every_unknown_track_is_submitted_in_turn(self):
        tracks = [make_track(tid) for tid in (1, 2, 3, 4, 5)]
        tracker = make_tracker(tracks, known={})

        run_recognition_passes(tracker, passes=3)

        assert all(len(ids) <= 2 for ids in tracker.submitted)
        assert {tid for ids in tracker.submitted for tid in ids} == {1, 2, 3, 4, 5}

    def test_update_records_stage_timings(self):
        """La page Diagnostic lit ces durées ; la reconnaissance n'y figure que si elle a tourné."""
        tracker = make_tracker([make_track(1)], known={})

        tracker.update(FRAME, 1)
        assert set(tracker.last_timings) == {'detection', 'tracking'}

        tracker.update(FRAME, config.FACE_RECOGNITION_SKIP)
        assert set(tracker.last_timings) == {'detection', 'tracking', 'recognition'}
        assert all(v >= 0 for v in tracker.last_timings.values())

    def test_new_track_goes_before_tracks_already_tried(self):
        tracks = [make_track(1), make_track(2)]
        tracker = make_tracker(tracks, known={})
        run_recognition_passes(tracker, passes=2)

        tracks.append(make_track(9))
        tracker.update(FRAME, 2 * config.FACE_RECOGNITION_SKIP)

        assert 9 in tracker.submitted[-1]
