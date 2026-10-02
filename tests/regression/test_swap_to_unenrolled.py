"""
Regression test — après un échange de pistes vers une personne non enrôlée,
le nom tombe et l'alerte « inconnu » part.

Bug d'origine : _associate_faces_to_tracks ignorait tout résultat « Inconnu ».
Quand DeepSORT échange les pistes d'Alice (enrôlée) et d'une personne non
enrôlée, la piste de l'inconnu gardait « Alice » indéfiniment : l'historique
de présence enregistrait Alice, et UnknownWatcher ne signalait jamais rien.
Fix : après RECOGNITION_UNKNOWN_STREAK visages nets successifs reconnus
« Inconnu », la piste perd son nom. Un « Inconnu » isolé ne suffit pas.

Le détecteur, le tracker et InsightFace sont remplacés par des doubles.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

import config  # noqa: E402
from core.events import UnknownWatcher  # noqa: E402
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
FPS = 30.0


def make_track(track_id):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = [100 * track_id, 100, 100 * track_id + 80, 400]
    return track


def make_tracker(track):
    """Tracker sur une seule piste ; `tracker.face_name` est ce que voit InsightFace."""
    tracker = FaceBodyTracker(MagicMock())
    x1, y1, x2, y2 = track.to_ltrb()
    tracker._detect_bodies = lambda frame: [([x1, y1, x2 - x1, y2 - y1], 0.9, 'person', {})]
    tracker.body_tracker = MagicMock()
    tracker.body_tracker.update.return_value = [track]
    tracker.face_name = 'Alice'

    def fake_recognize(frame, tracks_to_recognize):
        name = tracker.face_name
        if callable(name):
            name = name()
        confidence = 0.9 if name != 'Inconnu' else 0.2
        return [{'name': name, 'confidence': confidence, 'bbox': [0, 0, 1, 1],
                 'source_track_id': t.track_id} for t in tracks_to_recognize]

    tracker._recognize_faces_for_tracks = fake_recognize
    return tracker


def run(tracker, start, frames, watcher=None):
    """Fait tourner `frames` images ; retourne (nom final, alertes émises)."""
    alerts = []
    for frame_count in range(start, start + frames):
        persons = tracker.update(FRAME, frame_count)
        if watcher is not None:
            alerts += watcher.update([(p.track_id, p.name) for p in persons],
                                     frame_count / FPS)
    return persons[0].name, alerts


class TestSwapToUnenrolled:

    def test_name_drops_and_alert_fires(self):
        tracker = make_tracker(make_track(1))
        watcher = UnknownWatcher(config.UNKNOWN_ALERT_S)
        name, _ = run(tracker, 0, 60, watcher)
        assert name == 'Alice'

        # Échange de pistes : le corps suivi est désormais celui d'un inconnu.
        tracker.face_name = 'Inconnu'
        name, alerts = run(tracker, 60, 300, watcher)

        assert name == 'Inconnu'
        assert alerts == ['1']

    def test_name_drops_within_bounded_delay(self):
        """Visage périmé au plus après FACE_FRESHNESS_FRAMES, puis K passes de reconnaissance."""
        tracker = make_tracker(make_track(1))
        run(tracker, 0, 60)

        tracker.face_name = 'Inconnu'
        bound = (config.FACE_FRESHNESS_FRAMES
                 + (config.RECOGNITION_UNKNOWN_STREAK + 1) * config.FACE_RECOGNITION_SKIP)
        name, _ = run(tracker, 60, bound)

        assert name == 'Inconnu'

    def test_isolated_unknown_keeps_the_name(self):
        """Alice de profil, un visage sur deux non reconnu : elle garde son nom."""
        tracker = make_tracker(make_track(1))
        run(tracker, 0, 60)

        answers = iter(['Inconnu', 'Alice'] * 200)
        tracker.face_name = lambda: next(answers)
        name, _ = run(tracker, 60, 600)

        assert name == 'Alice'

    def test_unknown_streak_one_short_keeps_the_name(self):
        tracker = make_tracker(make_track(1))
        run(tracker, 0, 60)

        answers = iter(['Inconnu'] * (config.RECOGNITION_UNKNOWN_STREAK - 1) + ['Alice'] * 200)
        tracker.face_name = lambda: next(answers)
        name, _ = run(tracker, 60, 300)

        assert name == 'Alice'


def test_identity_move_is_logged_with_string_track_ids(caplog):
    """deep_sort_realtime donne des track_id en chaîne : le message ne doit pas casser.

    Bug d'origine : « passe du corps #%d au corps #%d » levait TypeError dans le
    logging à chaque correction d'usurpation ; le message était perdu.
    """
    tracker = FaceBodyTracker(MagicMock())
    tracks = []
    for i, track_id in enumerate(('1', '51')):
        track = make_track(i + 1)
        track.track_id = track_id
        tracks.append(track)
    tracker._identity_map['1'] = {'name': 'Alice', 'confidence': 0.9, 'last_face_frame': 0}
    faces = [{'name': 'Alice', 'confidence': 0.9, 'bbox': [0, 0, 1, 1], 'source_track_id': '51'}
             for _ in range(2)]

    with caplog.at_level("WARNING", logger="core.face_body_tracker"):
        tracker._associate_faces_to_tracks(tracks, faces, frame_count=10)

    assert tracker._identity_map['51']['name'] == 'Alice'
    messages = [r.getMessage() for r in caplog.records]
    assert any("passe du corps #1 au corps #51" in m for m in messages)
