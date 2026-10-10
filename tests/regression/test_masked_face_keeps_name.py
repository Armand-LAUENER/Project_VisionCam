"""
Regression test — une personne reconnue qui se masque le visage garde son nom.

Bug d'origine (webcam, 2026-10-10) : Armand met un masque puis une cagoule
sans quitter le champ. Sa piste reste la #1, mais SCRFD trouve encore un
visage de face qu'ArcFace juge « Inconnu » : après RECOGNITION_UNKNOWN_STREAK
visages, le nom tombait (« 3 visages « Inconnu » d'affilée, 'Armand' retiré »),
comme pour un échange de pistes vers une personne non enrôlée.
Fix : l'« Inconnu » est ignoré tant que la piste est restée seule (pas de
voisine, pas d'échange possible) et dans le même corps (couleur des
vêtements, BODY_KEEP_SIMILARITY) depuis son dernier visage reconnu.

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
from core.face_body_tracker import FaceBodyTracker  # noqa: E402


@pytest.fixture(autouse=True)
def no_body_tracker(monkeypatch):
    """Ni DeepSORT ni son embedder (cf. test_swap_to_unenrolled)."""
    monkeypatch.setattr("core.face_body_tracker.build_body_tracker", MagicMock)


ARMAND_BOX = (100, 100, 180, 400)
NEIGHBOUR_BOX = (200, 100, 280, 400)     # centres à 100 px, 1,25 largeur


def scene(armand_bgr, neighbour_bgr=None):
    """Image unie aux couleurs des vêtements de chaque personne."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    x1, y1, x2, y2 = ARMAND_BOX
    frame[y1:y2, x1:x2] = armand_bgr
    if neighbour_bgr is not None:
        x1, y1, x2, y2 = NEIGHBOUR_BOX
        frame[y1:y2, x1:x2] = neighbour_bgr
    return frame


RED, BLUE = (0, 0, 200), (200, 0, 0)


def make_track(track_id, box):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltrb.return_value = list(box)
    return track


def make_tracker():
    """Piste #1 (Armand) ; `tracker.tracks` et `tracker.face_name` pilotent la scène."""
    tracker = FaceBodyTracker(MagicMock())
    tracker.tracks = [make_track(1, ARMAND_BOX)]
    tracker._detect_bodies = lambda frame: [
        ([t.to_ltrb()[0], t.to_ltrb()[1], 80, 300], 0.9, 'person', {}) for t in tracker.tracks]
    tracker.body_tracker = MagicMock()
    tracker.body_tracker.update.side_effect = lambda dets, frame: tracker.tracks
    tracker.face_name = 'Armand'

    def fake_recognize(frame, tracks_to_recognize):
        name = tracker.face_name
        return [{'name': name if t.track_id == 1 else 'Inconnu',
                 'confidence': 0.9 if name != 'Inconnu' else 0.2,
                 'bbox': [0, 0, 1, 1], 'source_track_id': t.track_id, 'frontal': True}
                for t in tracks_to_recognize]

    tracker._recognize_faces_for_tracks = fake_recognize
    return tracker


def run(tracker, frame, start, frames):
    for frame_count in range(start, start + frames):
        persons = tracker.update(frame, frame_count)
    return next(p.name for p in persons if p.track_id == 1)


def test_masked_face_keeps_the_name():
    tracker = make_tracker()
    assert run(tracker, scene(RED), 0, 300) == 'Armand'   # nom établi : 10 s de face

    tracker.face_name = 'Inconnu'            # masque : visage de face non reconnu
    assert run(tracker, scene(RED), 300, 900) == 'Armand'


def test_masked_face_is_shown_as_body_tracking():
    """Le nom tenu sans visage récent : last_face_frame vieillit (boîte « corps »)."""
    tracker = make_tracker()
    run(tracker, scene(RED), 0, 300)
    tracker.face_name = 'Inconnu'
    run(tracker, scene(RED), 300, 300)

    person = tracker.update(scene(RED), 600)[0]
    assert person.name == 'Armand'
    assert 600 - person.last_face_frame > config.FACE_FRESHNESS_FRAMES


def test_body_change_drops_the_name():
    """Autre corps sous la piste (échange sans voisine visible) : le nom tombe."""
    tracker = make_tracker()
    run(tracker, scene(RED), 0, 60)

    tracker.face_name = 'Inconnu'
    assert run(tracker, scene(BLUE), 60, 300) == 'Inconnu'


def test_neighbour_breaks_isolation():
    """Une voisine s'est approchée : même vêtements ou pas, l'« Inconnu » compte."""
    tracker = make_tracker()
    run(tracker, scene(RED), 0, 60)

    tracker.tracks.append(make_track(2, NEIGHBOUR_BOX))
    run(tracker, scene(RED, RED), 60, 5)
    tracker.tracks.pop()

    tracker.face_name = 'Inconnu'
    assert run(tracker, scene(RED), 65, 300) == 'Inconnu'


def test_isolation_returns_with_a_recognised_face():
    tracker = make_tracker()
    run(tracker, scene(RED), 0, 60)
    tracker.tracks.append(make_track(2, NEIGHBOUR_BOX))
    run(tracker, scene(RED, BLUE), 60, 5)
    tracker.tracks.pop()

    run(tracker, scene(RED), 65, 300)         # visage reconnu à nouveau, seul
    tracker.face_name = 'Inconnu'
    assert run(tracker, scene(RED), 365, 600) == 'Armand'


def test_disabled_restores_the_unknown_streak(monkeypatch):
    monkeypatch.setattr(config, 'BODY_KEEP_SIMILARITY', 0.0)
    tracker = make_tracker()
    run(tracker, scene(RED), 0, 60)

    tracker.face_name = 'Inconnu'
    assert run(tracker, scene(RED), 60, 300) == 'Inconnu'


def test_name_from_few_votes_is_not_protected():
    """Régression (CHIRLA seq_026) : un nom donné à tort par 2 votes doit
    toujours tomber sur des visages « Inconnu », même corps et piste seule."""
    tracker = make_tracker()
    answers = iter(['Armand', 'Armand'] + ['Inconnu'] * 500)
    tracker._recognize_faces_for_tracks = lambda frame, tracks: [
        {'name': (name := next(answers)), 'confidence': 0.9 if name != 'Inconnu' else 0.2,
         'bbox': [0, 0, 1, 1], 'source_track_id': t.track_id, 'frontal': True}
        for t in tracks]
    assert run(tracker, scene(RED), 0, 2 * config.FACE_RECOGNITION_SKIP) == 'Armand'

    assert run(tracker, scene(RED), 2 * config.FACE_RECOGNITION_SKIP, 300) == 'Inconnu'


def test_alone_mode_any_visible_person_breaks_isolation(monkeypatch):
    """BODY_NEIGHBOUR_WIDTHS 0 : seule dans l'image, une personne au loin suffit."""
    monkeypatch.setattr(config, 'BODY_NEIGHBOUR_WIDTHS', 0.0)
    tracker = make_tracker()
    run(tracker, scene(RED), 0, 300)

    tracker.tracks.append(make_track(2, (1000, 100, 1080, 400)))
    run(tracker, scene(RED), 300, 5)
    tracker.tracks.pop()

    tracker.face_name = 'Inconnu'
    assert run(tracker, scene(RED), 305, 300) == 'Inconnu'


def test_alone_mode_keeps_the_name_when_alone(monkeypatch):
    monkeypatch.setattr(config, 'BODY_NEIGHBOUR_WIDTHS', 0.0)
    tracker = make_tracker()
    run(tracker, scene(RED), 0, 300)

    tracker.face_name = 'Inconnu'
    assert run(tracker, scene(RED), 300, 900) == 'Armand'
