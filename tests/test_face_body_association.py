"""
tests/test_face_body_association.py

Teste la logique d'association géométrique Visage ↔ Corps de FaceBodyTracker.

Aucun modèle GPU n'est chargé : ultralytics, deep_sort_realtime et FaceRecognizer
sont mockés dans sys.modules AVANT l'import du module, pour que les tests tournent
même si ces bibliothèques ne sont pas encore installées.

Lancer : pytest tests/test_face_body_association.py -v
"""

import sys
import pytest
from unittest.mock import MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Injection des dépendances lourdes dans sys.modules avant tout import
# ─────────────────────────────────────────────────────────────────────────────

# ultralytics.YOLO
_mock_ultralytics = MagicMock()
_mock_ultralytics.YOLO = MagicMock(return_value=MagicMock())
sys.modules.setdefault('ultralytics', _mock_ultralytics)

# deep_sort_realtime.deepsort_tracker.DeepSort
_mock_deepsort_module = MagicMock()
_mock_deepsort_module.DeepSort = MagicMock(return_value=MagicMock())
sys.modules.setdefault('deep_sort_realtime', MagicMock())
sys.modules.setdefault('deep_sort_realtime.deepsort_tracker', _mock_deepsort_module)

# insightface (requis par core.face_recognition à l'import)
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

# Forcer le rechargement de face_body_tracker si déjà en cache sans les mocks
if 'core.face_body_tracker' in sys.modules:
    del sys.modules['core.face_body_tracker']

from core.face_body_tracker import FaceBodyTracker  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de construction de faux objets
# ─────────────────────────────────────────────────────────────────────────────

def make_track(track_id: int, ltrb: list[float], confirmed: bool = True) -> MagicMock:
    """Simule un objet Track de DeepSORT (deep_sort_realtime)."""
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = confirmed
    track.to_ltrb.return_value = ltrb
    return track


def make_face(name: str, confidence: float, bbox: list[int]) -> dict:
    """Simule un résultat de FaceRecognizer.detect_and_recognize()."""
    return {'name': name, 'confidence': confidence, 'bbox': bbox}


# ─────────────────────────────────────────────────────────────────────────────
# Fixture : FaceBodyTracker sans GPU
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker_no_gpu():
    """
    FaceBodyTracker avec toutes les dépendances GPU mockées.
    YOLO et DeepSort ont déjà été mockés au niveau sys.modules ci-dessus.
    """
    face_recognizer = MagicMock()
    face_recognizer.known_names = ['Armand']
    t = FaceBodyTracker(face_recognizer)
    yield t


# ─────────────────────────────────────────────────────────────────────────────
# Tests : _find_containing_track
# ─────────────────────────────────────────────────────────────────────────────

class TestFindContainingTrack:

    def test_face_inside_body_upper_half_returns_track_id(self, tracker_no_gpu):
        """Cas nominal : visage bien dans la zone supérieure du corps."""
        # Corps [100,50,300,400] — zone supérieure = y < 50 + 350*0.6 = 260
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        # Visage centré en (200, 150) → dans la zone supérieure ✓
        face_bbox = [170, 100, 230, 200]

        result = tracker_no_gpu._find_containing_track(face_bbox, [track])

        assert result == 1

    def test_face_below_upper_limit_returns_none(self, tracker_no_gpu):
        """Visage trop bas dans le corps (torse) → pas d'association."""
        # Corps [100,50,300,400] — limite supérieure = 260
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        # Centre visage en (200, 350) → sous la limite 260 ✗
        face_bbox = [170, 300, 230, 400]

        result = tracker_no_gpu._find_containing_track(face_bbox, [track])

        assert result is None

    def test_face_outside_body_horizontally_returns_none(self, tracker_no_gpu):
        """Visage hors des limites horizontales du corps."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        # Centre visage en (50, 150) → à gauche du corps ✗
        face_bbox = [20, 100, 80, 200]

        result = tracker_no_gpu._find_containing_track(face_bbox, [track])

        assert result is None

    def test_two_overlapping_bodies_returns_closest(self, tracker_no_gpu):
        """
        Deux corps proches : le visage est dans les deux zones.
        Doit retourner le track dont le centre corps est le plus proche.
        """
        # Corps 1 centré en x=200 — Corps 2 centré en x=500
        track1 = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        track2 = make_track(track_id=2, ltrb=[200, 50, 800, 400])

        # Visage centré en (200, 150) → plus proche du corps 1
        face_bbox = [170, 100, 230, 200]

        result = tracker_no_gpu._find_containing_track(face_bbox, [track1, track2])

        assert result == 1

    def test_no_tracks_returns_none(self, tracker_no_gpu):
        """Aucun track actif → None."""
        face_bbox = [100, 100, 200, 200]
        result = tracker_no_gpu._find_containing_track(face_bbox, [])
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests : _associate_faces_to_tracks
# ─────────────────────────────────────────────────────────────────────────────

class TestAssociateFacesToTracks:

    def test_known_face_updates_identity_map(self, tracker_no_gpu):
        """Un visage reconnu met à jour l'identity_map pour le bon track."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Armand', confidence=0.85, bbox=[170, 100, 230, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=10)

        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'
        assert tracker_no_gpu._identity_map[1]['confidence'] == pytest.approx(0.85)
        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 10

    def test_unknown_face_does_not_override_established_identity(self, tracker_no_gpu):
        """'Inconnu' ne doit JAMAIS écraser une identité déjà établie."""
        tracker_no_gpu._identity_map[1] = {
            'name': 'Armand', 'confidence': 0.85, 'last_face_frame': 5
        }
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Inconnu', confidence=0.0, bbox=[170, 100, 230, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=20)

        # L'identité doit rester inchangée
        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'
        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 5

    def test_unknown_face_on_empty_map_stays_empty(self, tracker_no_gpu):
        """'Inconnu' sur un track sans identité → la map reste vide."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Inconnu', confidence=0.0, bbox=[170, 100, 230, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=1)

        assert 1 not in tracker_no_gpu._identity_map

    def test_face_not_in_any_body_does_not_update_map(self, tracker_no_gpu):
        """Un visage hors de tout corps → la map reste intacte."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        # Visage hors du corps (x=600)
        face = make_face(name='Armand', confidence=0.9, bbox=[560, 100, 640, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=1)

        assert 1 not in tracker_no_gpu._identity_map

    def test_multiple_faces_each_associated_to_correct_track(self, tracker_no_gpu):
        """Deux personnes → chaque visage est associé au bon track."""
        track1 = make_track(track_id=1, ltrb=[50, 50, 250, 400])
        track2 = make_track(track_id=2, ltrb=[300, 50, 500, 400])

        face1 = make_face(name='Armand', confidence=0.9, bbox=[120, 80, 180, 160])
        face2 = make_face(name='Alice',  confidence=0.8, bbox=[360, 80, 440, 160])

        tracker_no_gpu._associate_faces_to_tracks(
            [track1, track2], [face1, face2], frame_count=15
        )

        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'
        assert tracker_no_gpu._identity_map[2]['name'] == 'Alice'


# ─────────────────────────────────────────────────────────────────────────────
# Tests : _face_belongs_to_any_body
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceBelongsToAnyBody:

    def test_face_inside_body_returns_true(self, tracker_no_gpu):
        body_rois = [[100, 50, 300, 400]]
        face_bbox = [170, 100, 230, 200]
        assert tracker_no_gpu._face_belongs_to_any_body(face_bbox, body_rois) is True

    def test_face_outside_all_bodies_returns_false(self, tracker_no_gpu):
        body_rois = [[100, 50, 300, 400]]
        face_bbox = [400, 100, 500, 200]  # À droite de tous les corps
        assert tracker_no_gpu._face_belongs_to_any_body(face_bbox, body_rois) is False

    def test_empty_body_rois_returns_false(self, tracker_no_gpu):
        assert tracker_no_gpu._face_belongs_to_any_body([10, 10, 50, 50], []) is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests : stratégie de mise à jour d'identité (gate de confiance)
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityUpdateStrategy:

    def test_low_confidence_face_does_not_update_empty_map(self, tracker_no_gpu):
        """
        Un visage détecté sous le seuil RECOGNITION_THRESHOLD ne doit pas
        initialiser une identité — protection contre les faux positifs.
        """
        import config
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        # Confiance juste en dessous du seuil
        face = make_face(name='Armand', confidence=config.RECOGNITION_THRESHOLD - 0.01,
                         bbox=[170, 100, 230, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=1)

        assert 1 not in tracker_no_gpu._identity_map

    def test_low_confidence_face_does_not_override_established_identity(self, tracker_no_gpu):
        """
        Une détection faible ne doit pas non plus modifier une identité établie,
        même si c'est un nom différent (protection contre les confusions d'angle).
        """
        import config
        tracker_no_gpu._identity_map[1] = {
            'name': 'Armand', 'confidence': 0.88, 'last_face_frame': 10
        }
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Alice', confidence=config.RECOGNITION_THRESHOLD - 0.01,
                         bbox=[170, 100, 230, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=20)

        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'

    def test_high_confidence_face_replaces_older_identity(self, tracker_no_gpu):
        """
        Une détection valide (>= threshold) met à jour l'identité,
        même si la confiance précédente était plus haute — on préfère le plus récent.
        """
        import config
        tracker_no_gpu._identity_map[1] = {
            'name': 'Armand', 'confidence': 0.92, 'last_face_frame': 5
        }
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Armand', confidence=config.RECOGNITION_THRESHOLD + 0.01,
                         bbox=[170, 100, 230, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=50)

        # last_face_frame doit être mis à jour
        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 50


# ─────────────────────────────────────────────────────────────────────────────
# Tests : logique de fraîcheur du visage (FACE_FRESHNESS_FRAMES)
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceFreshness:
    """
    Vérifie que TrackedPerson.last_face_frame est correctement
    renseigné pour permettre le calcul de fraîcheur dans app.py.
    """

    def test_last_face_frame_updated_on_association(self, tracker_no_gpu):
        """Après association, last_face_frame doit valoir frame_count."""
        import config
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Armand', confidence=config.RECOGNITION_THRESHOLD + 0.1,
                         bbox=[170, 100, 230, 200])

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=42)

        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 42

    def test_last_face_frame_minus_one_when_no_identity(self, tracker_no_gpu):
        """
        Un track sans entrée dans identity_map doit produire un TrackedPerson
        avec last_face_frame = -1 (valeur sentinelle = jamais vu).
        """
        from core.face_body_tracker import TrackedPerson
        track = make_track(track_id=99, ltrb=[0, 0, 100, 200])

        results = tracker_no_gpu._build_results([track])

        assert len(results) == 1
        assert results[0].last_face_frame == -1
        assert results[0].name == 'Inconnu'
