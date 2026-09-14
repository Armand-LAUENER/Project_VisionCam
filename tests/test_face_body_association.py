"""
tests/test_face_body_association.py

Teste la logique d'association Visage ↔ Corps de FaceBodyTracker (pipeline YOLOv8-Pose).

Architecture testée :
  - _find_containing_track  : fallback nose-proximity (track.others['nose'])
  - _associate_faces_to_tracks : chemin nominal via source_track_id + vote buffer
  - _build_results            : TrackedPerson.last_face_frame

Aucun modèle GPU n'est chargé : ultralytics, deep_sort_realtime et FaceRecognizer
sont mockés dans sys.modules AVANT l'import du module.

Lancer : pytest tests/test_face_body_association.py -v
"""

import sys
from unittest.mock import MagicMock

import pytest

import config

# ─────────────────────────────────────────────────────────────────────────────
# Injection des dépendances lourdes dans sys.modules avant tout import
# ─────────────────────────────────────────────────────────────────────────────

_mock_ultralytics = MagicMock()
_mock_ultralytics.YOLO = MagicMock(return_value=MagicMock())
sys.modules.setdefault('ultralytics', _mock_ultralytics)

_mock_deepsort_module = MagicMock()
_mock_deepsort_module.DeepSort = MagicMock(return_value=MagicMock())
sys.modules.setdefault('deep_sort_realtime', MagicMock())
sys.modules.setdefault('deep_sort_realtime.deepsort_tracker', _mock_deepsort_module)

sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

if 'core.face_body_tracker' in sys.modules:
    del sys.modules['core.face_body_tracker']

from core.face_body_tracker import FaceBodyTracker  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_track(track_id: int, ltrb: list[float],
               confirmed: bool = True) -> MagicMock:
    """
    Simule un Track DeepSORT.
    Le nez est injecté via tracker._nose_map, pas via track.others
    (track.others n'est pas fiable dans DeepSORT 1.3.x).
    """
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = confirmed
    track.to_ltrb.return_value = ltrb
    return track


def make_face(name: str, confidence: float, bbox: list[int],
              source_track_id: int | None = None) -> dict:
    """
    Simule un résultat de FaceRecognizer.recognize_center_face()
    enrichi par _recognize_faces_for_tracks (source_track_id obligatoire
    pour le chemin nominal d'association directe).
    """
    face = {'name': name, 'confidence': confidence, 'bbox': bbox}
    if source_track_id is not None:
        face['source_track_id'] = source_track_id
    return face


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker_no_gpu():
    face_recognizer = MagicMock()
    face_recognizer.known_names = ['Armand']
    t = FaceBodyTracker(face_recognizer)
    yield t


class TestYoloModelLoading:

    def test_missing_engine_explains_how_to_build_it(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'YOLO_MODEL', str(tmp_path / 'yolov8s-pose.engine'))

        with pytest.raises(FileNotFoundError, match='yolo export'):
            FaceBodyTracker(MagicMock())

    def test_existing_engine_is_loaded(self, monkeypatch, tmp_path):
        engine = tmp_path / 'yolov8s-pose.engine'
        engine.write_bytes(b'')
        monkeypatch.setattr(config, 'YOLO_MODEL', str(engine))

        FaceBodyTracker(MagicMock())

    def test_missing_pt_is_left_to_ultralytics(self, monkeypatch):
        """Un .pt absent est auto-téléchargé par ultralytics : pas d'erreur ici."""
        monkeypatch.setattr(config, 'YOLO_MODEL', 'absent-pose.pt')

        FaceBodyTracker(MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# Tests : _find_containing_track  (fallback nose-proximity)
# ─────────────────────────────────────────────────────────────────────────────

class TestFindContainingTrack:

    def test_face_near_nose_returns_track_id(self, tracker_no_gpu):
        """Cas nominal : face_center proche du nez → track retourné."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        tracker_no_gpu._nose_map[1] = (200, 120, 0.9)
        face_bbox = [170, 120, 230, 180]  # centre (200, 150) → dist=30 < 160 ✓

        result = tracker_no_gpu._find_containing_track(face_bbox, [track])

        assert result == 1

    def test_face_too_far_from_nose_returns_none(self, tracker_no_gpu):
        """Face center trop loin du nez (> body_width × 0.8) → None."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        tracker_no_gpu._nose_map[1] = (200, 120, 0.9)
        face_bbox = [470, 120, 530, 180]  # centre (500,150) → dist≈301 > 160 ✗

        result = tracker_no_gpu._find_containing_track(face_bbox, [track])

        assert result is None

    def test_track_without_nose_is_skipped(self, tracker_no_gpu):
        """Track absent du nose_map → skip, retourne None."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        # Pas d'entrée dans _nose_map
        face_bbox = [170, 100, 230, 180]

        result = tracker_no_gpu._find_containing_track(face_bbox, [track])

        assert result is None

    def test_track_with_low_nose_conf_is_skipped(self, tracker_no_gpu):
        """Nez avec confiance < POSE_NOSE_CONF_THRESHOLD → skip."""
        import config
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        tracker_no_gpu._nose_map[1] = (200, 120, config.POSE_NOSE_CONF_THRESHOLD - 0.01)
        face_bbox = [170, 100, 230, 180]

        result = tracker_no_gpu._find_containing_track(face_bbox, [track])

        assert result is None

    def test_two_tracks_returns_closest_nose(self, tracker_no_gpu):
        """Deux tracks : retourne celui dont le nez est le plus proche."""
        track1 = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        track2 = make_track(track_id=2, ltrb=[300, 50, 500, 400])
        tracker_no_gpu._nose_map[1] = (205, 140, 0.9)   # dist≈11 ✓
        tracker_no_gpu._nose_map[2] = (400, 140, 0.9)   # dist≈201, max=160 ✗
        face_bbox = [170, 120, 230, 180]  # centre (200, 150)

        result = tracker_no_gpu._find_containing_track(face_bbox, [track1, track2])

        assert result == 1

    def test_no_tracks_returns_none(self, tracker_no_gpu):
        """Aucun track actif → None."""
        result = tracker_no_gpu._find_containing_track([100, 100, 200, 200], [])
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests : _associate_faces_to_tracks
# ─────────────────────────────────────────────────────────────────────────────

class TestAssociateFacesToTracks:

    def test_known_face_updates_identity_map(self, tracker_no_gpu):
        """Un visage avec source_track_id confirme l'identité après consensus."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Armand', confidence=0.85, bbox=[170, 100, 230, 200],
                         source_track_id=1)

        required = (FaceBodyTracker.VOTE_WINDOW + 1) // 2
        for i in range(required):
            tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=10 + i)

        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'
        assert abs(tracker_no_gpu._identity_map[1]['confidence'] - 0.85) < 1e-6
        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 10 + required - 1

    def test_unknown_face_does_not_override_established_identity(self, tracker_no_gpu):
        """'Inconnu' ne doit JAMAIS écraser une identité déjà établie."""
        tracker_no_gpu._identity_map[1] = {
            'name': 'Armand', 'confidence': 0.85, 'last_face_frame': 5
        }
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Inconnu', confidence=0.0, bbox=[170, 100, 230, 200],
                         source_track_id=1)

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=20)

        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'
        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 5

    def test_unknown_face_on_empty_map_stays_empty(self, tracker_no_gpu):
        """'Inconnu' sur un track sans identité → la map reste vide."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Inconnu', confidence=0.0, bbox=[170, 100, 230, 200],
                         source_track_id=1)

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=1)

        assert 1 not in tracker_no_gpu._identity_map

    def test_face_without_source_track_id_uses_fallback(self, tracker_no_gpu):
        """Sans source_track_id, le fallback géométrique (nose-proximity) est utilisé."""
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        tracker_no_gpu._nose_map[1] = (200, 120, 0.9)
        face = make_face(name='Armand', confidence=0.9, bbox=[170, 100, 230, 180])

        required = (FaceBodyTracker.VOTE_WINDOW + 1) // 2
        for i in range(required):
            tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=5 + i)

        assert tracker_no_gpu._identity_map.get(1, {}).get('name') == 'Armand'

    def test_multiple_faces_each_associated_to_correct_track(self, tracker_no_gpu):
        """Deux personnes → chaque visage est associé au bon track après consensus."""
        track1 = make_track(track_id=1, ltrb=[50, 50, 250, 400])
        track2 = make_track(track_id=2, ltrb=[300, 50, 500, 400])
        tracker_no_gpu._nose_map[1] = (150, 100, 0.9)
        tracker_no_gpu._nose_map[2] = (400, 100, 0.9)

        face1 = make_face(name='Armand', confidence=0.9, bbox=[120, 80, 180, 160],
                          source_track_id=1)
        face2 = make_face(name='Alice', confidence=0.8, bbox=[370, 80, 430, 160],
                          source_track_id=2)

        required = (FaceBodyTracker.VOTE_WINDOW + 1) // 2
        for i in range(required):
            tracker_no_gpu._associate_faces_to_tracks(
                [track1, track2], [face1, face2], frame_count=15 + i
            )

        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'
        assert tracker_no_gpu._identity_map[2]['name'] == 'Alice'


# ─────────────────────────────────────────────────────────────────────────────
# Tests : stratégie de mise à jour d'identité (gate de confiance)
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityUpdateStrategy:

    def test_low_confidence_face_does_not_update_empty_map(self, tracker_no_gpu):
        """Visage sous RECOGNITION_THRESHOLD → pas d'identité initialisée."""
        import config
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Armand',
                         confidence=config.RECOGNITION_THRESHOLD - 0.01,
                         bbox=[170, 100, 230, 200],
                         source_track_id=1)

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=1)

        assert 1 not in tracker_no_gpu._identity_map

    def test_low_confidence_face_does_not_override_established_identity(self, tracker_no_gpu):
        """Détection faible ≠ écrasement d'une identité établie (même autre nom)."""
        import config
        tracker_no_gpu._identity_map[1] = {
            'name': 'Armand', 'confidence': 0.88, 'last_face_frame': 10
        }
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Alice',
                         confidence=config.RECOGNITION_THRESHOLD - 0.01,
                         bbox=[170, 100, 230, 200],
                         source_track_id=1)

        tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=20)

        assert tracker_no_gpu._identity_map[1]['name'] == 'Armand'

    def test_high_confidence_face_updates_last_face_frame_after_consensus(self, tracker_no_gpu):
        """Après consensus, last_face_frame est mis à jour."""
        import config
        tracker_no_gpu._identity_map[1] = {
            'name': 'Armand', 'confidence': 0.92, 'last_face_frame': 5
        }
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Armand',
                         confidence=config.RECOGNITION_THRESHOLD + 0.01,
                         bbox=[170, 100, 230, 200],
                         source_track_id=1)

        required = (FaceBodyTracker.VOTE_WINDOW + 1) // 2
        for i in range(required):
            tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=50 + i)

        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 50 + required - 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests : logique de fraîcheur du visage (FACE_FRESHNESS_FRAMES)
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceFreshness:

    def test_last_face_frame_updated_on_association(self, tracker_no_gpu):
        """Après consensus, last_face_frame vaut le dernier frame_count utilisé."""
        import config
        track = make_track(track_id=1, ltrb=[100, 50, 300, 400])
        face = make_face(name='Armand',
                         confidence=config.RECOGNITION_THRESHOLD + 0.1,
                         bbox=[170, 100, 230, 200],
                         source_track_id=1)

        required = (FaceBodyTracker.VOTE_WINDOW + 1) // 2
        for i in range(required):
            tracker_no_gpu._associate_faces_to_tracks([track], [face], frame_count=42 + i)

        assert tracker_no_gpu._identity_map[1]['last_face_frame'] == 42 + required - 1

    def test_last_face_frame_minus_one_when_no_identity(self, tracker_no_gpu):
        """Track sans identité → TrackedPerson.last_face_frame == -1 (sentinelle)."""
        track = make_track(track_id=99, ltrb=[0, 0, 100, 200])

        results = tracker_no_gpu._build_results([track])

        assert len(results) == 1
        assert results[0].last_face_frame == -1
        assert results[0].name == 'Inconnu'


# ─────────────────────────────────────────────────────────────────────────────
# Tests : dimensionnement du crop tête envoyé à InsightFace
# ─────────────────────────────────────────────────────────────────────────────

class TestCropHalfSize:
    """SCRFD ne détecte pas un visage qui remplit son image d'entrée : le crop
    doit grandir avec le VISAGE.

    Régression : la taille était dérivée de la hauteur du corps, qui de près
    n'est plus qu'une tête-épaules et sous-estime le visage. Un plan rapproché
    tombait sur le plancher de 125 px — crop 250×250 pour un visage de 246 px —
    et InsightFace n'y trouvait rien, laissant la personne « Inconnu » quelle
    que soit la qualité de l'enrôlement.

    Les valeurs de référence viennent de 70 mesures sur le flux live (visages
    de 51 à 246 px) : `min_half / kp_span` vaut 1.16 en médiane, 1.48 au pire.
    """

    # Écart entre keypoints et demi-taille minimale effectivement mesurée,
    # aux deux extrémités de la plage couverte.
    CLOSE_UP_SPAN, CLOSE_UP_MEASURED_MIN = 184.5, 210
    DISTANT_SPAN, DISTANT_MEASURED_MIN = 41.0, 60

    @staticmethod
    def kps_spanning(distance: float) -> list[tuple[float, float]]:
        """Deux keypoints faciaux séparés de `distance` pixels."""
        return [(600.0, 400.0), (600.0 + distance, 400.0)]

    def test_close_up_covers_the_measured_minimum(self, tracker_no_gpu):
        half = tracker_no_gpu._crop_half_size(
            self.kps_spanning(self.CLOSE_UP_SPAN), body_height=451)

        assert half >= self.CLOSE_UP_MEASURED_MIN
        # Et le corps ne doit plus décider : à 451 px il donnait 125 (plancher),
        # soit la moitié de ce qu'il faut.
        assert half > config.POSE_CROP_HALF_SIZE

    def test_distant_person_falls_back_on_the_floor(self, tracker_no_gpu):
        """Le plancher doit reprendre la main quand le visage est minuscule —
        l'abaisser au ratio seul donnerait un crop plus petit que le minimum
        mesuré."""
        half = tracker_no_gpu._crop_half_size(
            self.kps_spanning(self.DISTANT_SPAN), body_height=126)

        assert half == config.POSE_CROP_HALF_SIZE
        assert half >= self.DISTANT_MEASURED_MIN

    def test_crop_grows_with_the_face_not_the_body(self, tracker_no_gpu):
        """Le cœur du correctif : à corps identique, un visage plus grand doit
        donner un crop plus grand. L'ancienne règle rendait la même valeur."""
        small = tracker_no_gpu._crop_half_size(self.kps_spanning(90), body_height=450)
        large = tracker_no_gpu._crop_half_size(self.kps_spanning(185), body_height=450)

        assert large > small

    def test_single_keypoint_falls_back_on_body_height(self, tracker_no_gpu):
        """De profil marqué ou de dos, il n'y a pas d'écart à mesurer."""
        half = tracker_no_gpu._crop_half_size([(600.0, 400.0)], body_height=500)

        assert half == int(500 * config.POSE_CROP_BODY_RATIO)

    def test_no_keypoint_falls_back_on_body_height(self, tracker_no_gpu):
        half = tracker_no_gpu._crop_half_size([], body_height=500)

        assert half == int(500 * config.POSE_CROP_BODY_RATIO)


class TestHeadCropGeometry:
    """Le crop réellement découpé dans la frame suit bien cette demi-taille."""

    @staticmethod
    def _captured_crop(tracker, track, frame_size=(720, 1280)):
        import numpy as np

        crops = []
        tracker.face_recognizer.recognize_center_face = lambda crop: crops.append(crop)
        frame = np.zeros((*frame_size, 3), dtype=np.uint8)
        tracker._recognize_faces_for_tracks(frame, [track])
        return crops[0] if crops else None

    def test_crop_is_sized_from_the_facial_keypoints(self, tracker_no_gpu):
        track = make_track(track_id=1, ltrb=[366, 210, 899, 709])
        # Cinq keypoints étalés sur 180 px, centrés loin des bords.
        tracker_no_gpu._face_kps_map[1] = [
            (560.0, 360.0, 0.99), (740.0, 360.0, 0.99), (650.0, 380.0, 0.99),
            (600.0, 400.0, 0.99), (700.0, 400.0, 0.99),
        ]

        crop = self._captured_crop(tracker_no_gpu, track)

        expected = 2 * tracker_no_gpu._crop_half_size(
            [(x, y) for x, y, _ in tracker_no_gpu._face_kps_map[1]], 499)
        assert crop.shape[:2] == (expected, expected)
