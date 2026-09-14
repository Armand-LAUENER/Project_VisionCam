"""
tests/test_web_routes.py

Teste la page HTML et les routes Flask de VisionCam.

app.py instancie FaceRecognizer, FaceBodyTracker et PoseEstimator au niveau
module : ces trois-là sont donc remplacés dans sys.modules AVANT l'import, ce
qui permet de faire tourner ces tests sans GPU, sans caméra et sans modèle.
Les threads caméra/traitement ne démarrent que sous `__main__`, l'import seul
ne lance rien.

Ce qui est couvert : rendu de la page (éléments et scripts attendus présents),
contrat JSON de /status, et les cas d'erreur des routes d'écriture — c'est là
que les régressions passent inaperçues, un chemin nominal cassé se voyant tout
de suite à l'écran.

Lancer : pytest tests/test_web_routes.py -v
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Mocks des modules lourds, avant l'import de app
# ─────────────────────────────────────────────────────────────────────────────

_recognizer = MagicMock()
_recognizer.known_names = ['Armand_Lauener']
_mock_face_recognition = MagicMock()
_mock_face_recognition.FaceRecognizer = MagicMock(return_value=_recognizer)
sys.modules['core.face_recognition'] = _mock_face_recognition

_mock_tracker_module = MagicMock()
_mock_tracker_module.FaceBodyTracker = MagicMock(return_value=MagicMock())
sys.modules['core.face_body_tracker'] = _mock_tracker_module

_mock_pose_module = MagicMock()
_mock_pose_module.PoseEstimator = MagicMock(return_value=MagicMock())
sys.modules['core.pose_estimation'] = _mock_pose_module

import app as visioncam  # noqa: E402


@pytest.fixture
def client():
    visioncam.app.config['TESTING'] = True
    with visioncam.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    """Repart d'un état vierge : aucune frame, aucune personne suivie."""
    with visioncam.state.lock:
        visioncam.state.bench_frame = None
        visioncam.state.bench_persons = []
        visioncam.state.current_frame = None
        visioncam.state.currently_present = []
        visioncam.state.fps = 0.0
    yield


# ─────────────────────────────────────────────────────────────────────────────
# La page HTML
# ─────────────────────────────────────────────────────────────────────────────

class TestPageHTML:

    def test_la_page_se_rend(self, client):
        res = client.get('/')
        assert res.status_code == 200
        assert res.mimetype == 'text/html'

    def test_le_flux_video_est_reference(self, client):
        """Sans cette source, la page s'affiche mais reste noire."""
        assert b'/video' in client.get('/').data

    def test_les_conteneurs_dynamiques_sont_presents(self, client):
        """Chaque id est manipulé par le JS ; en supprimer un casse la page en silence."""
        html = client.get('/').data
        for element_id in (b'presentList', b'eventLog', b'statusText',
                           b'benchBtn', b'benchResult', b'benchSamples'):
            assert b'id="' + element_id + b'"' in html

    def test_le_panneau_de_mesure_est_cable(self, client):
        html = client.get('/').data
        assert b'runPoseBench()' in html          # le bouton appelle bien la fonction
        assert b'function runPoseBench' in html   # et la fonction est définie

    def test_le_js_est_dans_le_script_principal(self, client):
        """
        Non-régression : runPoseBench doit vivre dans le script du corps de page,
        pas dans le bloc de configuration Tailwind de l'en-tête.
        """
        html = client.get('/').data.decode()
        assert html.index('function runPoseBench') > html.index('cdn.tailwindcss.com')
        assert html.index('function runPoseBench') > html.rindex('<body')


# ─────────────────────────────────────────────────────────────────────────────
# /status — contrat de l'API de présence
# ─────────────────────────────────────────────────────────────────────────────

class TestStatus:

    def test_contrat_json(self, client):
        res = client.get('/status')
        assert res.status_code == 200
        assert set(res.get_json()) == {'currently_present', 'fps', 'total_known'}

    def test_reflete_les_personnes_presentes(self, client):
        with visioncam.state.lock:
            visioncam.state.currently_present = [
                {'name': 'Armand_Lauener', 'track_id': '1', 'confidence': 0.85,
                 'pose': 'Face', 'last_face_frame': 42}
            ]
            visioncam.state.fps = 24.5

        data = client.get('/status').get_json()
        assert data['fps'] == 24.5
        assert data['currently_present'][0]['name'] == 'Armand_Lauener'


# ─────────────────────────────────────────────────────────────────────────────
# Routes d'écriture — cas d'erreur
# ─────────────────────────────────────────────────────────────────────────────

class TestEnroll:

    def test_nom_manquant(self, client):
        res = client.post('/enroll', data={})
        assert res.status_code == 400
        assert res.get_json()['success'] is False

    def test_methode_invalide(self, client):
        res = client.post('/enroll', data={'name': 'Armand', 'method': 'magie'})
        assert res.status_code == 400
        assert 'method' in res.get_json()['message']

    def test_aucune_image(self, client):
        res = client.post('/enroll', data={'name': 'Armand'})
        assert res.status_code == 400


class TestCapture:

    def test_nom_manquant(self, client):
        assert client.post('/capture', data={}).status_code == 400

    def test_aucune_frame_disponible(self, client):
        """Caméra déconnectée : 503, pas un 500."""
        res = client.post('/capture', data={'name': 'Armand'})
        assert res.status_code == 503

    def test_enrole_depuis_la_frame_brute(self, client):
        """
        Non-régression : l'enrôlement partait du JPEG streamé, avec les boîtes
        et les textes dessinés dessus. Il doit partir de la frame brute.
        """
        raw = np.full((48, 64, 3), 7, dtype=np.uint8)
        with visioncam.state.lock:
            visioncam.state.bench_frame = raw
            visioncam.state.bench_persons = [MagicMock()]
            visioncam.state.current_frame = b'jpeg annote'
        _recognizer.enroll_person_average.reset_mock()
        _recognizer.enroll_person_average.return_value = (True, 'ok')

        res = client.post('/capture', data={'name': 'Armand'})

        assert res.status_code == 200
        (name, frames), _ = _recognizer.enroll_person_average.call_args
        assert name == 'Armand'
        assert len(frames) == 1 and np.array_equal(frames[0], raw)

    def test_plusieurs_personnes_refuse(self, client):
        """Deux personnes à l'écran : on ne devine pas laquelle enrôler."""
        with visioncam.state.lock:
            visioncam.state.bench_frame = np.zeros((48, 64, 3), dtype=np.uint8)
            visioncam.state.bench_persons = [MagicMock(), MagicMock()]
        _recognizer.enroll_person_average.reset_mock()

        res = client.post('/capture', data={'name': 'Armand'})

        assert res.status_code == 409
        assert res.get_json()['success'] is False
        _recognizer.enroll_person_average.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# /bench/pose — mesure comparative
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchPose:

    def test_parametre_samples_invalide(self, client):
        res = client.post('/bench/pose', data={'samples': 'beaucoup'})
        assert res.status_code == 400
        assert 'samples' in res.get_json()['message']

    def test_aucune_personne_suivie(self, client):
        """Rien à mesurer : 503 explicite plutôt qu'un rapport vide."""
        res = client.post('/bench/pose', data={'samples': '5'})
        assert res.status_code == 503
        assert res.get_json()['success'] is False

    def test_mesure_concurrente_refusee(self, client):
        """Mediapipe n'est pas réentrant : la seconde mesure doit être rejetée."""
        assert visioncam._bench_lock.acquire(blocking=False)
        try:
            res = client.post('/bench/pose', data={'samples': '5'})
            assert res.status_code == 409
        finally:
            visioncam._bench_lock.release()
