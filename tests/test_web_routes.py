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

import concurrent.futures
import io
import sys
import threading
import time
import urllib.request
from unittest.mock import MagicMock

import cv2
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
        visioncam.state.stream_clients = 0
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
# /video — flux MJPEG
# ─────────────────────────────────────────────────────────────────────────────

def _next_chunk(gen, timeout):
    """next(gen) dans un thread : None si rien n'arrive avant `timeout`."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, gen)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # Débloque le générateur pour que le thread se termine.
            visioncam._publish_display_frame(np.zeros((8, 8, 3), dtype=np.uint8))
            future.result(timeout=2)
            return None


class TestVideoStream:

    def test_pas_d_encodage_sans_client(self):
        """Personne ne regarde : la frame n'est pas encodée."""
        with visioncam.state.lock:
            visioncam.state.stream_clients = 0
        visioncam._publish_display_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        assert visioncam.state.current_frame is None

    def test_n_envoie_que_les_nouvelles_frames(self, monkeypatch):
        """
        Non-régression : le générateur renvoyait la même frame à MJPEG_FPS_LIMIT
        tant que le pipeline n'en publiait pas de nouvelle.
        """
        monkeypatch.setattr(visioncam.config, 'MJPEG_FPS_LIMIT', 1000)
        gen = visioncam.generate_frames()
        try:
            first = _next_chunk_after_publish(gen)
            assert first.startswith(b'--frame')
            assert visioncam.state.stream_clients == 1

            assert _next_chunk(gen, timeout=0.3) is None
        finally:
            gen.close()

        assert visioncam.state.stream_clients == 0


def _next_chunk_after_publish(gen):
    """Le générateur s'enregistre au premier next() : publier juste après."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, gen)
        deadline = time.monotonic() + 2
        while visioncam.state.stream_clients == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        visioncam._publish_display_frame(np.full((8, 8, 3), 200, dtype=np.uint8))
        return future.result(timeout=2)


# ─────────────────────────────────────────────────────────────────────────────
# Serveur de production (waitress)
# ─────────────────────────────────────────────────────────────────────────────

class TestWaitressServer:

    def test_status_answers_while_streams_are_open(self):
        """
        Chaque flux /video garde un thread du serveur. Avec les 4 threads par
        défaut de waitress, quatre onglets bloquaient /status : le nombre de
        threads configuré doit laisser répondre les autres routes.
        """
        from waitress import create_server

        server = create_server(visioncam.app, host="127.0.0.1", port=0,
                               threads=visioncam.config.SERVER_THREADS)
        threading.Thread(target=server.run, daemon=True).start()
        base = f"http://127.0.0.1:{server.effective_port}"

        # Un flux ne remarque la fermeture de son client qu'en lui envoyant
        # une image : on publie jusqu'à ce que tous les flux soient refermés.
        publishing = threading.Event()
        publishing.set()

        def publish():
            while publishing.is_set():
                visioncam._publish_display_frame(np.full((48, 64, 3), 128, dtype=np.uint8))
                time.sleep(0.02)

        threading.Thread(target=publish, daemon=True).start()
        streams = []
        try:
            for _ in range(4):
                streams.append(urllib.request.urlopen(f"{base}/video", timeout=5))
            for stream in streams:
                assert stream.read(64).startswith(b"--frame")

            with urllib.request.urlopen(f"{base}/status", timeout=5) as res:
                assert res.status == 200
        finally:
            for stream in streams:
                stream.close()
            deadline = time.monotonic() + 10
            while visioncam.state.stream_clients > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            publishing.clear()
            server.close()

        assert visioncam.state.stream_clients == 0

    def test_enough_threads_for_several_viewers(self):
        assert visioncam.config.SERVER_THREADS >= 8


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


class TestRebuild:

    def test_lance_la_reconstruction(self, client):
        _recognizer.rebuild_database.reset_mock()

        res = client.post('/rebuild')

        assert res.status_code == 202
        # Le verrou est rendu à la fin du thread de reconstruction.
        assert visioncam._rebuild_lock.acquire(timeout=2)
        visioncam._rebuild_lock.release()
        _recognizer.rebuild_database.assert_called_once()

    def test_reconstruction_concurrente_refusee(self, client):
        """Deux rebuilds en parallèle se marcheraient dessus : le second est rejeté."""
        _recognizer.rebuild_database.reset_mock()
        assert visioncam._rebuild_lock.acquire(blocking=False)
        try:
            res = client.post('/rebuild')
            assert res.status_code == 409
            assert res.get_json()['started'] is False
        finally:
            visioncam._rebuild_lock.release()
        _recognizer.rebuild_database.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# /api/people — gestion des personnes
# ─────────────────────────────────────────────────────────────────────────────

class TestPeopleApi:

    def test_liste_sans_chemin_disque(self, client, tmp_path):
        thumb = tmp_path / "a.jpg"
        cv2.imwrite(str(thumb), np.full((400, 300, 3), 90, dtype=np.uint8))
        _recognizer.list_people.return_value = [
            {'name': 'Alice', 'templates': [], 'photos': 3, 'enrolled': True, 'thumbnail': str(thumb)},
            {'name': 'Bob', 'templates': ['Face'], 'photos': 0, 'enrolled': False, 'thumbnail': None},
        ]

        people = client.get('/api/people').get_json()['people']

        assert [p['name'] for p in people] == ['Alice', 'Bob']
        assert people[0]['thumbnail_url'] == '/api/people/Alice/thumbnail'
        assert people[1]['thumbnail_url'] is None
        assert all('thumbnail' not in p for p in people)

        res = client.get('/api/people/Alice/thumbnail')
        assert res.status_code == 200 and res.mimetype == 'image/jpeg'
        image = cv2.imdecode(np.frombuffer(res.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert max(image.shape[:2]) == visioncam.THUMBNAIL_SIZE
        assert client.get('/api/people/Bob/thumbnail').status_code == 404

    @pytest.mark.parametrize("status, code", [('ok', 200), ('invalid', 400),
                                              ('not_found', 404), ('conflict', 409)])
    def test_renommer_traduit_le_statut(self, client, status, code):
        _recognizer.rename_person.reset_mock()
        _recognizer.rename_person.return_value = (status, 'msg')

        res = client.post('/api/people/Alice/rename', json={'new_name': ' Alicia '})

        assert res.status_code == code
        _recognizer.rename_person.assert_called_once_with('Alice', 'Alicia')

    def test_supprimer(self, client):
        _recognizer.delete_person.return_value = ('not_found', "'Zoé' n'existe pas.")
        assert client.delete('/api/people/Zoé').status_code == 404

        _recognizer.delete_person.return_value = ('ok', 'supprimé')
        assert client.delete('/api/people/Alice').status_code == 200

    def test_ajouter_des_photos(self, client):
        _, jpeg = cv2.imencode('.jpg', np.full((50, 50, 3), 120, dtype=np.uint8))
        _recognizer.enroll_person_average.reset_mock()
        _recognizer.enroll_person_average.return_value = (True, 'ok')

        res = client.post('/api/people/Alice/photos',
                          data={'images': [(io.BytesIO(jpeg.tobytes()), 'a.jpg')]},
                          content_type='multipart/form-data')

        assert res.status_code == 200
        name, images = _recognizer.enroll_person_average.call_args[0]
        assert name == 'Alice' and len(images) == 1

    def test_ajouter_sans_image_lisible(self, client):
        res = client.post('/api/people/Alice/photos',
                          data={'images': [(io.BytesIO(b'pas une image'), 'a.jpg')]},
                          content_type='multipart/form-data')
        assert res.status_code == 400


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
