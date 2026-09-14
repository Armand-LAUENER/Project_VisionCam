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
import json
import sys
import threading
import time
import types
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
def auth_disabled(monkeypatch):
    """Accès ouvert par défaut, quoi que contienne le .env de la machine."""
    monkeypatch.setattr(visioncam.config, 'ADMIN_PASSWORD', '')
    monkeypatch.setattr(visioncam.config, 'ADMIN_PASSWORD_HASH', '')
    visioncam._login_failures.clear()


@pytest.fixture(autouse=True)
def isolated_presence_log(monkeypatch, tmp_path):
    """Jamais la vraie data/presence.db : une base temporaire par test."""
    from core.presence_log import PresenceLog

    monkeypatch.setattr(visioncam, "presence_log", PresenceLog(str(tmp_path / "presence-test.db")))


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
# Accès protégé
# ─────────────────────────────────────────────────────────────────────────────

class TestAuth:

    @pytest.fixture
    def password(self, monkeypatch):
        monkeypatch.setattr(visioncam.config, 'ADMIN_PASSWORD', 'secret-visioncam')
        return 'secret-visioncam'

    def test_sans_mot_de_passe_l_acces_reste_ouvert(self, client):
        assert client.get('/status').status_code == 200
        assert client.get('/login').status_code == 302

    def test_page_redirige_vers_la_connexion(self, client, password):
        res = client.get('/?tab=live', headers={'Accept': 'text/html'})

        assert res.status_code == 302
        assert res.headers['Location'].startswith('/login?next=')

    @pytest.mark.parametrize("method, path", [
        ('get', '/status'), ('get', '/video'), ('get', '/api/people'), ('get', '/api/events'),
        ('get', '/api/history.csv'), ('delete', '/api/people/Alice'), ('post', '/api/capture'),
        ('post', '/enroll'), ('post', '/rebuild'),
    ])
    def test_api_repond_401_sans_session(self, client, password, method, path):
        res = getattr(client, method)(path, headers={'Accept': 'text/html'})

        assert res.status_code == 401
        assert res.get_json()['success'] is False

    def test_connexion_puis_deconnexion(self, client, password):
        res = client.post('/login?next=/api/people', data={'password': password})
        assert res.status_code == 302 and res.headers['Location'] == '/api/people'
        assert client.get('/status').status_code == 200

        client.post('/logout')

        assert client.get('/status').status_code == 401

    def test_mauvais_mot_de_passe(self, client, password):
        res = client.post('/login', data={'password': 'faux'})

        assert res.status_code == 401
        assert 'incorrect' in res.data.decode()
        assert client.get('/status').status_code == 401

    def test_hash_prefere_au_mot_de_passe_en_clair(self, client, monkeypatch):
        from werkzeug.security import generate_password_hash

        monkeypatch.setattr(visioncam.config, 'ADMIN_PASSWORD_HASH', generate_password_hash('haché'))
        monkeypatch.setattr(visioncam.config, 'ADMIN_PASSWORD', 'clair')

        assert client.post('/login', data={'password': 'clair'}).status_code == 401
        assert client.post('/login', data={'password': 'haché'}).status_code == 302

    def test_blocage_apres_trop_d_echecs(self, client, password, monkeypatch):
        monkeypatch.setattr(visioncam.config, 'LOGIN_MAX_FAILURES', 3)
        for _ in range(3):
            client.post('/login', data={'password': 'faux'})

        res = client.post('/login', data={'password': password})

        assert res.status_code == 429
        assert client.get('/status').status_code == 401

    @pytest.mark.parametrize("target", ['https://evil.example', '//evil.example', '/\\evil.example'])
    def test_pas_de_redirection_vers_un_autre_site(self, client, password, target):
        res = client.post('/login', query_string={'next': target}, data={'password': password})

        assert res.status_code == 302 and res.headers['Location'] == '/'

    def test_cookie_de_session_protege(self, client, password):
        res = client.post('/login', data={'password': password})

        cookie = res.headers['Set-Cookie']
        assert 'HttpOnly' in cookie and 'SameSite=Lax' in cookie

    def test_cle_de_session_conservee_sur_disque(self, monkeypatch, tmp_path):
        path = tmp_path / 'data' / 'secret_key'
        monkeypatch.setattr(visioncam.config, 'SECRET_KEY', '')
        monkeypatch.setattr(visioncam.config, 'SECRET_KEY_PATH', str(path))
        monkeypatch.setattr(visioncam.app, 'secret_key', 'cle-en-memoire')

        visioncam._persist_secret_key()
        assert path.read_text() == 'cle-en-memoire'
        assert oct(path.stat().st_mode & 0o777) == '0o600'

        monkeypatch.setattr(visioncam.app, 'secret_key', 'autre')
        visioncam._persist_secret_key()
        assert visioncam.app.secret_key == 'cle-en-memoire'


# ─────────────────────────────────────────────────────────────────────────────
# /status — contrat de l'API de présence
# ─────────────────────────────────────────────────────────────────────────────

class TestStatus:

    def test_contrat_json(self, client):
        res = client.get('/status')
        assert res.status_code == 200
        assert set(res.get_json()) == {'currently_present', 'fps', 'total_known',
                                       'tracks', 'frame_size'}

    def test_boites_des_personnes_visibles(self, client):
        with visioncam.state.lock:
            visioncam.state.bench_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            visioncam.state.bench_persons = [_person('3', [10.4, 20, 110, 320], name='Armand')]
            visioncam.state.currently_present = [{'name': 'Armand', 'track_id': '3',
                                                  'confidence': 0.8, 'pose': 'Face',
                                                  'last_face_frame': 1}]

        data = client.get('/status').get_json()

        assert data['frame_size'] == [1280, 720]
        assert data['tracks'] == [{'track_id': '3', 'name': 'Armand',
                                   'bbox': [10, 20, 110, 320], 'pose': 'Face'}]

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
        """Deux connexions longues par onglet (flux vidéo + événements)."""
        assert visioncam.config.SERVER_THREADS >= 16


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
# /api/capture — enrôlement d'une personne désignée dans le flux
# ─────────────────────────────────────────────────────────────────────────────

def _person(track_id, bbox, name='Inconnu'):
    return types.SimpleNamespace(track_id=track_id, body_bbox=bbox, name=name,
                                 pose_kps=[(60.0, 50.0, 0.9)] * 7)


@pytest.fixture
def capture_scene(monkeypatch):
    """Deux personnes visibles ; le recadrage de tête et le visage sont simulés."""
    frame = np.arange(720 * 1280 * 3, dtype=np.uint32).reshape(720, 1280, 3).astype(np.uint8)
    with visioncam.state.lock:
        visioncam.state.bench_frame = frame
        visioncam.state.bench_persons = [_person('1', [0, 0, 200, 600]),
                                         _person('2', [600, 0, 800, 600])]
    head_crop = MagicMock(return_value=(100, 50, 300, 250))
    monkeypatch.setattr(visioncam.FaceBodyTracker, 'head_crop_box', head_crop)
    for method in ('recognize_center_face', 'enroll_person_average',
                   'enroll_person_multitemplate'):
        getattr(_recognizer, method).reset_mock()
    _recognizer.recognize_center_face.return_value = {'name': 'Inconnu', 'confidence': 0.0,
                                                      'bbox': [40, 40, 160, 160]}
    _recognizer.enroll_person_average.return_value = (True, 'ok')
    _recognizer.enroll_person_multitemplate.return_value = (True, 'ok')
    return frame, head_crop


class TestCaptureApi:

    def test_enrole_le_recadrage_de_la_personne_cliquee(self, client, capture_scene):
        frame, head_crop = capture_scene

        res = client.post('/api/capture', json={'name': 'Armand', 'track_id': '2'})

        assert res.status_code == 200
        _, body_bbox, _ = head_crop.call_args[0]
        assert body_bbox == [600, 0, 800, 600]
        name, (crop,) = _recognizer.enroll_person_average.call_args[0]
        assert name == 'Armand'
        np.testing.assert_array_equal(crop, frame[50:250, 100:300])

    def test_mode_guide_enrole_un_template_par_angle(self, client, capture_scene):
        res = client.post('/api/capture', json={'name': 'Armand', 'track_id': '1',
                                                'label': 'ProfilG'})

        assert res.status_code == 200
        name, templates = _recognizer.enroll_person_multitemplate.call_args[0]
        assert name == 'Armand' and list(templates) == ['ProfilG']
        _recognizer.enroll_person_average.assert_not_called()

    @pytest.mark.parametrize("body", [{}, {'name': 'Armand'}, {'track_id': '1'},
                                      {'name': 'Armand', 'track_id': '1', 'label': 'Dos'}])
    def test_champs_invalides(self, client, capture_scene, body):
        assert client.post('/api/capture', json=body).status_code == 400

    def test_personne_plus_visible(self, client, capture_scene):
        assert client.post('/api/capture', json={'name': 'Armand', 'track_id': '9'}).status_code == 404

    def test_aucun_visage(self, client, capture_scene):
        _recognizer.recognize_center_face.return_value = None

        assert client.post('/api/capture', json={'name': 'Armand', 'track_id': '1'}).status_code == 422
        _recognizer.enroll_person_average.assert_not_called()

    def test_visage_trop_petit(self, client, capture_scene, monkeypatch):
        monkeypatch.setattr(visioncam.config, 'RECOGNITION_MIN_FACE_PX', 40)
        _recognizer.recognize_center_face.return_value = {'name': 'Inconnu', 'confidence': 0.0,
                                                          'bbox': [40, 40, 119, 160]}

        res = client.post('/api/capture', json={'name': 'Armand', 'track_id': '1'})

        assert res.status_code == 422 and 'rapprochez' in res.get_json()['message']
        _recognizer.enroll_person_average.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# /api/events et /api/diagnostics — temps réel et diagnostic
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_bus(monkeypatch):
    from core.events import EventBus, StageTimer

    bus = EventBus()
    monkeypatch.setattr(visioncam, 'events', bus)
    monkeypatch.setattr(visioncam, 'timings', StageTimer())
    return bus


def _sse_messages(chunks, count):
    """Les `count` premiers messages `data:` d'un flux SSE, décodés."""
    messages = []
    for chunk in chunks:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        messages += [json.loads(line[len('data: '):]) for line in text.splitlines()
                     if line.startswith('data: ')]
        if len(messages) >= count:
            return messages[:count]
    return messages


class TestEventsStream:

    def test_pousse_l_etat_puis_les_evenements(self, client, fresh_bus, monkeypatch):
        monkeypatch.setattr(visioncam, 'STATUS_PUSH_INTERVAL', 60)
        res = client.get('/api/events', buffered=False)
        chunks = iter(res.response)
        try:
            assert next(chunks).startswith(b'retry:')
            (status,) = _sse_messages(chunks, 1)
            assert status['type'] == 'status' and 'tracks' in status
            assert fresh_bus.subscriber_count == 1

            visioncam._publish('arrival', name='Armand')
            (event,) = _sse_messages(chunks, 1)
            assert event['type'] == 'arrival' and event['name'] == 'Armand' and event['time']
        finally:
            res.close()

        assert fresh_bus.subscriber_count == 0
        assert res.mimetype == 'text/event-stream'

    def test_les_routes_publient_leurs_evenements(self, client, fresh_bus):
        subscription = fresh_bus.subscribe()
        _recognizer.rename_person.return_value = ('ok', 'renommé')
        _recognizer.delete_person.return_value = ('ok', 'supprimé')

        client.post('/api/people/Alice/rename', json={'new_name': 'Alicia'})
        client.delete('/api/people/Bob')

        received = [subscription.get_nowait() for _ in range(2)]
        assert [(e['type'], e['name']) for e in received] == [('renamed', 'Alicia'),
                                                              ('deleted', 'Bob')]
        assert received[0]['old_name'] == 'Alice'


class TestDiagnostics:

    def test_contrat(self, client, fresh_bus):
        visioncam.timings.record('detection', 0.006)

        data = client.get('/api/diagnostics').get_json()

        assert set(data) == {'auth_enabled', 'uptime_s', 'fps', 'timings', 'gpu', 'models', 'recognition',
                             'tracking', 'camera', 'clients'}
        assert data['timings']['detection']['median_ms'] == 6.0
        assert data['recognition']['min_face_px'] == visioncam.config.RECOGNITION_MIN_FACE_PX
        assert data['tracking']['n_init'] == visioncam.config.DEEPSORT_N_INIT
        assert data['clients'] == {'video': 0, 'events': 0}
        assert data['gpu'] is None or {'name', 'memory_used_mb', 'memory_total_mb'} <= set(data['gpu'])


# ─────────────────────────────────────────────────────────────────────────────
# /api/history — historique des présences
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def history(monkeypatch, tmp_path):
    from datetime import datetime as dt

    from core.presence_log import PresenceLog

    log = PresenceLog(str(tmp_path / "presence.db"), gap=60)
    t0 = dt(2026, 9, 14, 9, 0, 0).timestamp()
    for offset, names in [(0, ["Alice"]), (30, ["Alice"]), (200, []),
                          (86400, ["Bob"]), (86460, ["Bob"]), (86600, [])]:
        log.update(names, now=t0 + offset)
    monkeypatch.setattr(visioncam, "presence_log", log)
    return log


class TestHistoryApi:

    def test_liste_et_noms(self, client, history):
        data = client.get('/api/history').get_json()

        assert [s['name'] for s in data['sessions']] == ['Bob', 'Alice']
        assert data['names'] == ['Alice', 'Bob']
        assert data['sessions'][1]['duration_s'] == 30

    def test_filtre_par_personne_et_jour(self, client, history):
        assert [s['name'] for s in
                client.get('/api/history?name=Alice').get_json()['sessions']] == ['Alice']
        one_day = client.get('/api/history?from=2026-09-15&to=2026-09-15').get_json()['sessions']
        assert [s['name'] for s in one_day] == ['Bob']

    def test_filtres_invalides(self, client, history):
        assert client.get('/api/history?from=14/09/2026').status_code == 400
        assert client.get('/api/history?limit=beaucoup').status_code == 400

    def test_export_csv(self, client, history):
        res = client.get('/api/history.csv?name=Bob')

        assert res.status_code == 200 and res.mimetype == 'text/csv'
        assert 'attachment' in res.headers['Content-Disposition']
        lines = res.data.decode().strip().splitlines()
        assert lines[0].startswith('nom,') and len(lines) == 2 and lines[1].startswith('Bob,')

    def test_supprimer_une_personne_efface_son_historique(self, client, history):
        _recognizer.delete_person.return_value = ('ok', 'supprimé')

        client.delete('/api/people/Alice')

        assert [s['name'] for s in history.sessions()] == ['Bob']

    def test_renommer_une_personne_renomme_son_historique(self, client, history):
        _recognizer.rename_person.return_value = ('ok', 'renommé')

        client.post('/api/people/Alice/rename', json={'new_name': 'Alicia'})

        assert history.names() == ['Alicia', 'Bob']


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
