"""
tests/ui/test_browser.py

Teste les pages dans un vrai navigateur (Chromium headless, Playwright).

Les tests de tests/test_web_routes.py vérifient le HTML servi ; seuls ceux-ci
exécutent les modules JavaScript : une erreur JS, un id renommé côté page mais
pas côté script, ou une page qui déborde sur téléphone ne se voient qu'ici.
Même application simulée que test_web_routes (ni GPU, ni caméra, ni modèle),
servie par waitress sur un port libre.

Sans Chromium installé, les tests sont ignorés en local, mais échouent en CI
(variable CI définie) : un navigateur manquant ne doit pas passer pour un succès.

Installer le navigateur : uv run playwright install --only-shell chromium
Lancer : pytest tests/ui -v
"""

import os
import threading
import time

import numpy as np
import pytest

from tests.test_web_routes import _recognizer, visioncam

playwright_api = pytest.importorskip("playwright.sync_api")

PAGES = ['/', '/people', '/history', '/diagnostics']
FRAME_W, FRAME_H = 1280, 720


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            chromium = p.chromium.launch()
        except playwright_api.Error as exc:
            if os.getenv("CI"):
                raise
            pytest.skip(f"Chromium indisponible ({exc.message.splitlines()[0]})")
        yield chromium
        chromium.close()


@pytest.fixture(scope="module")
def server():
    from waitress import create_server

    httpd = create_server(visioncam.app, host="127.0.0.1", port=0,
                          threads=visioncam.config.SERVER_THREADS)
    threading.Thread(target=httpd.run, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.effective_port}"
    httpd.close()


@pytest.fixture(autouse=True)
def scene(monkeypatch, tmp_path):
    """Une personne inconnue à l'écran, une personne en base, un historique."""
    from core.presence_log import PresenceLog

    monkeypatch.setattr(visioncam.config, 'ADMIN_PASSWORD', '')
    monkeypatch.setattr(visioncam.config, 'ADMIN_PASSWORD_HASH', '')

    log = PresenceLog(str(tmp_path / "presence.db"))
    now = time.time()
    log.update(["Alice"], now=now - 600)
    log.update([], now=now - 300)
    monkeypatch.setattr(visioncam, "presence_log", log)

    person = type("Person", (), {"track_id": 7, "name": "Inconnu",
                                 "body_bbox": [500, 100, 800, 700],
                                 "pose_kps": [(650.0, 150.0, 0.9)] * 7})()
    with visioncam.state.lock:
        visioncam.state.bench_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        visioncam.state.bench_persons = [person]
        visioncam.state.currently_present = []
        visioncam.state.fps = 29.0

    monkeypatch.setattr(visioncam.FaceBodyTracker, 'head_crop_box',
                        lambda *_: (560, 100, 740, 280), raising=False)
    _recognizer.reset_mock()
    _recognizer.list_people.return_value = [
        {'name': 'Alice', 'templates': [], 'photos': 3, 'enrolled': True, 'thumbnail': None}]
    _recognizer.rename_person.return_value = ('ok', "'Alice' renommé en 'Alicia'.")
    _recognizer.recognize_center_face.return_value = {'name': 'Inconnu', 'confidence': 0.0,
                                                      'bbox': [20, 20, 160, 170]}
    _recognizer.enroll_person_average.return_value = (True, "'Armand' enrôlé.")
    yield
    with visioncam.state.lock:
        visioncam.state.bench_frame = None
        visioncam.state.bench_persons = []


@pytest.fixture
def page(browser, server):
    """Page qui échoue le test à la moindre erreur JavaScript ou console."""
    context = browser.new_context(base_url=server)
    tab = context.new_page()
    errors = []
    tab.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    tab.on("console", lambda msg: msg.type == "error" and errors.append(f"console: {msg.text}"))
    yield tab
    context.close()
    assert errors == []


def test_every_page_runs_its_script_without_errors(page):
    for path in PAGES:
        page.goto(path, wait_until="domcontentloaded")
        # Le flux d'événements se connecte : les modules ont été exécutés.
        playwright_api.expect(page.locator("#livePill")).to_have_text("En direct")


def test_live_page_draws_status_and_enrolls_the_clicked_person(page):
    page.goto("/", wait_until="domcontentloaded")
    playwright_api.expect(page.locator("#statVisible")).to_have_text("1")
    playwright_api.expect(page.locator("#statFps")).to_have_text("29")

    # Clic au centre de la boîte, converti comme le fait live.js (object-fit: contain).
    box = page.locator("#overlay").bounding_box()
    scale = min(box["width"] / FRAME_W, box["height"] / FRAME_H)
    offset_x = (box["width"] - FRAME_W * scale) / 2
    offset_y = (box["height"] - FRAME_H * scale) / 2
    page.mouse.click(box["x"] + offset_x + 650 * scale, box["y"] + offset_y + 400 * scale)

    dialog = page.locator("#captureDialog")
    playwright_api.expect(dialog).to_be_visible()
    playwright_api.expect(page.locator("#captureTarget")).to_contain_text("#7")
    page.fill("#captureName", "Armand")
    page.click("#captureSubmit")

    playwright_api.expect(page.locator(".toast.ok")).to_contain_text("enrôlé")
    playwright_api.expect(dialog).to_be_hidden()
    name, _ = _recognizer.enroll_person_average.call_args[0]
    assert name == "Armand"


def test_people_page_lists_and_renames(page):
    page.goto("/people", wait_until="domcontentloaded")
    card = page.locator(".card", has_text="Alice")
    playwright_api.expect(card).to_contain_text("3 photo(s)")

    card.get_by_role("button", name="Renommer").click()
    page.fill("#renameInput", "Alicia")
    page.locator("#renameForm").get_by_role("button", name="Renommer").click()

    playwright_api.expect(page.locator("#renameDialog")).to_be_hidden()
    _recognizer.rename_person.assert_called_once_with("Alice", "Alicia")


def test_history_page_shows_sessions(page):
    page.goto("/history", wait_until="domcontentloaded")

    playwright_api.expect(page.locator("#sessions tr")).to_have_count(1)
    playwright_api.expect(page.locator("#sessions")).to_contain_text("Alice")
    playwright_api.expect(page.locator("#historySummary")).to_contain_text("1 session(s)")


def test_diagnostics_page_fills_its_panels(page):
    page.goto("/diagnostics", wait_until="domcontentloaded")

    playwright_api.expect(page.locator("#hardware dt").first).to_be_visible()
    playwright_api.expect(page.locator("#settings dt").first).to_be_visible()


@pytest.mark.parametrize("path", PAGES)
def test_pages_fit_a_phone_screen(browser, server, path):
    context = browser.new_context(base_url=server, viewport={"width": 390, "height": 844})
    try:
        tab = context.new_page()
        tab.goto(path, wait_until="domcontentloaded")
        playwright_api.expect(tab.locator("#livePill")).to_have_text("En direct")
        overflow = tab.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 0, f"{path} déborde de {overflow}px en largeur"
    finally:
        context.close()
