"""
capture.py — Lecture d'une caméra ou d'une vidéo, dans son propre thread.

Sans dépendance à l'application : utilisé par les threads de app.py comme par
les processus de caméras (core/camera_worker.py). `camera` est tout objet qui
expose `id`, `source` et `is_file` ; `running()` dit s'il faut continuer, et
`tuning.max_fps`, s'il est donné, est relu à chaque image (core/adaptive.py) ;
`counters` reçoit les images jetées et sautées (`frames_dropped`,
`source_skipped`) pour le journal d'endurance, et `record(étape, secondes)`
le coût de lecture de chaque image : « decode » pour une vidéo (décodage seul,
hors attente de l'heure de l'image), « read » pour une caméra (attente de
l'image comprise : OpenCV ne sépare pas les deux).
"""

from __future__ import annotations

import logging
import queue
import time

import cv2

import config
from core.video_source import FileSource

logger = logging.getLogger(__name__)


def open_source(camera):
    """
    Ouvre la caméra avec un timeout court pour ne pas bloquer le thread.

    Une source fichier (VIDEO_FILE, ou un chemin dans CAMERAS) remplace la caméra,
    avec la même interface.
    """
    if camera.is_file:
        source = FileSource(camera.source, mode=config.VIDEO_MODE, loop=config.VIDEO_LOOP)
        if not source.isOpened():
            logger.error("Vidéo illisible : %s", camera.source)
            return None
        logger.info("[%s] Vidéo à la place de la caméra : %s (%s, %.0f i/s%s)", camera.id,
                    camera.source, config.VIDEO_MODE, source.fps,
                    ", en boucle" if config.VIDEO_LOOP else "")
        return source

    cap = cv2.VideoCapture(camera.source)

    # Timeout de connexion et de lecture (ms). Sans ça, VideoCapture sur HTTP
    # peut bloquer jusqu'à 30s sans aucun log — invisible pour l'utilisateur.
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.CAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.CAM_READ_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)

    if not cap.isOpened():
        cap.release()
        return None
    logger.info("[%s] Caméra ouverte : %dx%d", camera.id, int(cap.get(3)), int(cap.get(4)))
    return cap


def reconnect(cap, camera, running):
    """
    Tente de rouvrir la caméra avec exponential backoff (1s → 30s max).
    Bloque jusqu'à la reconnexion ou l'arrêt de l'application.
    Retourne un nouveau VideoCapture ouvert, ou None si running() devient faux.
    """
    if cap is not None:
        cap.release()

    delay = 1.0
    attempt = 0
    while running():
        attempt += 1
        logger.warning("Reconnexion tentative #%d dans %.0fs...", attempt, delay)
        time.sleep(delay)

        new_cap = open_source(camera)
        if new_cap is not None:
            logger.info("Reconnexion réussie après %d tentative(s)", attempt)
            return new_cap

        delay = min(delay * 2, 30.0)

    return None


class Throttle:
    """Laisse passer au plus `max_fps` images par seconde, en moyenne exacte.

    Une grille de temps (prochaine échéance += 1/max_fps) plutôt qu'un délai
    depuis la dernière image : avec une source à 30 i/s et 10 i/s visés, un
    délai de 100 ms ne laisserait passer qu'une image sur 4 (7,5 i/s).
    """

    def __init__(self, max_fps: float, clock=time.monotonic):
        self._clock = clock
        self._next = None
        self.max_fps = None
        self.set_rate(max_fps)

    def set_rate(self, max_fps: float) -> None:
        """Change la cadence en marche (adaptation au nombre de caméras)."""
        if max_fps == self.max_fps:
            return
        self.max_fps = max_fps
        self.interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._next = None

    def allow(self) -> bool:
        if not self.interval:
            return True
        now = self._clock()
        if self._next is None or self._next < now - self.interval:
            self._next = now          # premier passage, ou retard : on repart d'ici
        if now < self._next:
            return False
        self._next += self.interval
        return True


def capture_loop(camera, frame_queue, running, counters, record=lambda stage, seconds: None,
                 tuning=None):
    """
    Lit les frames depuis la caméra et les pousse dans frame_queue, avec
    leur heure de réception (perf_counter) pour mesurer la latence.
    Entièrement découplé du pipeline AI : cap.read() ne bloque jamais
    le calcul YOLO/InsightFace. La queue conserve toujours la frame
    la plus récente (drop oldest si pleine).
    """
    logger.info("[%s] Thread caméra démarré.", camera.id)
    cap = open_source(camera)
    if cap is None and camera.is_file:
        return   # un fichier illisible ne se répare pas en réessayant
    if cap is None:
        logger.error("[%s] Impossible d'ouvrir la caméra au démarrage : %s",
                     camera.id, camera.source)
        cap = reconnect(None, camera, running)
        if cap is None:
            return

    consecutive_failures = 0
    every_frame = isinstance(cap, FileSource) and cap.mode == "every_frame"
    throttle = Throttle(0 if every_frame else config.CAMERA_MAX_FPS)

    while running():
        read_start = time.perf_counter()
        ret, frame = cap.read()
        if ret:
            if isinstance(cap, FileSource):
                record('decode', cap.last_decode_s)
            else:
                record('read', time.perf_counter() - read_start)
        if not ret and isinstance(cap, FileSource):
            logger.info("[%s] Fin de la vidéo : %s", camera.id, camera.source)
            break
        if not ret:
            consecutive_failures += 1
            if consecutive_failures == 1:
                logger.warning("Frame perdue...")
            elif consecutive_failures >= config.CAM_MAX_FAILURES:
                logger.error("%d échecs consécutifs — déconnexion détectée.", consecutive_failures)
                cap = reconnect(cap, camera, running)
                if cap is None:
                    break
                consecutive_failures = 0
            else:
                time.sleep(0.05)
            continue

        consecutive_failures = 0
        if tuning is not None and not every_frame:
            throttle.set_rate(tuning.max_fps)
        if not throttle.allow():
            continue
        item = (time.perf_counter(), frame)
        if isinstance(cap, FileSource):
            counters.source_skipped = cap.skipped

        # every_frame : aucune image perdue, la lecture attend le pipeline.
        if every_frame:
            put_waiting(frame_queue, item, running)
            continue

        # Toujours garder la frame la plus fraîche : drop oldest si queue pleine.
        try:
            frame_queue.put_nowait(item)
        except queue.Full:
            counters.frames_dropped += 1
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(item)
            except queue.Full:
                pass

    cap.release()
    logger.info("[%s] Thread caméra arrêté.", camera.id)


def put_waiting(frame_queue, item, running) -> None:
    """Met l'image en file en attendant qu'une place se libère (mode every_frame)."""
    while running():
        try:
            frame_queue.put(item, timeout=0.5)
            return
        except queue.Full:
            continue
