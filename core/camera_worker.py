"""
camera_worker.py — Processus de caméras (roadmap 2.2, CAMERA_WORKERS=process).

Les threads d'un même processus partagent un GIL : au-delà de deux caméras,
le traitement plafonne vers 72 i/s au total alors que le GPU suit
(docs/performance.md). Chaque processus de caméras a son GIL : il lit ses
caméras, fait tourner leur pipeline (core/frame_pipeline.py) et renvoie à
l'application un FrameResult par image, sans l'image brute : le JPEG du flux
seulement si une page le regarde.

Lancé par app.py avec `python -m core.camera_worker` plutôt que par
multiprocessing : en `spawn`, multiprocessing réexécute le script principal
dans l'enfant, et app.py charge à l'import Flask, la reconnaissance, un
tracker par caméra et l'écriture SQLite.

Deux connexions vers l'application (multiprocessing.connection, socket Unix
authentifiée) :
  - résultats : worker → application, un FrameResult par image ;
  - commandes : application → worker :
      ("stream", caméra, bool)  le flux de la caméra est-il regardé ;
      ("frame", requête, caméra) renvoyer la dernière image brute
                                 (FrameSnapshot, pour /api/capture et /bench/pose) ;
      ("reload",)               relire la base de visages (embeddings.npz) ;
      ("stop",).
    Si l'application disparaît, la connexion se ferme et le worker s'arrête.

Tout ce qui traverse la connexion (FrameResult, FrameSnapshot) est défini dans
core/frame_pipeline.py, pas ici : lancé avec -m, ce module s'appelle
`__main__` dans le worker, et l'application ne saurait pas désérialiser une
classe `__main__.X`.

L'adresse et la clé arrivent par l'environnement (VISIONCAM_WORKER_ADDRESS,
VISIONCAM_WORKER_KEY), les caméras par la ligne de commande (JSON).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client

import cv2

import config
from core import capture
from core.frame_pipeline import FramePipeline, FrameSnapshot, build_pose_estimator

logger = logging.getLogger(__name__)

ADDRESS_ENV = "VISIONCAM_WORKER_ADDRESS"
KEY_ENV = "VISIONCAM_WORKER_KEY"


@dataclass
class WorkerCamera:
    """Une caméra du worker : ce qu'il faut à core/capture.py et au pipeline."""
    id: str
    source: str | int
    is_file: bool
    tracks_path: str = ""


class _Counters:
    """Images jetées et sautées, remplies par core/capture.py."""

    def __init__(self):
        self.frames_dropped = 0
        self.source_skipped = 0


def encode_specs(cameras: list[WorkerCamera]) -> str:
    return json.dumps([camera.__dict__ for camera in cameras])


def decode_specs(text: str) -> list[WorkerCamera]:
    return [WorkerCamera(**item) for item in json.loads(text)]


def strip_for_transfer(result, jpeg, timings, counters, sizes):
    """Retire les images d'un FrameResult et y joint le JPEG et les mesures."""
    result.frame = None
    result.display_frame = None
    result.jpeg = jpeg
    result.stats = {'timings': timings, 'frames_dropped': counters.frames_dropped,
                    'source_skipped': counters.source_skipped, 'sizes': sizes}
    return result


class Worker:
    """Les caméras d'un processus : un thread de lecture et un de traitement chacune."""

    def __init__(self, cameras: list[WorkerCamera], results, build_tracker, pose_estimator,
                 reload_faces=lambda: None):
        self.cameras = cameras
        self._reload_faces = reload_faces
        # Dernière image brute et personnes de chaque caméra : ("frame", …).
        self._latest: dict = {}
        self._results = results
        self._send_lock = threading.Lock()
        self._build_tracker = build_tracker
        self._pose_estimator = pose_estimator
        self._stream = {camera.id: False for camera in cameras}
        self._stop = threading.Event()

    def running(self) -> bool:
        return not self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def handle(self, command) -> None:
        """Commande venue de l'application."""
        if command[0] == "stream":
            _, camera_id, wanted = command
            if camera_id in self._stream:
                self._stream[camera_id] = bool(wanted)
        elif command[0] == "frame":
            _, request_id, camera_id = command
            frame, persons = self._latest.get(camera_id, (None, []))
            self.send(FrameSnapshot(request_id, camera_id, frame, persons))
        elif command[0] == "reload":
            self._reload_faces()
        elif command[0] == "stop":
            self.stop()
        else:
            logger.warning("Commande inconnue : %r", command[0])

    def send(self, result) -> bool:
        """Envoie un résultat ; False si l'application n'écoute plus."""
        try:
            with self._send_lock:
                self._results.send(result)
            return True
        except (OSError, EOFError):
            self.stop()
            return False

    def start(self) -> list[threading.Thread]:
        threads = []
        for camera in self.cameras:
            frames: queue.Queue = queue.Queue(maxsize=2)
            counters = _Counters()
            threads += [
                threading.Thread(target=capture.capture_loop,
                                 args=(camera, frames, self.running, counters),
                                 daemon=True, name=f"camera-{camera.id}"),
                threading.Thread(target=self._process_loop, args=(camera, frames, counters),
                                 daemon=True, name=f"processing-{camera.id}"),
            ]
        for thread in threads:
            thread.start()
        return threads

    def _process_loop(self, camera: WorkerCamera, frames: queue.Queue, counters) -> None:
        logger.info("[%s] Thread de traitement démarré (processus %d).", camera.id, os.getpid())
        timings: list = []
        pipeline = FramePipeline(camera.id, self._build_tracker(),
                                 lambda stage, seconds: timings.append((stage, seconds)),
                                 self._pose_estimator, camera.tracks_path)
        while self.running():
            try:
                received, frame = frames.get(timeout=1.0)
            except queue.Empty:
                continue
            result = pipeline.process(received, frame)
            self._latest[camera.id] = (result.frame, result.persons)
            jpeg = None
            if self._stream[camera.id]:
                start = time.perf_counter()
                _, buffer = cv2.imencode('.jpg', result.display_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY])
                jpeg = buffer.tobytes()
                timings.append(('encode', time.perf_counter() - start))
            sizes = {**pipeline.tracker.state_sizes(), 'last_seen': len(pipeline.last_seen)}
            batch, timings[:] = list(timings), []
            if not self.send(strip_for_transfer(result, jpeg, batch, counters, sizes)):
                break
        pipeline.close()
        logger.info("[%s] Thread de traitement arrêté.", camera.id)


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cameras = decode_specs(argv[1])
    address, key = os.environ[ADDRESS_ENV], bytes.fromhex(os.environ[KEY_ENV])
    results = Client(address, authkey=key)
    commands = Client(address, authkey=key)

    # Imports lourds ici : décoder les arguments et se connecter ne doit pas
    # attendre torch ni TensorRT.
    from core.face_body_tracker import FaceBodyTracker
    from core.face_recognition import FaceRecognizer

    recognizer = FaceRecognizer(config.KNOWN_FACES_DIR, threshold=config.RECOGNITION_THRESHOLD,
                                cache_path=config.EMBEDDINGS_CACHE_PATH)
    worker = Worker(cameras, results, lambda: FaceBodyTracker(recognizer),
                    build_pose_estimator(), recognizer.reload_cache)
    threads = worker.start()
    logger.info("Processus de caméras %d : %s", os.getpid(), ", ".join(c.id for c in cameras))

    while worker.running():
        try:
            if commands.poll(1.0):
                worker.handle(commands.recv())
        except (OSError, EOFError):
            logger.warning("Application injoignable : arrêt du processus de caméras.")
            worker.stop()
    for thread in threads:
        thread.join(timeout=10)
    results.close()
    commands.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
