"""
tests/test_webcam_bridge.py

Regression test — le pont webcam n'envoie chaque image qu'une fois.

Bug d'origine : le flux MJPEG ré-encodait et renvoyait la dernière image en
boucle, sans attendre la suivante. Mesuré : 556 images/s envoyées pour 30
réellement différentes, un cœur du processeur Windows occupé en permanence
(coupures du son de Discord), et VisionCam qui traitait des doublons.
Fix : encodage unique dans le thread de capture, et attente de l'image
suivante par client. La webcam est aussi ouverte en MJPG, pour ne pas saturer
l'USB partagé avec un casque.

Lancer : pytest tests/test_webcam_bridge.py -v
"""

import threading
import time

import cv2
import numpy as np

from tools.webcam_bridge import CameraStream


class FakeCapture:
    """Webcam simulée : `frames` images, une toutes les `period` secondes."""

    def __init__(self, frames=3, period=0.05):
        self.remaining = frames
        self.period = period
        self.settings = []
        self.reads = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.settings.append(prop)
        return True

    def get(self, prop):
        return 0

    def read(self):
        time.sleep(self.period)
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        self.reads += 1
        return True, np.full((24, 32, 3), 10 * self.reads, dtype=np.uint8)

    def release(self):
        pass


def test_each_frame_is_delivered_once():
    camera = CameraStream(0, 32, 24, 30, capture=FakeCapture(frames=3))
    try:
        seq, delivered = 0, []
        deadline = time.monotonic() + 3
        while len(delivered) < 3 and time.monotonic() < deadline:
            seq, jpeg = camera.wait_jpeg(seq, timeout=0.5)
            if jpeg is not None:
                delivered.append(seq)

        assert delivered == [1, 2, 3]
        # Plus aucune nouvelle image : l'attente expire sans renvoyer la dernière.
        assert camera.wait_jpeg(seq, timeout=0.2) == (seq, None)
    finally:
        camera.stop()


def test_frames_are_encoded_once_whatever_the_number_of_clients(monkeypatch):
    calls = []
    real_imencode = cv2.imencode
    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: calls.append(1) or real_imencode(*a, **k))
    camera = CameraStream(0, 32, 24, 30, capture=FakeCapture(frames=4, period=0.03))
    received = {0: [], 1: [], 2: []}

    def client(n):
        seq = 0
        for _ in range(20):
            seq, jpeg = camera.wait_jpeg(seq, timeout=0.3)
            if jpeg is not None:
                received[n].append(seq)

    threads = [threading.Thread(target=client, args=(n,)) for n in received]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
    finally:
        camera.stop()

    assert len(calls) == 4
    assert all(len(seqs) == len(set(seqs)) for seqs in received.values())


def test_mjpg_is_requested_before_the_resolution():
    capture = FakeCapture(frames=0)
    camera = CameraStream(0, 1280, 720, 30, capture=capture)
    camera.stop()

    assert capture.settings.index(cv2.CAP_PROP_FOURCC) < capture.settings.index(cv2.CAP_PROP_FRAME_WIDTH)


def test_capture_failure_wakes_waiting_clients():
    """Webcam débranchée : un client en attente ne doit pas rester bloqué."""
    camera = CameraStream(0, 32, 24, 30, capture=FakeCapture(frames=0, period=0.001))
    try:
        start = time.monotonic()
        while camera.running and time.monotonic() - start < 2:
            camera.wait_jpeg(0, timeout=0.1)
        assert not camera.running
    finally:
        camera.stop()
