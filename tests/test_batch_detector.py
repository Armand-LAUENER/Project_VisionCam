"""
tests/test_batch_detector.py

Détection par lots partagée (core/batch_detector.py), sans GPU : `predict`
est un double qui renvoie, pour chaque image, son propre identifiant.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from core.batch_detector import BatchDetector


class FakeModel:
    def __init__(self, delay=0.0, error=None):
        self.delay, self.error = delay, error
        self.calls = []

    def predict(self, frames, **kwargs):
        self.calls.append(len(frames))
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return [f"det-{frame}" for frame in frames]


def make(model, **kw):
    return BatchDetector(model.predict, lambda result: [result], conf=0.5, iou=0.6, **kw)


def detect_concurrently(detector, frames):
    out = {}

    def one(frame):
        out[frame] = detector.detect(frame)

    threads = [threading.Thread(target=one, args=(f,)) for f in frames]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    return out


def test_each_caller_gets_its_own_detections():
    detector = make(FakeModel(delay=0.01), wait_s=0.02)

    out = detect_concurrently(detector, ["a", "b", "c", "d"])

    assert out == {f: [f"det-{f}"] for f in "abcd"}
    detector.close()


def test_frames_submitted_together_share_a_batch():
    model = FakeModel(delay=0.01)
    detector = make(model, wait_s=0.05)

    detect_concurrently(detector, ["a", "b", "c", "d"])

    assert max(model.calls) > 1
    assert detector.mean_batch() > 1
    detector.close()


def test_batches_never_exceed_the_maximum():
    model = FakeModel(delay=0.02)
    detector = make(model, max_batch=2, wait_s=0.05)

    detect_concurrently(detector, list("abcdefgh"))

    assert max(model.calls) <= 2 and sum(model.calls) == 8
    detector.close()


def test_a_lone_frame_does_not_wait_longer_than_the_window():
    detector = make(FakeModel(), wait_s=0.01)

    start = time.perf_counter()
    assert detector.detect("a") == ["det-a"]

    assert time.perf_counter() - start < 0.5
    detector.close()


def test_a_model_error_reaches_every_caller_of_the_batch():
    detector = make(FakeModel(error=RuntimeError("GPU")), wait_s=0.02)

    with pytest.raises(RuntimeError, match="GPU"):
        detector.detect("a")
    detector.close()


def test_a_tracker_with_a_shared_detector_loads_no_yolo(monkeypatch):
    from core import face_body_tracker

    monkeypatch.setattr(face_body_tracker, "build_body_tracker", MagicMock)
    monkeypatch.setattr(face_body_tracker, "load_yolo",
                        MagicMock(side_effect=AssertionError("YOLO chargé")))
    detections = [([0, 0, 10, 20], 0.9, "person", {})]

    tracker = face_body_tracker.FaceBodyTracker(MagicMock(), detector=lambda frame: detections)

    assert tracker.yolo is None
    assert tracker._detect_bodies(object()) == detections
