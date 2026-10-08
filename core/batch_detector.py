"""
batch_detector.py — YOLO-Pose partagé par les caméras d'un processus, par lots.

Un moteur TensorRT à taille de lot variable traite 4 images en 3,2 ms chacune,
contre 4,8 ms une à une (docs/performance.md, « YOLO par lots ») ; le GPU est
le plafond suivant de l'application. Les threads de traitement des caméras
d'un processus de caméras soumettent leur image à `detect()`, qui bloque
jusqu'au résultat ; un thread unique regroupe les images en attente (au plus
`max_batch`, en attendant au plus `wait_s` après la première), lance un seul
`predict` et rend à chacun ses détections.

Sous charge, les images s'accumulent d'elles-mêmes pendant qu'un lot passe
sur le GPU : l'attente ne sert qu'à regrouper des images arrivées presque
ensemble. Un seul modèle par processus au lieu d'un par caméra : moins de
RAM et de VRAM.

`predict` et `parse` sont injectés : `YOLO(...).predict` et
`FaceBodyTracker._parse_result` dans l'application, des doubles dans les tests.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter

logger = logging.getLogger(__name__)


class _Request:
    __slots__ = ("frame", "done", "detections", "error")

    def __init__(self, frame):
        self.frame = frame
        self.done = threading.Event()
        self.detections = None
        self.error = None


class BatchDetector:
    """Détection YOLO-Pose par lots, partagée entre threads."""

    def __init__(self, predict, parse, *, conf: float, iou: float, max_batch: int = 4,
                 wait_s: float = 0.002, clock=time.monotonic):
        self._predict = predict
        self._parse = parse
        self._conf = conf
        self._iou = iou
        self.max_batch = max_batch
        self.wait_s = wait_s
        self._clock = clock
        self._requests: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        # Taille des lots lancés : moyenne relevée dans les journaux.
        self.batch_sizes: Counter = Counter()
        self._thread = threading.Thread(target=self._run, daemon=True, name="batch-detector")
        self._thread.start()

    def detect(self, frame) -> list[tuple]:
        """Détections d'une image, au format de FaceBodyTracker._parse_result."""
        request = _Request(frame)
        self._requests.put(request)
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.detections

    def _collect(self) -> list[_Request] | None:
        try:
            batch = [self._requests.get(timeout=0.5)]
        except queue.Empty:
            return None
        deadline = self._clock() + self.wait_s
        while len(batch) < self.max_batch:
            try:
                # Les images déjà en file passent sans attendre.
                batch.append(self._requests.get_nowait())
                continue
            except queue.Empty:
                pass
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            try:
                batch.append(self._requests.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue
            self.batch_sizes[len(batch)] += 1
            try:
                results = self._predict([r.frame for r in batch], classes=[0], conf=self._conf,
                                        iou=self._iou, device=0, verbose=False)
                for request, result in zip(batch, results):
                    request.detections = self._parse(result)
            except Exception as e:
                for request in batch:
                    request.error = e
            finally:
                for request in batch:
                    request.done.set()

    def mean_batch(self) -> float:
        frames = sum(size * n for size, n in self.batch_sizes.items())
        batches = sum(self.batch_sizes.values())
        return frames / batches if batches else 0.0

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        logger.info("Détection par lots : %.2f image(s) par lot en moyenne (%s)",
                    self.mean_batch(), dict(sorted(self.batch_sizes.items())))
