"""
events.py — Temps réel et mesures du pipeline, indépendants de Flask.

- EventBus       : diffusion d'événements vers les pages ouvertes (Server-Sent
                   Events). Chaque abonné a sa file bornée : un client lent perd
                   ses plus vieux événements au lieu de bloquer le traitement.
- StageTimer     : temps des étapes du pipeline sur une fenêtre glissante.
- UnknownWatcher : signale une fois une personne restée « Inconnu » plus
                   longtemps qu'un délai (une piste neuve n'a pas encore de nom).
"""

from __future__ import annotations

import queue
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager

UNKNOWN = "Inconnu"


class EventBus:
    def __init__(self, max_queue: int = 100) -> None:
        self._max_queue = max_queue
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def publish(self, event: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            while True:
                try:
                    q.put_nowait(event)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass


class StageTimer:
    def __init__(self, window: int = 300) -> None:
        self._window = window
        self._samples: dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._samples.setdefault(stage, deque(maxlen=self._window)).append(seconds)

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - start)

    def summary(self) -> dict[str, dict]:
        """{étape: {median_ms, p95_ms, samples}} sur la fenêtre glissante."""
        with self._lock:
            snapshot = {stage: list(values) for stage, values in self._samples.items()}
        result = {}
        for stage, values in snapshot.items():
            if not values:
                continue
            ordered = sorted(values)
            p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
            result[stage] = {"median_ms": round(statistics.median(ordered) * 1000, 2),
                             "p95_ms": round(p95 * 1000, 2), "samples": len(ordered)}
        return result


class UnknownWatcher:
    def __init__(self, delay_s: float = 3.0) -> None:
        self.delay_s = delay_s
        self._first_unknown: dict[str, float] = {}
        self._alerted: set[str] = set()

    def update(self, tracks, now: float) -> list[str]:
        """tracks : [(track_id, nom)] visibles ; retourne les pistes à signaler."""
        visible = {str(tid) for tid, _ in tracks}
        alerts = []
        for track_id, name in tracks:
            track_id = str(track_id)
            if name != UNKNOWN:
                self._first_unknown.pop(track_id, None)
                continue
            since = self._first_unknown.setdefault(track_id, now)
            if now - since >= self.delay_s and track_id not in self._alerted:
                self._alerted.add(track_id)
                alerts.append(track_id)
        # Oublier les pistes disparues : la mémoire ne doit pas grossir sans fin.
        for track_id in set(self._first_unknown) - visible:
            del self._first_unknown[track_id]
        self._alerted &= visible
        return alerts
