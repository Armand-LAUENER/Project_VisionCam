"""
tests/test_events.py

Couvre core/events.py : diffusion d'événements, mesures par étape et alerte
des personnes restées inconnues.

Lancer : pytest tests/test_events.py -v
"""

import pytest

from core.events import EventBus, StageTimer, UnknownWatcher


class TestEventBus:

    def test_every_subscriber_receives_events(self):
        bus = EventBus()
        a, b = bus.subscribe(), bus.subscribe()

        bus.publish({"type": "arrival", "name": "Alice"})

        assert a.get_nowait() == b.get_nowait() == {"type": "arrival", "name": "Alice"}
        assert bus.subscriber_count == 2

    def test_slow_subscriber_loses_oldest_events_without_blocking(self):
        bus = EventBus(max_queue=3)
        q = bus.subscribe()

        for i in range(10):
            bus.publish({"n": i})

        assert [q.get_nowait()["n"] for _ in range(3)] == [7, 8, 9]

    def test_unsubscribed_queue_receives_nothing(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.unsubscribe(q)

        bus.publish({"type": "x"})

        assert q.empty() and bus.subscriber_count == 0


class TestStageTimer:

    def test_median_and_p95_per_stage(self):
        timer = StageTimer(window=100)
        for ms in range(1, 101):
            timer.record("yolo", ms / 1000)
        timer.record("encode", 0.004)

        summary = timer.summary()

        assert summary["yolo"]["median_ms"] == pytest.approx(50.5)
        assert summary["yolo"]["p95_ms"] == pytest.approx(96.0)
        assert summary["encode"] == {"median_ms": 4.0, "p95_ms": 4.0, "samples": 1}

    def test_window_keeps_only_recent_samples(self):
        timer = StageTimer(window=3)
        for seconds in (1.0, 1.0, 0.001, 0.001, 0.001):
            timer.record("stage", seconds)

        assert timer.summary()["stage"]["p95_ms"] == 1.0

    def test_measure_context_manager(self):
        timer = StageTimer()
        with timer.measure("block"):
            pass

        assert timer.summary()["block"]["samples"] == 1


class TestUnknownWatcher:

    def test_alerts_once_after_the_delay(self):
        watcher = UnknownWatcher(delay_s=3)

        assert watcher.update([("1", "Inconnu")], now=0) == []
        assert watcher.update([("1", "Inconnu")], now=2.9) == []
        assert watcher.update([("1", "Inconnu")], now=3.0) == ["1"]
        assert watcher.update([("1", "Inconnu")], now=10) == []

    def test_named_before_the_delay_is_not_an_alert(self):
        watcher = UnknownWatcher(delay_s=3)

        watcher.update([("1", "Inconnu")], now=0)
        watcher.update([("1", "Armand")], now=1)

        assert watcher.update([("1", "Armand")], now=5) == []

    def test_forgets_tracks_that_left(self):
        watcher = UnknownWatcher(delay_s=3)
        watcher.update([("1", "Inconnu")], now=0)
        watcher.update([("1", "Inconnu")], now=3)

        watcher.update([], now=4)

        assert watcher._first_unknown == {} and watcher._alerted == set()

    def test_integer_track_ids_from_the_rust_backend(self):
        watcher = UnknownWatcher(delay_s=0)

        assert watcher.update([(7, "Inconnu")], now=0) == ["7"]
