"""
tests/test_history.py

Couvre l'historique servi par l'API (core/history.py, roadmap 2.1) : les
sessions sont recalculées à partir du journal d'événements, la donnée source ;
la table `sessions` ne fournit plus que l'historique antérieur au premier
événement. Mêmes filtres et même forme de réponse qu'avant.

Lancer : pytest tests/test_history.py -v
"""

from datetime import datetime

import pytest

from core.event_log import EventLog, TrackEvents
from core.history import history_names, presence_history
from core.presence_log import PresenceLog

T0 = datetime(2026, 10, 4, 9, 0, 0).timestamp()
GAP, FACE_MAX_AGE = 60.0, 300.0


@pytest.fixture
def logs(tmp_path):
    db = str(tmp_path / "presence.db")
    return PresenceLog(db, gap=GAP), EventLog(db, run_id="run-a")


def track_visit(event_log, track_id, name, start, end, step=1.0):
    """Une piste visible et reconnue (visage à chaque image) de start à end, puis finie."""
    transitions = TrackEvents("cam0", 5.0, face_max_age=FACE_MAX_AGE, face_interval=60.0)
    frame, t = 0, start
    while t <= end:
        event_log.write(transitions.update([(track_id, name, frame)], T0 + t))
        frame, t = frame + 1, t + step
    event_log.write(transitions.finish())


def history(logs, now=T0 + 10_000, **filters):
    presence_log, event_log = logs
    return presence_history(presence_log, event_log, GAP, FACE_MAX_AGE, now, **filters)


class TestFromEvents:

    def test_sessions_come_from_the_events(self, logs):
        track_visit(logs[1], 1, "Alice", 0, 120)
        track_visit(logs[1], 2, "Bob", 50, 80)

        sessions = history(logs)

        assert [(s["name"], s["duration_s"], s["ongoing"]) for s in sessions] == [
            ("Bob", 30, False), ("Alice", 120, False)]
        assert set(sessions[0]) == {"id", "name", "arrived", "departed", "last_seen",
                                    "ongoing", "duration_s"}

    def test_ongoing_session(self, logs):
        transitions = TrackEvents("cam0", 5.0, face_max_age=FACE_MAX_AGE, face_interval=60.0)
        for i in range(31):
            logs[1].write(transitions.update([(1, "Alice", i)], T0 + i))

        [session] = history(logs, now=T0 + 30)

        assert session["ongoing"] and session["departed"] is None
        assert session["duration_s"] == 30

    def test_filters(self, logs):
        track_visit(logs[1], 1, "Alice", 0, 60)
        track_visit(logs[1], 2, "Bob", 86_400, 86_460)   # le lendemain

        assert [s["name"] for s in history(logs, name="Bob")] == ["Bob"]
        assert [s["name"] for s in history(logs, now=T0 + 90_000, start=T0 + 80_000)] == ["Bob"]
        assert [s["name"] for s in history(logs, now=T0 + 90_000, end=T0 + 80_000)] == ["Alice"]
        assert len(history(logs, now=T0 + 90_000, limit=1)) == 1


class TestLegacyHistory:

    def test_sessions_before_the_first_event_are_kept(self, logs):
        """Historique d'avant le journal d'événements : seule la table le connaît."""
        presence_log, event_log = logs
        presence_log.update(["Carol"], now=T0 - 7200)
        presence_log.update([], now=T0 - 3000)
        track_visit(event_log, 1, "Alice", 0, 60)

        assert [s["name"] for s in history(logs)] == ["Alice", "Carol"]

    def test_cache_rows_after_the_first_event_are_not_doubled(self, logs):
        """La table reste écrite au fil de l'eau : ses sessions récentes doublent les événements."""
        presence_log, event_log = logs
        track_visit(event_log, 1, "Alice", 0, 60)
        presence_log.update(["Alice"], now=T0)
        presence_log.update(["Alice"], now=T0 + 60)
        presence_log.close_all()

        assert [s["name"] for s in history(logs)] == ["Alice"]

    def test_without_events_the_table_is_the_history(self, logs):
        presence_log, _ = logs
        presence_log.update(["Carol"], now=T0)
        presence_log.update([], now=T0 + 100)

        assert [(s["name"], s["duration_s"]) for s in history(logs)] == [("Carol", 0)]


def test_names_cover_both_sources(logs):
    presence_log, event_log = logs
    presence_log.update(["Carol"], now=T0 - 7200)
    track_visit(event_log, 1, "Alice", 0, 10)

    assert history_names(presence_log, event_log) == ["Alice", "Carol"]
