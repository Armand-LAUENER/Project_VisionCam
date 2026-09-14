"""
tests/test_presence_log.py

Couvre l'historique des présences : ouverture et fermeture des sessions,
tolérance aux courtes absences, reprise après un arrêt brutal, renommage,
effacement, filtres et export CSV. Horloge simulée, base SQLite temporaire.

Lancer : pytest tests/test_presence_log.py -v
"""

import csv
import io
import os
import sqlite3
from datetime import datetime

from core.presence_log import PresenceLog

T0 = datetime(2026, 9, 14, 9, 0, 0).timestamp()


def make_log(tmp_path, gap=60, flush=10):
    return PresenceLog(str(tmp_path / "presence.db"), gap=gap, flush_interval=flush)


def feed(log, schedule):
    """schedule : [(secondes depuis T0, noms vus)] ; retourne tous les événements."""
    events = []
    for offset, names in schedule:
        events += log.update(names, now=T0 + offset)
    return events


class TestSessions:

    def test_arrival_then_departure_after_the_gap(self, tmp_path):
        log = make_log(tmp_path)

        events = feed(log, [(0, ["Alice"]), (30, ["Alice"]), (60, []), (100, [])])

        assert [(e["type"], e["name"]) for e in events] == [("arrival", "Alice"),
                                                            ("departure", "Alice")]
        (session,) = log.sessions()
        assert session["arrived"] == "2026-09-14T09:00:00"
        assert session["departed"] == "2026-09-14T09:00:30"
        assert session["duration_s"] == 30 and not session["ongoing"]

    def test_short_absence_does_not_split_the_session(self, tmp_path):
        log = make_log(tmp_path, gap=60)

        feed(log, [(0, ["Alice"]), (10, []), (50, []), (55, ["Alice"]), (200, [])])

        assert len(log.sessions()) == 1

    def test_absence_longer_than_the_gap_starts_a_new_session(self, tmp_path):
        log = make_log(tmp_path, gap=60)

        feed(log, [(0, ["Alice"]), (100, []), (200, ["Alice"])])

        assert [s["ongoing"] for s in log.sessions()] == [True, False]

    def test_unknown_people_are_not_logged(self, tmp_path):
        log = make_log(tmp_path)

        assert feed(log, [(0, ["Inconnu", ""])]) == []
        assert log.sessions() == []

    def test_people_have_independent_sessions(self, tmp_path):
        log = make_log(tmp_path)

        feed(log, [(0, ["Alice"]), (10, ["Alice", "Bob"]), (20, ["Bob"]), (100, ["Bob"])])

        by_name = {s["name"]: s for s in log.sessions()}
        assert not by_name["Alice"]["ongoing"] and by_name["Bob"]["ongoing"]
        assert log.names() == ["Alice", "Bob"]

    def test_ongoing_duration_follows_the_live_last_seen(self, tmp_path):
        """La dernière vue n'est écrite que toutes les 10 s ; la lecture reste à jour."""
        log = make_log(tmp_path, flush=10)

        feed(log, [(0, ["Alice"]), (4, ["Alice"])])

        (session,) = log.sessions()
        assert session["ongoing"] and session["duration_s"] == 4


class TestDurability:

    def test_writes_are_throttled(self, tmp_path):
        log = make_log(tmp_path, flush=10)
        feed(log, [(t, ["Alice"]) for t in range(0, 30)])

        db = sqlite3.connect(str(tmp_path / "presence.db"))
        (last_seen,) = db.execute("SELECT last_seen FROM sessions").fetchone()
        assert last_seen == T0 + 20

    def test_sessions_left_open_by_a_crash_are_closed_on_restart(self, tmp_path):
        """Fermée à la dernière vue écrite : 15 s (écrite, 15 s après la précédente),
        pas 18 s (encore en mémoire au moment de l'arrêt)."""
        crashed = make_log(tmp_path, flush=10)
        feed(crashed, [(0, ["Alice"]), (15, ["Alice"]), (18, ["Alice"])])

        restarted = make_log(tmp_path)
        (session,) = restarted.sessions()

        assert not session["ongoing"]
        assert session["departed"] == "2026-09-14T09:00:15"

    def test_clean_shutdown_closes_sessions(self, tmp_path):
        log = make_log(tmp_path)
        feed(log, [(0, ["Alice"]), (7, ["Alice"])])

        log.close_all()

        (session,) = make_log(tmp_path).sessions()
        assert session["duration_s"] == 7

    def test_importing_does_not_create_the_database(self, tmp_path):
        make_log(tmp_path)
        assert not os.path.exists(tmp_path / "presence.db")


class TestPeopleChanges:

    def test_rename_moves_history_and_open_session(self, tmp_path):
        log = make_log(tmp_path)
        feed(log, [(0, ["Alice"]), (100, []), (200, ["Alice"])])

        log.rename("Alice", "Alicia")
        feed(log, [(210, ["Alicia"])])

        assert {s["name"] for s in log.sessions()} == {"Alicia"}
        assert len(log.sessions()) == 2

    def test_forget_erases_all_history(self, tmp_path):
        log = make_log(tmp_path)
        feed(log, [(0, ["Alice", "Bob"]), (100, []), (200, ["Alice"])])

        assert log.forget("Alice") == 2

        assert [s["name"] for s in log.sessions()] == ["Bob"]
        assert feed(log, [(210, ["Alice"])])[0]["type"] == "arrival"


class TestQueries:

    def test_filters_by_name_and_period(self, tmp_path):
        log = make_log(tmp_path, gap=60)
        feed(log, [(0, ["Alice"]), (100, []),                       # 09:00
                   (3600, ["Alice", "Bob"]), (3700, []),            # 10:00
                   (86400, ["Bob"]), (86500, [])])                  # lendemain

        assert len(log.sessions(name="Alice")) == 2
        ten_to_eleven = log.sessions(start=T0 + 3000, end=T0 + 7200)
        assert sorted(s["name"] for s in ten_to_eleven) == ["Alice", "Bob"]
        assert [s["name"] for s in log.sessions(start=T0 + 80000)] == ["Bob"]

    def test_csv_export(self, tmp_path):
        log = make_log(tmp_path)
        feed(log, [(0, ["Alice"]), (30, ["Alice"]), (200, [])])

        rows = list(csv.reader(io.StringIO(PresenceLog.to_csv(log.sessions()))))

        assert rows[0] == ["nom", "arrivée", "départ", "durée (s)", "en cours"]
        assert rows[1] == ["Alice", "2026-09-14T09:00:00", "2026-09-14T09:00:30", "30", "non"]
