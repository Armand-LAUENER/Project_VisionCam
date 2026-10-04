"""
tests/test_db_writer.py

Couvre le thread d'écriture SQLite (core/db_writer.py) : les écritures quittent
la boucle vidéo, s'exécutent dans l'ordre, les lectures voient les écritures
en attente, et l'arrêt vide la file. Puis l'historique de présence et le
journal d'événements en arrière-plan : leurs appels rendent la main même
quand le disque est lent.

Bug d'origine : en endurance, ces écritures se faisaient dans la boucle de
traitement, une connexion et un fsync par opération ; quand le disque
attendait (jusqu'à 5,3 s), toute l'image attendait (jusqu'à 14 s).

Lancer : pytest tests/test_db_writer.py -v
"""

import sqlite3
import threading
import time

import pytest

from core.db_writer import DbWriter
from core.event_log import EventLog
from core.presence_log import PresenceLog

T0 = 1_790_000_000.0


def make_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")


@pytest.fixture(params=[True, False], ids=["arrière-plan", "synchrone"])
def writer(request, tmp_path):
    w = DbWriter(str(tmp_path / "base.db"), background=request.param)
    w.call(make_table)
    yield w
    w.close()


class TestDbWriter:

    def test_writes_run_in_order_and_reads_see_them(self, writer):
        for v in "abc":
            writer.submit(lambda conn, v=v: conn.execute("INSERT INTO t (v) VALUES (?)", (v,)))

        rows = writer.call(lambda conn: [r[0] for r in conn.execute("SELECT v FROM t ORDER BY id")])

        assert rows == ["a", "b", "c"]

    def test_failed_write_does_not_stop_the_next_ones(self, writer, caplog):
        writer.submit(lambda conn: conn.execute("INSERT INTO absente VALUES (1)"))
        writer.submit(lambda conn: conn.execute("INSERT INTO t (v) VALUES ('suite')"))

        assert writer.call(lambda conn: [tuple(r) for r in conn.execute("SELECT v FROM t")]) == [("suite",)]
        assert "absente" in caplog.text

    def test_writes_are_durable_after_close(self, writer, tmp_path):
        writer.submit(lambda conn: conn.execute("INSERT INTO t (v) VALUES ('x')"))
        writer.close()

        conn = sqlite3.connect(tmp_path / "base.db")
        assert conn.execute("SELECT v FROM t").fetchall() == [("x",)]
        conn.close()

    def test_nothing_is_created_before_the_first_task(self, tmp_path):
        w = DbWriter(str(tmp_path / "sub" / "base.db"))
        assert not (tmp_path / "sub").exists()
        w.close()


class TestBackground:

    def test_submit_returns_while_the_disk_is_slow(self, tmp_path):
        w = DbWriter(str(tmp_path / "base.db"), background=True)
        w.call(make_table)
        release = threading.Event()
        w.submit(lambda conn: release.wait(5))   # disque qui attend

        start = time.perf_counter()
        w.submit(lambda conn: conn.execute("INSERT INTO t (v) VALUES ('y')"))
        elapsed = time.perf_counter() - start

        release.set()
        assert elapsed < 0.05
        assert w.call(lambda conn: [tuple(r) for r in conn.execute("SELECT v FROM t")]) == [("y",)]
        w.close()

    def test_flush_waits_for_pending_writes(self, tmp_path):
        w = DbWriter(str(tmp_path / "base.db"), background=True)
        w.call(make_table)
        w.submit(lambda conn: (time.sleep(0.2), conn.execute("INSERT INTO t (v) VALUES ('z')")))

        w.flush()

        conn = sqlite3.connect(tmp_path / "base.db")
        assert conn.execute("SELECT v FROM t").fetchall() == [("z",)]
        conn.close()
        w.close()

    def test_wal_without_fsync_on_every_commit(self, tmp_path):
        w = DbWriter(str(tmp_path / "base.db"), background=True)

        modes = w.call(lambda conn: (conn.execute("PRAGMA journal_mode").fetchone()[0],
                                     conn.execute("PRAGMA synchronous").fetchone()[0]))

        assert modes == ("wal", 1)   # 1 = NORMAL
        w.close()


def slow_writer(tmp_path):
    """Écrivain en arrière-plan dont la première écriture attend 2 s, comme le disque en endurance."""
    w = DbWriter(str(tmp_path / "presence.db"), background=True)
    release = threading.Event()
    w.submit(lambda conn: release.wait(2))
    return w, release


class TestPresenceLogInBackground:

    def test_update_returns_at_once_and_history_is_complete(self, tmp_path):
        w, release = slow_writer(tmp_path)
        log = PresenceLog(str(tmp_path / "presence.db"), gap=60, writer=w)

        start = time.perf_counter()
        events = log.update(["Alice"], now=T0) + log.update(["Alice"], now=T0 + 20)
        events += log.update([], now=T0 + 100)
        elapsed = time.perf_counter() - start

        release.set()
        assert elapsed < 0.05
        assert [e["type"] for e in events] == ["arrival", "departure"]
        sessions = log.sessions()
        assert [(s["name"], s["duration_s"], s["ongoing"]) for s in sessions] == [("Alice", 20, False)]
        w.close()

    def test_session_ids_continue_after_restart(self, tmp_path):
        w = DbWriter(str(tmp_path / "presence.db"), background=True)
        log = PresenceLog(str(tmp_path / "presence.db"), gap=60, writer=w)
        log.update(["Alice"], now=T0)
        log.update([], now=T0 + 100)
        w.close()

        w2 = DbWriter(str(tmp_path / "presence.db"), background=True)
        log2 = PresenceLog(str(tmp_path / "presence.db"), gap=60, writer=w2)
        log2.update(["Bob"], now=T0 + 200)
        log2.close_all()

        assert sorted(s["name"] for s in log2.sessions()) == ["Alice", "Bob"]
        assert len({s["id"] for s in log2.sessions()}) == 2
        w2.close()

    def test_forget_is_done_when_it_returns(self, tmp_path):
        """Droit à l'effacement : après l'appel, la donnée n'est plus sur le disque."""
        w = DbWriter(str(tmp_path / "presence.db"), background=True)
        log = PresenceLog(str(tmp_path / "presence.db"), gap=60, writer=w)
        log.update(["Alice"], now=T0)
        log.update(["Alice", "Bob"], now=T0 + 30)

        assert log.forget("Alice") == 1

        conn = sqlite3.connect(tmp_path / "presence.db")
        assert [r[0] for r in conn.execute("SELECT name FROM sessions")] == ["Bob"]
        conn.close()
        w.close()


class TestEventLogInBackground:

    def test_write_returns_at_once(self, tmp_path):
        w, release = slow_writer(tmp_path)
        log = EventLog(str(tmp_path / "presence.db"), run_id="run-a", writer=w)
        event = {"time": T0, "camera_id": "cam0", "track_id": "1", "type": "appeared",
                 "name": None, "zone": None}

        start = time.perf_counter()
        log.write([event])
        elapsed = time.perf_counter() - start

        release.set()
        assert elapsed < 0.05
        assert [e["type"] for e in log.events()] == ["appeared"]
        w.close()
