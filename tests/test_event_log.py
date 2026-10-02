"""
tests/test_event_log.py

Couvre le journal d'événements bruts (core/event_log.py) : transitions des
pistes (apparition, identification, perte du nom, fin), stockage en ajout
seul dans SQLite, horloge qui ne recule pas, rétention, renommage et
effacement d'une personne (RGPD), y compris d'un lancement à l'autre.
Horloges simulées, base temporaire.

Lancer : pytest tests/test_event_log.py -v
"""

import sqlite3

import pytest

from core.event_log import EventLog, ForwardClock, TrackEvents

T0 = 1_790_000_000.0


def kinds(events):
    return [(e["type"], e["track_id"], e["name"]) for e in events]


class TestTrackEvents:

    def feed(self, schedule, end_after=5.0):
        """schedule : [(secondes, [(track_id, nom)])] ; retourne tous les événements."""
        transitions = TrackEvents("cam0", end_after)
        events = []
        for offset, persons in schedule:
            events += transitions.update(persons, T0 + offset)
        return events

    def test_appearance_identification_and_end(self):
        events = self.feed([(0, [(1, "Inconnu")]),
                            (1, [(1, "Inconnu")]),
                            (2, [(1, "Alice")]),
                            (3, []),
                            (9, [])])

        assert kinds(events) == [("appeared", "1", None),
                                 ("identified", "1", "Alice"),
                                 ("ended", "1", None)]
        # La fin porte l'heure de la dernière image où la piste était visible.
        assert [e["time"] for e in events] == [T0, T0 + 2, T0 + 2]

    def test_name_lost_then_another_name(self):
        events = self.feed([(0, [(1, "Alice")]), (1, [(1, "Inconnu")]), (2, [(1, "Bob")])])

        assert kinds(events) == [("appeared", "1", None), ("identified", "1", "Alice"),
                                 ("unidentified", "1", None), ("identified", "1", "Bob")]

    def test_short_absence_is_not_an_end(self):
        """Piste en roue libre quelques images : pas de fin, pas de nouvelle apparition."""
        events = self.feed([(0, [(1, "Inconnu")]), (2, []), (4, [(1, "Inconnu")]), (8, [])])

        assert kinds(events) == [("appeared", "1", None)]

    def test_several_tracks_are_independent(self):
        events = self.feed([(0, [(1, "Inconnu"), (2, "Bob")]), (1, [(2, "Bob")]), (7, [])])

        assert sorted(kinds(events)) == sorted([("appeared", "1", None), ("appeared", "2", None),
                                                ("identified", "2", "Bob"),
                                                ("ended", "1", None), ("ended", "2", None)])

    def test_finish_ends_every_open_track(self):
        transitions = TrackEvents("cam0", 5.0)
        transitions.update([(1, "Alice"), (2, "Inconnu")], T0)

        events = transitions.finish()

        assert sorted(kinds(events)) == [("ended", "1", None), ("ended", "2", None)]
        assert transitions.update([], T0 + 100) == []

    def test_memory_is_bounded(self):
        transitions = TrackEvents("cam0", 5.0)
        for tid in range(100):
            transitions.update([(tid, "Inconnu")], T0 + tid)
        transitions.update([], T0 + 200)

        assert len(transitions._tracks) == 0


class TestForwardClock:

    def test_follows_the_monotonic_clock_from_the_wall_time(self):
        wall, mono = [T0], [50.0]
        clock = ForwardClock(wall=lambda: wall[0], monotonic=lambda: mono[0])
        mono[0] += 10
        wall[0] -= 0.8   # recalage de l'heure sous WSL2

        assert clock() == pytest.approx(T0 + 10)


class TestEventLog:

    def make(self, tmp_path, run_id="run-a", clock=None):
        return EventLog(str(tmp_path / "presence.db"), run_id=run_id,
                        clock=clock or (lambda: T0), retention=None)

    def event(self, track_id, kind, name=None, time=T0, camera="cam0"):
        return {"time": time, "camera_id": camera, "track_id": str(track_id),
                "type": kind, "name": name, "zone": None}

    def test_events_are_stored_in_order(self, tmp_path):
        log = self.make(tmp_path)
        log.write([self.event(1, "appeared"), self.event(1, "identified", "Alice", T0 + 2)])

        rows = log.events()

        assert [(r["type"], r["name"], r["run_id"]) for r in rows] == [
            ("appeared", None, "run-a"), ("identified", "Alice", "run-a")]

    def test_nothing_written_without_events(self, tmp_path):
        log = self.make(tmp_path)
        log.write([])

        assert not (tmp_path / "presence.db").exists()

    def test_forget_erases_every_event_of_the_person_tracks(self, tmp_path):
        """Apparition comprise, quand la piste n'avait pas encore de nom."""
        log = self.make(tmp_path)
        log.write([self.event(1, "appeared"), self.event(1, "identified", "Alice"),
                   self.event(1, "ended"), self.event(2, "appeared"),
                   self.event(2, "identified", "Bob")])

        assert log.forget("Alice") == 3
        assert {r["track_id"] for r in log.events()} == {"2"}

    def test_forget_keeps_same_track_number_of_another_run(self, tmp_path):
        """Les numéros de piste repartent de 1 à chaque lancement."""
        self.make(tmp_path, "run-a").write([self.event(1, "appeared"),
                                            self.event(1, "identified", "Alice")])
        later = self.make(tmp_path, "run-b")
        later.write([self.event(1, "appeared"), self.event(1, "identified", "Bob")])

        later.forget("Alice")

        assert [(r["run_id"], r["name"]) for r in later.events() if r["name"]] == [("run-b", "Bob")]

    def test_rename_moves_the_events(self, tmp_path):
        log = self.make(tmp_path)
        log.write([self.event(1, "identified", "Alice")])

        log.rename("Alice", "Alicia")

        assert [r["name"] for r in log.events()] == ["Alicia"]

    def test_retention_purges_old_events(self, tmp_path):
        now = [T0]
        log = EventLog(str(tmp_path / "presence.db"), run_id="run-a", clock=lambda: now[0],
                       retention=30 * 86400, purge_interval=3600)
        log.write([self.event(1, "appeared", time=T0), self.event(2, "appeared", time=T0 + 1)])

        now[0] = T0 + 30 * 86400 + 0.5
        log.write([self.event(3, "appeared", time=now[0])])

        assert {r["track_id"] for r in log.events()} == {"2", "3"}

    def test_public_interface_has_no_event_update(self, tmp_path):
        """Ajout seul : aucune méthode ne modifie un événement, hors renommage et RGPD."""
        public = {n for n in dir(EventLog) if not n.startswith("_")}

        assert public == {"write", "events", "forget", "rename", "purge"}

    def test_schema(self, tmp_path):
        log = self.make(tmp_path)
        log.write([self.event(1, "appeared")])

        conn = sqlite3.connect(tmp_path / "presence.db")
        columns = [row[1] for row in conn.execute("PRAGMA table_info(events)")]
        conn.close()

        assert columns == ["id", "time", "run_id", "camera_id", "track_id", "type", "name", "zone"]
