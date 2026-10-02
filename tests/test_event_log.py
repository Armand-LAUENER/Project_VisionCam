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


# ─────────────────────────────────────────────────────────────────────────────
# Sessions de présence recalculées à partir des événements (roadmap 2.1)
# ─────────────────────────────────────────────────────────────────────────────

FPS = 5.0
GAP = 60.0
FACE_MAX_AGE = 300.0


class Scene:
    """Pistes scriptées image par image : visibilité, nom, reconnaissances du visage."""

    def __init__(self):
        self.tracks = []   # (track_id, début, fin, [(depuis, nom)], visage(t) -> bool)

    def track(self, track_id, start, end, names, face=lambda t: True, hidden=()):
        self.tracks.append((track_id, start, end, names, face, hidden))
        return self

    def frames(self):
        end = max(t[2] for t in self.tracks) + 10
        last_face = {}
        for i in range(int(end * FPS)):
            now = i / FPS
            persons = []
            for track_id, start, stop, names, face, hidden in self.tracks:
                if not (start <= now <= stop) or any(a <= now < b for a, b in hidden):
                    continue
                name = [n for since, n in names if since <= now][-1]
                if name != "Inconnu" and face(now):
                    last_face[track_id] = i
                elif name == "Inconnu":
                    last_face.pop(track_id, None)
                persons.append((track_id, name, last_face.get(track_id, -1)))
            yield now, persons


def presence_sessions(scene, tmp_path):
    """Le calcul actuel, au fil de l'eau : FaceConfirmedNames puis PresenceLog."""
    from core.presence_log import FaceConfirmedNames, PresenceLog

    log = PresenceLog(str(tmp_path / "live.db"), gap=GAP)
    confirmed = FaceConfirmedNames(FACE_MAX_AGE)
    for now, persons in scene.frames():
        log.update(confirmed.update(persons, T0 + now), now=T0 + now)
    log.close_all()
    conn = sqlite3.connect(tmp_path / "live.db")
    rows = conn.execute("SELECT name, arrived, departed FROM sessions ORDER BY name, arrived")
    sessions = [tuple(r) for r in rows]
    conn.close()
    return sessions


def event_sessions(scene):
    from core.event_log import sessions_from_events

    transitions = TrackEvents("cam0", 5.0, face_max_age=FACE_MAX_AGE, face_interval=60.0)
    events = []
    for now, persons in scene.frames():
        events += transitions.update(persons, T0 + now)
    events += transitions.finish()
    for i, e in enumerate(events):
        e.update(id=i, run_id="run-a")
    return sorted((s["name"], s["arrived"], s["departed"])
                  for s in sessions_from_events(events, GAP, FACE_MAX_AGE)), events


def assert_same(live, rebuilt):
    assert [s[0] for s in live] == [s[0] for s in rebuilt]
    for (_, a1, d1), (_, a2, d2) in zip(live, rebuilt):
        assert a2 == pytest.approx(a1, abs=1.01 / FPS)
        assert d2 == pytest.approx(d1, abs=1.01 / FPS)


class TestSessionsFromEvents:

    @pytest.mark.parametrize("scene", [
        # Une personne reconnue en continu.
        Scene().track(1, 0, 400, [(0, "Alice")]),
        # Reconnue puis de dos : la marge de 5 min s'écoule, puis elle se retourne.
        Scene().track(1, 0, 1500, [(0, "Alice")],
                      face=lambda t: t < 200 or t > 900),
        # Reconnaissances espacées (toutes les 40 s) : jamais plus de 5 min sans visage.
        Scene().track(1, 0, 1200, [(0, "Alice")], face=lambda t: int(t) % 40 == 0),
        # Courte occultation : la session ne se coupe pas.
        Scene().track(1, 0, 300, [(0, "Alice")], hidden=[(100, 103)]),
        # Inconnue d'abord, nommée ensuite, puis le nom tombe (échange de pistes).
        Scene().track(1, 0, 600, [(0, "Inconnu"), (20, "Alice"), (300, "Inconnu")]),
        # Le nom change sur la piste, et deux pistes portent le même nom à la suite.
        Scene().track(1, 0, 500, [(0, "Alice"), (250, "Bob")])
               .track(2, 520, 700, [(520, "Alice")]),
        # Deux personnes en même temps, l'une revient après plus d'une minute.
        Scene().track(1, 0, 300, [(0, "Alice")])
               .track(2, 50, 200, [(50, "Bob")])
               .track(3, 400, 600, [(400, "Bob")]),
    ], ids=["continue", "de-dos", "espacees", "occultation", "nom-perdu", "nom-change",
            "deux-personnes"])
    def test_same_sessions_as_the_live_history(self, scene, tmp_path):
        rebuilt, _ = event_sessions(scene)

        assert_same(presence_sessions(scene, tmp_path), rebuilt)

    def test_face_events_are_throttled(self):
        """Visage reconnu à chaque image pendant 10 min : une ligne par minute environ."""
        _, events = event_sessions(Scene().track(1, 0, 600, [(0, "Alice")]))

        faces = [e for e in events if e["type"] == "face"]
        assert 9 <= len(faces) <= 12


def test_face_frames_without_face_events_are_ignored():
    """Sans face_max_age, last_face_frame est accepté mais ne produit rien."""
    transitions = TrackEvents("cam0", 5.0)

    events = transitions.update([(1, "Alice", 3)], T0) + transitions.update([(1, "Alice", 9)], T0 + 1)

    assert kinds(events) == [("appeared", "1", None), ("identified", "1", "Alice")]


def test_open_track_gives_an_ongoing_session():
    from core.event_log import sessions_from_events

    transitions = TrackEvents("cam0", 5.0, face_max_age=300.0)
    events = transitions.update([(1, "Alice", 1)], T0) + transitions.update([(1, "Alice", 2)], T0 + 10)

    sessions = sessions_from_events(events, 60.0, 300.0, now=T0 + 20)

    assert sessions == [{"name": "Alice", "arrived": T0, "departed": None}]


def test_check_sessions_pairs_by_name_and_arrival():
    from tools.check_sessions import compare

    stored = [{"name": "Alice", "arrived": T0, "departed": T0 + 100},
              {"name": "Bob", "arrived": T0 + 5, "departed": T0 + 50}]
    rebuilt = [{"name": "Alice", "arrived": T0 + 0.2, "departed": T0 + 100.2},
               {"name": "Carol", "arrived": T0 + 5, "departed": T0 + 50}]

    pairs, only_stored, only_rebuilt = compare(stored, rebuilt)

    assert [(s["name"], r["name"]) for s, r in pairs] == [("Alice", "Alice")]
    assert [s["name"] for s in only_stored] == ["Bob"]
    assert [r["name"] for r in only_rebuilt] == ["Carol"]
