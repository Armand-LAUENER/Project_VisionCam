"""
event_log.py — Journal d'événements bruts des pistes, en ajout seul (SQLite).

Une ligne par transition d'une piste visible, et non par image :

    appeared      la piste devient visible
    identified    elle reçoit un nom (ou en change : le nouveau nom)
    unidentified  elle redevient « Inconnu »
    face          son visage a été reconnu (piste nommée) : au plus un par
                  `face_interval` s, plus le dernier avant que la marge
                  `face_max_age` ne s'écoule, que le nom tombe ou que la piste
                  finisse ; c'est ce qui rend les sessions reconstruites exactes
    ended         elle n'est plus visible depuis `end_after_s` ; l'heure est
                  celle de la dernière image où elle l'était

Colonnes : time, run_id, camera_id, track_id, type, name, zone. Les numéros
de piste repartent de 1 à chaque lancement : une piste s'identifie par
(run_id, camera_id, track_id). `zone` est réservée aux entrées et sorties de
zone (roadmap 2.4).

L'heure vient de ForwardClock : l'heure murale lue une fois au démarrage, puis
avancée par l'horloge monotone. Sous WSL2, l'horloge murale recule d'environ
0,8 s toutes les 30 s : des événements pris sur elle se retrouveraient dans le
désordre.

Sessions de présence : sessions_from_events() les recalcule à partir des
événements, avec la règle de l'historique (core/presence_log.py) : un nom est
présent tant qu'une piste visible le porte et que son visage a été reconnu
depuis moins de `face_max_age`, et une session se ferme après `gap` sans
présence.

RGPD : les événements suivent la rétention de l'historique ; effacer une
personne efface tous les événements de ses pistes, y compris ceux d'avant
son identification ; un renommage les suit.

Même fichier que l'historique de présence (core/presence_log.py), table à part.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import defaultdict

UNKNOWN = "Inconnu"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    time      REAL    NOT NULL,
    run_id    TEXT    NOT NULL,
    camera_id TEXT    NOT NULL,
    track_id  TEXT    NOT NULL,
    type      TEXT    NOT NULL,
    name      TEXT,
    zone      TEXT
);
CREATE INDEX IF NOT EXISTS events_time ON events (time);
CREATE INDEX IF NOT EXISTS events_track ON events (run_id, camera_id, track_id);
CREATE INDEX IF NOT EXISTS events_name ON events (name);
"""


class ForwardClock:
    """Heure Unix qui ne recule jamais : heure murale au départ, puis horloge monotone."""

    def __init__(self, wall=time.time, monotonic=time.monotonic) -> None:
        self._monotonic = monotonic
        self._base = wall() - monotonic()

    def __call__(self) -> float:
        return self._base + self._monotonic()


class TrackEvents:
    """Transforme les pistes visibles de chaque image en événements de transition."""

    def __init__(self, camera_id: str, end_after_s: float,
                 face_max_age: float | None = None, face_interval: float = 60.0) -> None:
        self.camera_id = camera_id
        self.end_after_s = end_after_s
        self.face_max_age = face_max_age      # None : pas d'événement `face`
        self.face_interval = face_interval
        # { track_id: _Track }
        self._tracks: dict[str, _Track] = {}

    def _event(self, when, track_id, kind, name=None) -> dict:
        return {"time": when, "camera_id": self.camera_id, "track_id": track_id,
                "type": kind, "name": name, "zone": None}

    def update(self, persons, now: float) -> list[dict]:
        """persons : [(track_id, nom[, last_face_frame])] visibles sur cette image.

        Une reconnaissance du visage se voit au changement de last_face_frame,
        comme dans FaceConfirmedNames.
        """
        events = []
        visible = set()
        for track_id, name, *rest in persons:
            track_id = str(track_id)
            visible.add(track_id)
            name = None if name == UNKNOWN else name
            last_face_frame = rest[0] if rest else None
            track = self._tracks.get(track_id)
            if track is None:
                track = self._tracks[track_id] = _Track(now)
                events.append(self._event(now, track_id, "appeared"))
            if name != track.name:
                events += self._flush_face(track_id, track)
                events.append(self._event(now, track_id,
                                          "unidentified" if name is None else "identified", name))
                track.name = name
                track.last_face_frame = last_face_frame
                # Une identification vient d'une reconnaissance : elle compte comme visage.
                track.last_face_event = now if name else None
            elif (name and self.face_max_age is not None and last_face_frame is not None
                  and last_face_frame >= 0 and last_face_frame != track.last_face_frame):
                track.last_face_frame = last_face_frame
                track.pending_face = now
            track.last_seen = now

        for track_id, track in list(self._tracks.items()):
            if track_id not in visible and now - track.last_seen > self.end_after_s:
                events += self._flush_face(track_id, track)
                events.append(self._event(track.last_seen, track_id, "ended"))
                del self._tracks[track_id]
            elif track.pending_face is not None and (
                    track.pending_face - track.last_face_event >= self.face_interval
                    or now - track.pending_face >= self.face_max_age):
                events += self._flush_face(track_id, track)
        return events

    def _flush_face(self, track_id, track) -> list[dict]:
        """Écrit la dernière reconnaissance pas encore journalisée, à son heure."""
        if self.face_max_age is None or track.pending_face is None:
            return []
        event = self._event(track.pending_face, track_id, "face", track.name)
        track.last_face_event, track.pending_face = track.pending_face, None
        return [event]

    def finish(self) -> list[dict]:
        """Termine toutes les pistes ouvertes (arrêt de l'application)."""
        events = []
        for track_id, track in self._tracks.items():
            events += self._flush_face(track_id, track)
            events.append(self._event(track.last_seen, track_id, "ended"))
        self._tracks.clear()
        return events


class _Track:
    __slots__ = ("name", "last_seen", "last_face_frame", "pending_face", "last_face_event")

    def __init__(self, now: float) -> None:
        self.name: str | None = None
        self.last_seen = now
        self.last_face_frame = None
        self.pending_face: float | None = None     # reconnaissance pas encore journalisée
        self.last_face_event: float | None = None  # heure de la dernière journalisée


def sessions_from_events(events, gap: float, face_max_age: float,
                         now: float | None = None) -> list[dict]:
    """Sessions de présence {name, arrived, departed} recalculées à partir des événements.

    Une piste encore ouverte (sans `ended`) compte jusqu'à `now` (par défaut,
    le dernier événement) ; ses sessions ont departed=None.
    """
    by_track = defaultdict(list)
    for e in sorted(events, key=lambda e: (e["time"], e.get("id", 0))):
        by_track[(e.get("run_id"), e["camera_id"], e["track_id"])].append(e)
    if now is None:
        now = max((e["time"] for e in events), default=0.0)

    intervals = defaultdict(list)   # { nom: [(début, fin, ouverte)] }
    for track_events in by_track.values():
        name, faces = None, []

        def close(end, open_=False):
            for face in faces:
                intervals[name].append((face, min(face + face_max_age, end), open_))

        for e in track_events:
            if e["type"] in ("identified", "unidentified", "ended"):
                if name:
                    close(e["time"])
                name = e["name"] if e["type"] == "identified" else None
                faces = [e["time"]] if name else []
            elif e["type"] == "face" and name:
                faces.append(e["time"])
        if name and track_events[-1]["type"] != "ended":
            close(now, open_=True)

    sessions = []
    for name, spans in intervals.items():
        spans.sort()
        start, end, open_ = spans[0]
        for s, e, o in spans[1:]:
            if s - end <= gap:
                if e >= end:
                    end, open_ = e, o
            else:
                sessions.append({"name": name, "arrived": start, "departed": None if open_ else end})
                start, end, open_ = s, e, o
        sessions.append({"name": name, "arrived": start, "departed": None if open_ else end})
    return sorted(sessions, key=lambda s: (s["arrived"], s["name"]))


class EventLog:
    def __init__(self, db_path: str, run_id: str, clock=time.time,
                 retention: float | None = None, purge_interval: float = 3600.0) -> None:
        self.db_path = db_path
        self.run_id = run_id
        self.retention = retention
        self.purge_interval = purge_interval
        self._clock = clock
        self._last_purge: float | None = None
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        """Connexion courte, ouverte à chaque opération (appelée sous _lock)."""
        if not self._ready:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if not self._ready:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            self._ready = True
        return conn

    def write(self, events: list[dict]) -> None:
        """Ajoute les événements ; rien n'est écrit (ni la base créée) sans événement."""
        if not events:
            return
        now = self._clock()
        if self.retention and (self._last_purge is None
                               or now - self._last_purge >= self.purge_interval):
            self.purge(now)
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "INSERT INTO events (time, run_id, camera_id, track_id, type, name, zone) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(e["time"], self.run_id, e["camera_id"], e["track_id"], e["type"],
                      e.get("name"), e.get("zone")) for e in events])
                conn.commit()
            finally:
                conn.close()

    def events(self) -> list[dict]:
        """Tous les événements, dans l'ordre d'écriture."""
        with self._lock:
            conn = self._connect()
            try:
                return [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY id")]
            finally:
                conn.close()

    def forget(self, name: str) -> int:
        """Efface tous les événements des pistes qui ont porté ce nom ; retourne leur nombre."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM events WHERE (run_id, camera_id, track_id) IN "
                    "(SELECT run_id, camera_id, track_id FROM events WHERE name = ?)", (name,))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    def rename(self, old: str, new: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("UPDATE events SET name = ? WHERE name = ?", (new, old))
                conn.commit()
            finally:
                conn.close()

    def purge(self, now: float | None = None) -> int:
        """Supprime les événements plus anciens que la rétention ; retourne leur nombre."""
        now = self._clock() if now is None else now
        self._last_purge = now
        if not self.retention:
            return 0
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute("DELETE FROM events WHERE time < ?", (now - self.retention,))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
