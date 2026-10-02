"""
event_log.py — Journal d'événements bruts des pistes, en ajout seul (SQLite).

Une ligne par transition d'une piste visible, et non par image :

    appeared      la piste devient visible
    identified    elle reçoit un nom (ou en change : le nouveau nom)
    unidentified  elle redevient « Inconnu »
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

    def __init__(self, camera_id: str, end_after_s: float) -> None:
        self.camera_id = camera_id
        self.end_after_s = end_after_s
        # { track_id: [nom ou None, heure de la dernière image visible] }
        self._tracks: dict[str, list] = {}

    def _event(self, when, track_id, kind, name=None) -> dict:
        return {"time": when, "camera_id": self.camera_id, "track_id": track_id,
                "type": kind, "name": name, "zone": None}

    def update(self, persons, now: float) -> list[dict]:
        """persons : [(track_id, nom)] visibles sur cette image."""
        events = []
        visible = set()
        for track_id, name in persons:
            track_id = str(track_id)
            visible.add(track_id)
            name = None if name == UNKNOWN else name
            entry = self._tracks.get(track_id)
            if entry is None:
                entry = self._tracks[track_id] = [None, now]
                events.append(self._event(now, track_id, "appeared"))
            if name != entry[0]:
                events.append(self._event(now, track_id,
                                          "unidentified" if name is None else "identified", name))
                entry[0] = name
            entry[1] = now
        for track_id, (_name, last) in list(self._tracks.items()):
            if track_id not in visible and now - last > self.end_after_s:
                events.append(self._event(last, track_id, "ended"))
                del self._tracks[track_id]
        return events

    def finish(self) -> list[dict]:
        """Termine toutes les pistes ouvertes (arrêt de l'application)."""
        events = [self._event(last, track_id, "ended")
                  for track_id, (_name, last) in self._tracks.items()]
        self._tracks.clear()
        return events


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
