"""
presence_log.py — Historique des présences, par personne, dans SQLite.

Une session s'ouvre à la première apparition d'un nom et se ferme quand la
personne n'a plus été vue depuis `gap` secondes : sortir quelques instants du
champ ne coupe pas la session. Les personnes « Inconnu » ne sont pas
enregistrées.

Écritures limitées : une à l'arrivée, une au départ, et la dernière heure vue
au plus toutes les `flush_interval` secondes. Au démarrage, les sessions
restées ouvertes par un arrêt brutal sont fermées à leur dernière heure vue.

Conservation limitée : avec `retention` (secondes), les sessions terminées
depuis plus longtemps sont supprimées au premier passage, puis au plus une
fois par `purge_interval`.

Les horodatages sont stockés en secondes epoch et rendus en heure locale.
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import threading
import time
from datetime import datetime

UNKNOWN = "Inconnu"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL,
    arrived   REAL    NOT NULL,
    last_seen REAL    NOT NULL,
    departed  REAL
);
CREATE INDEX IF NOT EXISTS sessions_name_arrived ON sessions (name, arrived);
CREATE INDEX IF NOT EXISTS sessions_arrived ON sessions (arrived);
"""


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


class PresenceLog:
    def __init__(self, db_path: str, gap: float = 60.0, flush_interval: float = 10.0,
                 clock=time.time, retention: float | None = None,
                 purge_interval: float = 3600.0) -> None:
        self.db_path = db_path
        self.gap = gap
        self.flush_interval = flush_interval
        self.retention = retention
        self.purge_interval = purge_interval
        self._clock = clock
        self._last_purge: float | None = None
        self._lock = threading.Lock()
        self._ready = False
        # { nom: [id de session, dernière vue, dernière vue écrite] }
        self._open: dict[str, list] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Connexion
    # ─────────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Connexion courte, ouverte à chaque opération (appelée sous _lock).

        La base n'est créée qu'au premier usage : importer l'application ne
        touche pas le disque.
        """
        if not self._ready:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if not self._ready:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            # Sessions laissées ouvertes par un arrêt brutal.
            conn.execute("UPDATE sessions SET departed = last_seen WHERE departed IS NULL")
            conn.commit()
            self._ready = True
        return conn

    # ─────────────────────────────────────────────────────────────────────
    # Écriture
    # ─────────────────────────────────────────────────────────────────────

    def update(self, names, now: float | None = None) -> list[dict]:
        """Enregistre les noms vus à cet instant ; retourne arrivées et départs."""
        now = self._clock() if now is None else now
        if self.retention and (self._last_purge is None
                               or now - self._last_purge >= self.purge_interval):
            self.purge(now)
        seen = {n for n in names if n and n != UNKNOWN}
        events = []
        with self._lock:
            conn = None
            try:
                for name in sorted(seen):
                    session = self._open.get(name)
                    if session is None:
                        conn = conn or self._connect()
                        cursor = conn.execute(
                            "INSERT INTO sessions (name, arrived, last_seen) VALUES (?, ?, ?)",
                            (name, now, now))
                        self._open[name] = [cursor.lastrowid, now, now]
                        events.append({"type": "arrival", "name": name, "time": _iso(now)})
                    else:
                        session[1] = now
                        if now - session[2] >= self.flush_interval:
                            conn = conn or self._connect()
                            conn.execute("UPDATE sessions SET last_seen = ? WHERE id = ?",
                                         (now, session[0]))
                            session[2] = now
                for name, (session_id, last_seen, _) in list(self._open.items()):
                    if name not in seen and now - last_seen > self.gap:
                        conn = conn or self._connect()
                        conn.execute("UPDATE sessions SET last_seen = ?, departed = ? WHERE id = ?",
                                     (last_seen, last_seen, session_id))
                        del self._open[name]
                        events.append({"type": "departure", "name": name,
                                       "time": _iso(last_seen)})
                if conn is not None:
                    conn.commit()
            finally:
                if conn is not None:
                    conn.close()
        return events

    def close_all(self) -> None:
        """Ferme les sessions ouvertes à leur dernière heure vue (arrêt propre)."""
        with self._lock:
            if not self._open:
                return
            conn = self._connect()
            try:
                for session_id, last_seen, _ in self._open.values():
                    conn.execute("UPDATE sessions SET last_seen = ?, departed = ? WHERE id = ?",
                                 (last_seen, last_seen, session_id))
                conn.commit()
            finally:
                conn.close()
            self._open.clear()

    def rename(self, old: str, new: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("UPDATE sessions SET name = ? WHERE name = ?", (new, old))
                conn.commit()
            finally:
                conn.close()
            if old in self._open:
                self._open[new] = self._open.pop(old)

    def forget(self, name: str) -> int:
        """Efface tout l'historique d'une personne (droit à l'effacement)."""
        with self._lock:
            conn = self._connect()
            try:
                deleted = conn.execute("DELETE FROM sessions WHERE name = ?", (name,)).rowcount
                conn.commit()
            finally:
                conn.close()
            self._open.pop(name, None)
        return deleted

    def purge(self, now: float | None = None) -> int:
        """Supprime les sessions terminées depuis plus de `retention` secondes.

        Une session en cours n'est jamais supprimée, même commencée avant la limite.
        """
        now = self._clock() if now is None else now
        with self._lock:
            self._last_purge = now
            if not self.retention:
                return 0
            conn = self._connect()
            try:
                deleted = conn.execute(
                    "DELETE FROM sessions WHERE departed IS NOT NULL AND departed < ?",
                    (now - self.retention,)).rowcount
                conn.commit()
            finally:
                conn.close()
        return deleted

    # ─────────────────────────────────────────────────────────────────────
    # Lecture
    # ─────────────────────────────────────────────────────────────────────

    def sessions(self, name: str | None = None, start: float | None = None,
                 end: float | None = None, limit: int = 1000) -> list[dict]:
        """Sessions qui chevauchent [start, end[, les plus récentes d'abord.

        Une session en cours a `departed` à None ; sa durée court jusqu'à sa
        dernière heure vue.
        """
        clauses, params = [], []
        if name:
            clauses.append("name = ?")
            params.append(name)
        if start is not None:
            clauses.append("COALESCE(departed, last_seen) >= ?")
            params.append(start)
        if end is not None:
            clauses.append("arrived < ?")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT id, name, arrived, last_seen, departed FROM sessions {where} "
                    "ORDER BY arrived DESC LIMIT ?", (*params, limit)).fetchall()
            finally:
                conn.close()
            live = {session_id: last for session_id, last, _ in self._open.values()}
        result = []
        for row in rows:
            last_seen = live.get(row["id"], row["last_seen"])
            end_time = row["departed"] if row["departed"] is not None else last_seen
            result.append({
                "id": row["id"], "name": row["name"],
                "arrived": _iso(row["arrived"]), "departed": _iso(row["departed"]),
                "last_seen": _iso(last_seen), "ongoing": row["departed"] is None,
                "duration_s": round(end_time - row["arrived"]),
            })
        return result

    def names(self) -> list[str]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT DISTINCT name FROM sessions ORDER BY name").fetchall()
            finally:
                conn.close()
        return [row["name"] for row in rows]

    @staticmethod
    def to_csv(sessions: list[dict]) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["nom", "arrivée", "départ", "durée (s)", "en cours"])
        for s in sessions:
            writer.writerow([s["name"], s["arrived"], s["departed"] or "", s["duration_s"],
                             "oui" if s["ongoing"] else "non"])
        return buffer.getvalue()
