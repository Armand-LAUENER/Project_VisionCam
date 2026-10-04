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
import threading
import time
from datetime import datetime

from core.db_writer import DbWriter

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
                 purge_interval: float = 3600.0, writer: DbWriter | None = None) -> None:
        self.db_path = db_path
        self.gap = gap
        self.flush_interval = flush_interval
        self.retention = retention
        self.purge_interval = purge_interval
        self._clock = clock
        self._last_purge: float | None = None
        self._lock = threading.Lock()
        self._ready = False
        # Écritures déléguées (core/db_writer.py) : en arrière-plan dans
        # l'application, pour ne jamais faire attendre la boucle vidéo ; tout
        # de suite dans le thread appelant sinon (tests, outils).
        self._writer = writer or DbWriter(db_path)
        # { nom: [clé de session, dernière vue, dernière vue écrite] }. La clé
        # est attribuée ici ; le numéro SQLite, connu seulement à l'écriture,
        # est retrouvé par le thread d'écriture dans _rowids.
        self._open: dict[str, list] = {}
        self._next_key = 0
        self._rowids: dict[int, int] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Écriture (sous _lock ; les requêtes partent au thread d'écriture)
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        """Crée la base au premier usage : importer l'application ne touche pas le disque."""
        if self._ready:
            return

        def init(conn):
            conn.executescript(_SCHEMA)
            # Sessions laissées ouvertes par un arrêt brutal.
            conn.execute("UPDATE sessions SET departed = last_seen WHERE departed IS NULL")

        self._writer.submit(init)
        self._ready = True

    def _insert(self, key, name, now):
        def run(conn):
            cursor = conn.execute(
                "INSERT INTO sessions (name, arrived, last_seen) VALUES (?, ?, ?)",
                (name, now, now))
            self._rowids[key] = cursor.lastrowid
        return run

    def _set_last_seen(self, key, last_seen, departed: bool):
        def run(conn):
            rowid = self._rowids.get(key)
            if rowid is None:       # session effacée entre-temps (forget)
                return
            if departed:
                conn.execute("UPDATE sessions SET last_seen = ?, departed = ? WHERE id = ?",
                             (last_seen, last_seen, rowid))
                del self._rowids[key]
            else:
                conn.execute("UPDATE sessions SET last_seen = ? WHERE id = ?", (last_seen, rowid))
        return run

    def update(self, names, now: float | None = None) -> list[dict]:
        """Enregistre les noms vus à cet instant ; retourne arrivées et départs."""
        now = self._clock() if now is None else now
        seen = {n for n in names if n and n != UNKNOWN}
        events, tasks = [], []
        with self._lock:
            self._ensure_ready()
            if self.retention and (self._last_purge is None
                                   or now - self._last_purge >= self.purge_interval):
                self._last_purge = now
                tasks.append(self._purge_task(now))
            for name in sorted(seen):
                session = self._open.get(name)
                if session is None:
                    key, self._next_key = self._next_key, self._next_key + 1
                    tasks.append(self._insert(key, name, now))
                    self._open[name] = [key, now, now]
                    events.append({"type": "arrival", "name": name, "time": _iso(now)})
                else:
                    session[1] = now
                    if now - session[2] >= self.flush_interval:
                        tasks.append(self._set_last_seen(session[0], now, departed=False))
                        session[2] = now
            for name, (key, last_seen, _) in list(self._open.items()):
                if name not in seen and now - last_seen > self.gap:
                    tasks.append(self._set_last_seen(key, last_seen, departed=True))
                    del self._open[name]
                    events.append({"type": "departure", "name": name,
                                   "time": _iso(last_seen)})
            if tasks:
                self._writer.submit(lambda conn: [task(conn) for task in tasks])
        return events

    def close_all(self) -> None:
        """Ferme les sessions ouvertes à leur dernière heure vue (arrêt propre)."""
        with self._lock:
            tasks = [self._set_last_seen(key, last_seen, departed=True)
                     for key, last_seen, _ in self._open.values()]
            self._open.clear()
            if tasks:
                self._writer.submit(lambda conn: [task(conn) for task in tasks])
        self._writer.flush()

    def rename(self, old: str, new: str) -> None:
        with self._lock:
            self._ensure_ready()
            if old in self._open:
                self._open[new] = self._open.pop(old)
            self._writer.call(lambda conn: conn.execute(
                "UPDATE sessions SET name = ? WHERE name = ?", (new, old)))

    def forget(self, name: str) -> int:
        """Efface tout l'historique d'une personne (droit à l'effacement), avant de rendre la main."""
        with self._lock:
            self._ensure_ready()
            session = self._open.pop(name, None)

            def run(conn):
                if session is not None:
                    self._rowids.pop(session[0], None)
                return conn.execute("DELETE FROM sessions WHERE name = ?", (name,)).rowcount

            return self._writer.call(run)

    def _purge_task(self, now):
        cutoff = now - self.retention

        def run(conn):
            return conn.execute(
                "DELETE FROM sessions WHERE departed IS NOT NULL AND departed < ?",
                (cutoff,)).rowcount
        return run

    def purge(self, now: float | None = None) -> int:
        """Supprime les sessions terminées depuis plus de `retention` secondes.

        Une session en cours n'est jamais supprimée, même commencée avant la limite.
        """
        now = self._clock() if now is None else now
        with self._lock:
            self._last_purge = now
            if not self.retention:
                return 0
            self._ensure_ready()
            return self._writer.call(self._purge_task(now))

    # ─────────────────────────────────────────────────────────────────────
    # Lecture (après les écritures en attente)
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
            self._ensure_ready()
            live_by_key = {key: last for key, last, _ in self._open.values()}

            def read(conn):
                rows = conn.execute(
                    f"SELECT id, name, arrived, last_seen, departed FROM sessions {where} "
                    "ORDER BY arrived DESC LIMIT ?", (*params, limit)).fetchall()
                live = {self._rowids[k]: last for k, last in live_by_key.items()
                        if k in self._rowids}
                return rows, live

            rows, live = self._writer.call(read)
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
            self._ensure_ready()
            rows = self._writer.call(lambda conn: conn.execute(
                "SELECT DISTINCT name FROM sessions ORDER BY name").fetchall())
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


class FaceConfirmedNames:
    """Noms dont le visage a été reconnu depuis moins de `max_age` secondes.

    Une piste garde son nom quand la personne est de dos : c'est voulu à
    l'écran, mais après un échange de pistes vers une personne non enrôlée,
    l'historique inscrirait la mauvaise personne tant qu'elle reste visible.
    La marge, bien plus longue que l'affichage (FACE_FRESHNESS_FRAMES), évite
    de couper la session d'une personne enrôlée assise de dos quelques minutes.

    Une reconnaissance se voit au changement de `last_face_frame` de la piste.
    Une piste en roue libre qui revient garde la date de sa dernière
    reconnaissance : sa réapparition n'en est pas une.
    """

    def __init__(self, max_age: float) -> None:
        self.max_age = max_age
        # { track_id: (nom, last_face_frame, heure de la reconnaissance) }
        self._recognized: dict[str, tuple[str, int, float]] = {}

    def update(self, persons, now: float) -> set[str]:
        """persons : [(track_id, nom, last_face_frame)] visibles ; retourne les noms confirmés."""
        names = set()
        visible = set()
        for track_id, name, last_face_frame in persons:
            track_id = str(track_id)
            visible.add(track_id)
            if name == UNKNOWN or last_face_frame < 0:
                self._recognized.pop(track_id, None)
                continue
            entry = self._recognized.get(track_id)
            if entry is None or entry[:2] != (name, last_face_frame):
                entry = (name, last_face_frame, now)
                self._recognized[track_id] = entry
            if now - entry[2] <= self.max_age:
                names.add(name)
        # Les pistes disparues sont oubliées une fois leur marge écoulée.
        for track_id, entry in list(self._recognized.items()):
            if track_id not in visible and now - entry[2] > self.max_age:
                del self._recognized[track_id]
        return names
