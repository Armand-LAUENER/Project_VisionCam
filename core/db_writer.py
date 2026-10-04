"""
db_writer.py — Écritures SQLite hors de la boucle vidéo, dans un thread unique.

En endurance, l'historique de présence et le journal d'événements écrivaient
dans la boucle de traitement, avec une connexion et un fsync par opération :
12 à 14 ms chacune, et jusqu'à 5,3 s quand le disque attendait (sous WSL2, un
fsync attend aussi les écritures des autres programmes du même système de
fichiers). L'image entière attendait avec lui, jusqu'à 14 s.

DbWriter garde une connexion ouverte, en WAL avec `synchronous=NORMAL` : plus
de fsync à chaque commit, pas de corruption possible, au pire les dernières
secondes perdues en cas de coupure de courant. En arrière-plan :

- submit(fn) dépose une écriture et rend la main tout de suite ;
- call(fn) attend son résultat, après toutes les écritures déjà déposées :
  les lectures voient donc tout ce qui a été écrit avant elles ;
- flush() attend que la file soit vide, close() la vide puis s'arrête.

`fn` reçoit la connexion ; le commit suit chaque tâche. Une écriture qui
échoue est signalée dans les logs et n'arrête pas les suivantes.

En mode synchrone (`background=False`, le défaut), chaque tâche s'exécute
tout de suite dans le thread appelant : le comportement des tests et des
outils ne change pas.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading

logger = logging.getLogger(__name__)

_STOP = object()


class _Future:
    def __init__(self) -> None:
        self._done = threading.Event()
        self.result = None
        self.error: BaseException | None = None

    def set(self, result=None, error=None) -> None:
        self.result, self.error = result, error
        self._done.set()

    def wait(self):
        self._done.wait()
        if self.error is not None:
            raise self.error
        return self.result


class DbWriter:
    def __init__(self, db_path: str, background: bool = False,
                 synchronous: str = "NORMAL") -> None:
        self.db_path = db_path
        self.background = background
        self.synchronous = synchronous
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()          # mode synchrone : une tâche à la fois
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    # ── Connexion (dans le thread qui exécute les tâches) ────────────────────

    def _connection(self) -> sqlite3.Connection:
        """Créée à la première tâche : construire un DbWriter ne touche pas le disque."""
        if self._conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={self.synchronous}")
            self._conn = conn
        return self._conn

    def _run(self, fn):
        conn = self._connection()
        try:
            result = fn(conn)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise

    # ── Thread d'arrière-plan ────────────────────────────────────────────────

    def _ensure_thread(self) -> None:
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="db-writer", daemon=True)
                self._thread.start()

    def _loop(self) -> None:
        while True:
            fn, future = self._queue.get()
            if fn is _STOP:
                future.set()
                return
            try:
                result = self._run(fn)
            except Exception as e:
                if future is None:
                    logger.warning("Écriture SQLite échouée (%s) : %s: %s",
                                   os.path.basename(self.db_path), type(e).__name__, e)
                else:
                    future.set(error=e)
                continue
            if future is not None:
                future.set(result)

    # ── Interface ────────────────────────────────────────────────────────────

    def submit(self, fn) -> None:
        """Écriture sans attendre le résultat (erreurs : logs seulement)."""
        if self.background:
            self._ensure_thread()
            self._queue.put((fn, None))
            return
        with self._lock:
            try:
                self._run(fn)
            except Exception as e:
                logger.warning("Écriture SQLite échouée (%s) : %s: %s",
                               os.path.basename(self.db_path), type(e).__name__, e)

    def call(self, fn):
        """Exécute `fn` après les écritures déjà déposées et retourne son résultat."""
        if self.background:
            self._ensure_thread()
            future = _Future()
            self._queue.put((fn, future))
            return future.wait()
        with self._lock:
            return self._run(fn)

    def flush(self) -> None:
        """Attend que toutes les écritures déposées soient faites."""
        if self.background and self._thread is not None:
            self.call(lambda conn: None)

    def close(self) -> None:
        """Vide la file, arrête le thread et ferme la connexion."""
        if self.background and self._thread is not None and self._thread.is_alive():
            future = _Future()
            self._queue.put((_STOP, future))
            future.wait()
            self._thread.join(timeout=5)
            self._thread = None   # une tâche suivante relancerait le thread
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
