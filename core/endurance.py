"""
endurance.py — Journal périodique pour le test d'endurance (plusieurs heures).

Toutes les `interval_s`, une ligne CSV : heure, temps écoulé, puis les mesures
que renvoie `probe()` (mémoire, VRAM, FPS, taille des structures internes).
tools/endurance_report.py dit ensuite si elles restent stables.

Appelé depuis la boucle de traitement plutôt que par un thread à part : un
pipeline bloqué n'écrit plus rien, et le trou se voit dans le journal.

Le temps écoulé se mesure sur l'horloge monotone : sous WSL2, l'horloge murale
recule d'environ 0,8 s toutes les 30 s environ (recalage de l'heure), ce qui
faussait les intervalles. La colonne `time` garde l'heure murale, pour lire.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime


class EnduranceLog:
    def __init__(self, path: str, interval_s: float, probe, clock=time.monotonic,
                 wall_clock=time.time) -> None:
        self.path = path
        self.interval_s = interval_s
        self._probe = probe
        self._clock = clock
        self._wall_clock = wall_clock
        self._start: float | None = None
        self._next: float | None = None
        self._file = None
        self._writer = None

    def maybe_write(self) -> bool:
        """Écrit une ligne si l'intervalle est écoulé ; sinon ne mesure rien."""
        now = self._clock()
        if self._next is not None and now < self._next:
            return False
        if self._start is None:
            self._start = now
        row = {"time": datetime.fromtimestamp(self._wall_clock()).isoformat(timespec="seconds"),
               "elapsed_s": round(now - self._start, 1), **self._probe()}
        if self._writer is None:
            self._open(list(row))
        elif set(row) - set(self._writer.fieldnames):
            self._extend_header(row)
        self._writer.writerow(row)
        self._next = (self._next or now) + self.interval_s
        return True

    def _open(self, fieldnames, rows=()) -> None:
        # Ligne par ligne (buffering=1) : un arrêt brutal ne perd rien. Une
        # colonne absente d'une ligne (mesure pas encore disponible) reste vide.
        self._file = open(self.path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, restval="")
        self._writer.writeheader()
        self._writer.writerows(rows)

    def _extend_header(self, row) -> None:
        """Nouvelle colonne (compteurs Windows, étape pas encore passée) : réécrit le fichier.

        Rare (quelques fois par run) et sur quelques centaines de lignes au plus.
        """
        fieldnames = self._writer.fieldnames + [k for k in row if k not in self._writer.fieldnames]
        self._file.close()
        with open(self.path, newline="") as f:
            rows = list(csv.DictReader(f))
        self._open(fieldnames, rows)

    def close(self) -> None:
        if self._file:
            self._file.close()
