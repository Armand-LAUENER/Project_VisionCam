"""
adaptive.py — Réglages adaptés en marche au nombre de caméras actives (roadmap 2.7).

Une machine donnée traite un nombre borné d'images par seconde : ~72 par
processus (GIL), moins quand un processus porte beaucoup de caméras, et le GPU
au-delà de ~150 au total (docs/performance.md, test de charge). Avec peu de
caméras, il reste de la marge : la reconnaissance faciale peut passer plus
souvent (toutes les 2 images : mauvais noms 0,7 → 0,4 %). Avec beaucoup, il
faut alléger : reconnaissance toutes les 10 images (+11 % de débit), puis
baisser la cadence par caméra pour que toutes tiennent.

`plan()` est une fonction pure : caméras actives par processus → réglages par
caméra. `AdaptiveController` la rappelle quand l'ensemble des caméras actives
change (une caméra IP qui tombe ou revient) et pousse les nouveaux réglages.

Les capacités par défaut viennent des mesures sur RTX 4060 sous WSL2 :
ADAPTIVE_PROCESS_FPS et ADAPTIVE_GPU_FPS les remplacent sur une autre machine.
Un réglage fixé dans l'environnement (CAMERA_MAX_FPS, FACE_RECOGNITION_SKIP)
n'est jamais adapté.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)

SOURCE_FPS = 30
# Débit relatif selon la cadence de reconnaissance, mesuré à 8 × 30 i/s
# (docs/performance.md, balayage) : 17,2 / 19,0 / 21,1 i/s par caméra.
RECOGNITION_THROUGHPUT = {2: 0.91, 5: 1.0, 10: 1.11}
# Part de la capacité visée : au-delà, la latence p95 s'envole.
MARGIN = 0.9


@dataclass
class Tuning:
    """Réglages ajustables en marche d'une caméra.

    max_fps : cadence traitée au plus (0 : toutes les images de la source).
    recognition_skip : reconnaissance faciale toutes les N images.
    """
    max_fps: float
    recognition_skip: int


def process_capacity(cameras: int, per_process_fps: float) -> float:
    """Images par seconde que tient un processus portant `cameras` caméras.

    Jusqu'à 4 caméras, le plafond du GIL (~72 i/s) ; au-delà, chaque thread de
    plus en attente du GIL le fait baisser (8 caméras : ~50 i/s mesurés).
    """
    if cameras <= 4:
        return per_process_fps
    return max(per_process_fps / 2, per_process_fps - 5.5 * (cameras - 4))


def plan(active_per_process: list[int], per_process_fps: float, gpu_fps: float) -> list[Tuning]:
    """Réglages de chaque processus selon son nombre de caméras actives.

    Par processus : toutes les images si possible, avec la reconnaissance la
    plus fréquente (toutes les 2, 5 puis 10 images) dont le coût mesuré tient
    dans la capacité ; si même toutes les 10 ne suffit pas, cadence réduite
    pour tenir. La capacité d'un processus, prise avec MARGIN, est bornée par
    sa part du GPU.
    """
    total = sum(active_per_process)
    tunings = []
    for active in active_per_process:
        if active == 0:
            tunings.append(Tuning(0, config.FACE_RECOGNITION_SKIP))
            continue
        gpu_share = gpu_fps * active / total
        capacity = MARGIN * min(process_capacity(active, per_process_fps), gpu_share)
        demand = SOURCE_FPS * active
        skip = next((s for s in sorted(RECOGNITION_THROUGHPUT)
                     if demand <= capacity * RECOGNITION_THROUGHPUT[s]), None)
        if skip is not None:
            tunings.append(Tuning(0, skip))
        else:
            fps = max(1, math.floor(capacity * RECOGNITION_THROUGHPUT[10] / active))
            tunings.append(Tuning(fps, 10))
    return tunings


def with_overrides(tuning: Tuning) -> Tuning:
    """Un réglage fixé dans l'environnement prime sur l'adaptation."""
    return Tuning(
        config.CAMERA_MAX_FPS if config.CAMERA_MAX_FPS_FIXED else tuning.max_fps,
        config.FACE_RECOGNITION_SKIP if config.FACE_RECOGNITION_SKIP_FIXED
        else tuning.recognition_skip,
    )


def update_in_place(target: Tuning, tuning: Tuning) -> None:
    """Copie un réglage dans l'objet que capture et pipeline lisent déjà."""
    target.max_fps = tuning.max_fps
    target.recognition_skip = tuning.recognition_skip


class AdaptiveController:
    """Recalcule les réglages quand les caméras actives changent, ou quand ils
    ne sont pas tenus.

    `groups` : les caméras de chaque processus (un seul groupe en mode thread).
    Une caméra est active si elle a produit une image depuis moins de
    ACTIVE_S secondes (`camera.last_result`, horloge monotone). `apply(camera,
    tuning)` applique un réglage à la caméra (et met à jour `camera.tuning`) ;
    il n'est appelé que s'il change.

    Boucle de retour : la capacité réelle varie d'une machine et d'un moment à
    l'autre (16 caméras : 12,6 puis 11,4 i/s tenus pour 13 visés). Si un groupe
    reste sous HELD_RATIO de sa cadence visée (`measured_fps(camera)`) pendant
    SHORTFALL_STEPS tours, les capacités sont réduites d'après l'écart observé.
    Pas d'évaluation pendant GRACE_S après un changement (chauffe des
    processus, mesure des i/s sur une seconde) ; pas de remontée en marche,
    pour ne pas osciller.
    """

    ACTIVE_S = 5.0
    PERIOD_S = 2.0
    GRACE_S = 45.0
    HELD_RATIO = 0.93
    SHORTFALL_STEPS = 3
    MIN_SCALE = 0.5

    def __init__(self, groups, apply, clock=time.monotonic, measured_fps=None):
        self.groups = groups
        self._apply = apply
        self._clock = clock
        self._measured_fps = measured_fps or (lambda camera: camera.state.fps)
        self._active: tuple | None = None
        self._changed_at = None
        self._shortfall = 0
        self.scale = 1.0

    def _active_ids(self, now) -> tuple:
        return tuple(tuple(c.id for c in group
                           if c.last_result is not None and now - c.last_result <= self.ACTIVE_S)
                     for group in self.groups)

    def _held_ratio(self, active) -> float:
        """Plus faible rapport (i/s tenus / visés) parmi les groupes actifs."""
        ratios = []
        for group, ids in zip(self.groups, active):
            cams = [c for c in group if c.id in ids]
            if cams:
                target = sum(c.tuning.max_fps or SOURCE_FPS for c in cams)
                ratios.append(sum(self._measured_fps(c) for c in cams) / target)
        return min(ratios, default=1.0)

    def step(self) -> bool:
        """Un tour : True si les réglages ont été recalculés."""
        now = self._clock()
        active = self._active_ids(now)
        reason = None
        if active != self._active:
            reason = "caméras actives"
        elif now - self._changed_at >= self.GRACE_S:
            ratio = self._held_ratio(active)
            self._shortfall = self._shortfall + 1 if ratio < self.HELD_RATIO else 0
            if self._shortfall >= self.SHORTFALL_STEPS and self.scale > self.MIN_SCALE:
                self.scale = max(self.MIN_SCALE, self.scale * ratio)
                reason = f"cadence non tenue ({ratio:.0%}), capacité × {self.scale:.2f}"
        if reason is None:
            return False
        self._active, self._changed_at, self._shortfall = active, now, 0
        tunings = [with_overrides(t) for t in
                   plan([len(ids) for ids in active], config.ADAPTIVE_PROCESS_FPS * self.scale,
                        config.ADAPTIVE_GPU_FPS * self.scale)]
        for group, ids, tuning in zip(self.groups, active, tunings):
            for camera in group:
                if camera.id in ids and camera.tuning != tuning:
                    self._apply(camera, tuning)
        logger.info("Réglages adaptés (%s) : %d caméra(s) active(s) — %s", reason,
                    sum(map(len, active)),
                    "; ".join(f"{len(ids)} caméra(s) : "
                              f"{f'{t.max_fps:g} i/s' if t.max_fps else 'toutes les images'}, "
                              f"reconnaissance toutes les {t.recognition_skip}"
                              for ids, t in zip(active, tunings) if ids) or "aucune")
        return True

    def run(self, running) -> None:
        while running():
            try:
                self.step()
            except Exception as e:
                # Un tour raté ne doit pas arrêter l'adaptation.
                logger.warning("Adaptation impossible : %s: %s", type(e).__name__, e)
            time.sleep(self.PERIOD_S)

    def start(self, running) -> threading.Thread:
        thread = threading.Thread(target=self.run, args=(running,), daemon=True,
                                  name="adaptive")
        thread.start()
        return thread
