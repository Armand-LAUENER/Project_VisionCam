"""
tracker_backends.py — Deux implémentations interchangeables de l'association
de tracks pour le pipeline Body-First.

`FaceBodyTracker` n'utilise du tracker que trois choses : `track_id`,
`is_confirmed()` et `to_ltrb()`. Les deux backends ci-dessous exposent
exactement cette surface, ce qui permet de basculer de l'un à l'autre via
`config.TRACKER_BACKEND` sans toucher au reste du pipeline.

- "python" : deep_sort_realtime 1.3.2.
- "rust"   : crate deepsort-rs (PyO3).

Les deux ne font que l'association : les embeddings d'apparence sont calculés
ici, par le même embedder (`core.appearance_embedder`) et sur les mêmes crops
(`DeepSort.crop_bb`), pour que comparer les deux backends mesure bien le
tracking seul.

Deux écarts observables, sans conséquence sur le pipeline :

`track_id` est une chaîne côté deep_sort_realtime ("1") et un entier côté Rust
(1). Il ne sert que de clé de dictionnaire interne et d'affichage, jamais de
comparaison entre backends — les deux restent cohérents dans une même session.

Écart numérique : l'embedder tourne en demi-précision sur
GPU et deep_sort_realtime garde ses vecteurs en float16, alors que le backend
Rust les convertit en float32 avant de traverser la frontière. Les distances
cosinus diffèrent donc de l'ordre de 1e-3, ce qui peut changer une décision de
gating strictement à la limite du seuil. Aucune divergence d'algorithme.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
from core.appearance_embedder import EMBED_DIM, build_embedder

logger = logging.getLogger(__name__)


class TrackLike(Protocol):
    """Surface du tracker réellement consommée par `FaceBodyTracker`."""

    track_id: int

    def is_confirmed(self) -> bool: ...

    def to_ltrb(self): ...


class BodyTrackerBackend(Protocol):
    def update(self, detections: list[tuple], frame: np.ndarray) -> list[TrackLike]: ...


def _check_embedder() -> None:
    # Vérifié avant de charger le modèle : une configuration invalide doit
    # échouer tout de suite, sans payer le chargement de torch.
    if config.DEEPSORT_EMBEDDER != "mobilenet":
        raise ValueError(
            f"Seul DEEPSORT_EMBEDDER='mobilenet' est géré, pas '{config.DEEPSORT_EMBEDDER}'."
        )


def _embed(embedder, detections: list[tuple], frame: np.ndarray):
    """Détections exploitables et leurs embeddings, dans le même ordre.

    Une boîte de largeur ou hauteur nulle ferait planter le resize : elle est
    écartée comme le fait deep_sort_realtime. `crop_bb` est réutilisé tel quel
    pour que les crops soient bit à bit ceux de la référence.
    """
    detections = [d for d in detections if d[0][2] > 0 and d[0][3] > 0]
    if not detections:
        return detections, np.zeros((0, EMBED_DIM), dtype=np.float32)
    crops, _ = DeepSort.crop_bb(frame, detections)
    return detections, embedder.predict(crops)


class PythonDeepSortBackend:
    """deep_sort_realtime, avec les embeddings calculés en amont."""

    name = "python"

    def __init__(self) -> None:
        _check_embedder()
        self._embedder = build_embedder()
        self._tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            nn_budget=config.DEEPSORT_NN_BUDGET,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
            max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
            embedder=None,
        )

    def update(self, detections: list[tuple], frame: np.ndarray) -> list[TrackLike]:
        detections, embeddings = _embed(self._embedder, detections, frame)
        return self._tracker.update_tracks(detections, embeds=list(embeddings))


class RustDeepSortBackend:
    """Association en Rust (deepsort-rs), embeddings inchangés côté Python.

    Les paramètres passés au crate reproduisent ceux de `PythonDeepSortBackend`,
    y compris les défauts implicites de deep_sort_realtime que le crate ne
    partage pas. Ils sont donc tous explicites des deux côtés — c'est ce qui
    rend les deux backends comparables ligne à ligne.
    """

    name = "rust"

    def __init__(self) -> None:
        _check_embedder()

        import deepsort_rs

        self._embedder = build_embedder()
        self._tracker = deepsort_rs.Tracker(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
            nn_budget=config.DEEPSORT_NN_BUDGET,
            max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
        )

    def update(self, detections: list[tuple], frame: np.ndarray) -> list[TrackLike]:
        detections, embeddings = _embed(self._embedder, detections, frame)
        # Même sans détection, le tracker est appelé : c'est ce qui fait
        # avancer les prédictions de Kalman et vieillir les pistes.
        boxes = np.array(
            [[x, y, x + w, y + h] for (x, y, w, h), *_ in detections],
            dtype=np.float32,
        ).reshape(-1, 4)
        confidences = [d[1] for d in detections]
        return self._tracker.update(boxes, confidences,
                                    np.asarray(embeddings, dtype=np.float32))


def build_body_tracker() -> BodyTrackerBackend:
    """Instancie le backend désigné par `config.TRACKER_BACKEND`."""
    backend = config.TRACKER_BACKEND
    if backend == "python":
        logger.info("Tracker de corps : deep_sort_realtime (backend python)")
        return PythonDeepSortBackend()
    if backend == "rust":
        logger.info("Tracker de corps : deepsort-rs (backend rust)")
        return RustDeepSortBackend()
    raise ValueError(
        f"TRACKER_BACKEND='{backend}' inconnu — valeurs acceptées : 'python', 'rust'."
    )
