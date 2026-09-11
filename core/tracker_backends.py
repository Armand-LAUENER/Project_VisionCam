"""
tracker_backends.py — Deux implémentations interchangeables de l'association
de tracks pour le pipeline Body-First.

`FaceBodyTracker` n'utilise du tracker que trois choses : `track_id`,
`is_confirmed()` et `to_ltrb()`. Les deux backends ci-dessous exposent
exactement cette surface, ce qui permet de basculer de l'un à l'autre via
`config.TRACKER_BACKEND` sans toucher au reste du pipeline.

- "python" : deep_sort_realtime 1.3.2, qui calcule lui-même les embeddings
             d'apparence à partir de la frame.
- "rust"   : crate deepsort-rs (PyO3). Le crate ne fait que l'association ;
             les embeddings restent calculés ici, avec le même MobileNetV2 et
             les mêmes crops que la référence (`DeepSort.crop_bb`), pour que
             comparer les deux backends mesure bien le tracking seul.

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

logger = logging.getLogger(__name__)

# Dimension de sortie de MobileNetV2-bottleneck ; sert uniquement à former un
# tableau d'embeddings vide quand une frame ne contient aucune détection.
_EMBED_DIM = 1280


class TrackLike(Protocol):
    """Surface du tracker réellement consommée par `FaceBodyTracker`."""

    track_id: int

    def is_confirmed(self) -> bool: ...

    def to_ltrb(self): ...


class BodyTrackerBackend(Protocol):
    def update(self, detections: list[tuple], frame: np.ndarray) -> list[TrackLike]: ...


class PythonDeepSortBackend:
    """deep_sort_realtime tel qu'utilisé depuis le début du projet."""

    name = "python"

    def __init__(self) -> None:
        self._tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            nn_budget=config.DEEPSORT_NN_BUDGET,
            embedder=config.DEEPSORT_EMBEDDER,
            embedder_gpu=config.DEEPSORT_EMBEDDER_GPU,
        )

    def update(self, detections: list[tuple], frame: np.ndarray) -> list[TrackLike]:
        return self._tracker.update_tracks(detections, frame=frame)


class RustDeepSortBackend:
    """Association en Rust (deepsort-rs), embeddings inchangés côté Python.

    Les paramètres passés au crate reproduisent ceux de `PythonDeepSortBackend`,
    y compris les défauts implicites de deep_sort_realtime que le crate ne
    partage pas. Ils sont donc tous explicites des deux côtés — c'est ce qui
    rend les deux backends comparables ligne à ligne.
    """

    name = "rust"

    def __init__(self) -> None:
        # Vérifié avant les imports : une configuration invalide doit échouer
        # tout de suite, sans payer le chargement de torch et du modèle.
        if config.DEEPSORT_EMBEDDER != "mobilenet":
            raise ValueError(
                f"TRACKER_BACKEND=rust ne gère que DEEPSORT_EMBEDDER='mobilenet', "
                f"pas '{config.DEEPSORT_EMBEDDER}'."
            )

        import deepsort_rs
        from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

        self._embedder = MobileNetv2_Embedder(
            half=True,
            bgr=True,
            gpu=config.DEEPSORT_EMBEDDER_GPU,
        )
        self._tracker = deepsort_rs.Tracker(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_cosine_distance=0.2,
            nn_budget=config.DEEPSORT_NN_BUDGET,
            max_iou_distance=0.7,
        )

    def update(self, detections: list[tuple], frame: np.ndarray) -> list[TrackLike]:
        # Même filtre que deep_sort_realtime avant d'embarquer les crops :
        # une boîte de largeur ou hauteur nulle ferait planter le resize.
        detections = [d for d in detections if d[0][2] > 0 and d[0][3] > 0]

        if detections:
            # `crop_bb` est réutilisé tel quel pour que les crops — et donc les
            # embeddings — soient bit à bit ceux de la référence.
            crops, _ = DeepSort.crop_bb(frame, detections)
            embeddings = np.asarray(self._embedder.predict(crops), dtype=np.float32)
            boxes = np.array(
                [[x, y, x + w, y + h] for (x, y, w, h), *_ in detections],
                dtype=np.float32,
            )
            confidences = [d[1] for d in detections]
        else:
            # Une frame sans détection doit quand même faire avancer les
            # prédictions de Kalman et vieillir les pistes.
            boxes = np.zeros((0, 4), dtype=np.float32)
            embeddings = np.zeros((0, _EMBED_DIM), dtype=np.float32)
            confidences = []

        return self._tracker.update(boxes, confidences, embeddings)


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
