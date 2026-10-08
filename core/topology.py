"""
topology.py — Plausibilité d'une association entre caméras (roadmap 2.4).

Lit une topologie (topologies/*.yaml, tools/infer_topology.py) et répond : une
personne sortie de la caméra A peut-elle être celle qui apparaît sur B Δ
secondes plus tard (Δ négatif : champs qui se recouvrent) ? La fenêtre de
temps est une distribution, pas une contrainte :

    score(A, B, Δ) = P(B | sortie de A) × densité du délai Δ sur le lien A → B

P(B | A) : part des passages observés depuis A qui vont vers B ; densité :
noyau gaussien (BANDWIDTH_S) sur l'histogramme des temps de transit du lien.
Un lien jamais observé garde FLOOR, faible mais non nul.

Mesuré sur CHIRLA (appris en juin-juillet, jugé en décembre,
tools/eval_topology.py) : ce score sépare les vrais passages des autres
paires candidates avec une AUC de 0,961, contre 0,925 pour le seul délai.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

BANDWIDTH_S = 1.0
FLOOR = 1e-4


@dataclass(frozen=True)
class Link:
    passages: int
    centers: np.ndarray     # centres des cases de l'histogramme (s)
    counts: np.ndarray


class Topology:
    def __init__(self, links: dict, cameras: list[str], homographies: dict | None = None):
        self.links = links          # {(A, B): Link}
        self.cameras = cameras
        # {(A, B): matrice 3 × 3 sol A → sol B}, seulement les homographies
        # validées (tools/infer_topology.py, « usable »), dans les deux sens.
        self.homographies = homographies or {}
        out = {}
        for (a, _b), link in links.items():
            out[a] = out.get(a, 0) + link.passages
        self._passages_from = out

    @classmethod
    def load(cls, path: str) -> Topology:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict) -> Topology:
        links = {}
        for link in data.get("links", []):
            hist = link["transit_histogram"]
            counts = np.asarray(hist["counts"], dtype=float)
            centers = hist["start_s"] + hist["bin_s"] * (np.arange(len(counts)) + 0.5)
            links[(link["from"], link["to"])] = Link(link["passages"], centers, counts)
        homographies = {}
        for overlap in data.get("overlaps", []):
            h = overlap.get("homography")
            if h and h.get("usable"):
                a, b = overlap["cameras"]
                matrix = np.asarray(h["matrix"], dtype=float)
                homographies[(a, b)] = matrix
                homographies[(b, a)] = np.linalg.inv(matrix)
        return cls(links, sorted(data.get("cameras", {})), homographies)

    def project(self, from_camera: str, to_camera: str, point) -> tuple[float, float] | None:
        """Pied vu sur `from_camera` ramené dans l'image de `to_camera`, si leurs
        champs se recouvrent et que l'homographie de sol a été validée."""
        matrix = self.homographies.get((from_camera, to_camera))
        if matrix is None:
            return None
        x, y, w = matrix @ np.array([point[0], point[1], 1.0])
        return (x / w, y / w)

    def score(self, from_camera: str, to_camera: str, delta_s: float) -> float:
        """Vraisemblance qu'une sortie de `from_camera` soit l'entrée sur
        `to_camera` `delta_s` secondes plus tard."""
        link = self.links.get((from_camera, to_camera))
        if link is None or link.counts.sum() == 0:
            return FLOOR
        prior = link.passages / self._passages_from[from_camera]
        weights = np.exp(-0.5 * ((delta_s - link.centers) / BANDWIDTH_S) ** 2)
        density = float((weights * link.counts).sum()
                        / (link.counts.sum() * BANDWIDTH_S * np.sqrt(2 * np.pi)))
        return prior * density + FLOOR

    def plausible_sources(self, to_camera: str, delta_by_camera: dict[str, float],
                          threshold: float) -> list[tuple[str, float]]:
        """Caméras de sortie plausibles pour une entrée sur `to_camera` ;
        `delta_by_camera` : temps écoulé depuis la dernière sortie vue sur
        chacune. Triées par score décroissant."""
        scored = [(c, self.score(c, to_camera, d)) for c, d in delta_by_camera.items()]
        return sorted(((c, s) for c, s in scored if s >= threshold), key=lambda cs: -cs[1])
