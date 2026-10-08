"""
clothing.py — Couleur des vêtements, haut et bas du corps (roadmap 2.5).

Descripteur sans modèle : histogramme teinte × saturation (HSV) de deux
bandes du corps, normalisé, pour reconnaître une personne d'une caméra à
l'autre ou après une absence plus longue que `max_age`. Les bandes viennent
des keypoints épaules et hanches quand on les a (COCO 5-6, 11-12), sinon de
proportions fixes de la boîte (UPPER, LOWER, en part de la hauteur).

Le teint et le fond pèsent peu : la bande haute commence sous les épaules,
la bande basse s'arrête avant les pieds, et chaque bande est rognée de
MARGIN sur les côtés de la boîte.
"""

from __future__ import annotations

import cv2
import numpy as np

H_BINS, S_BINS = 12, 4
UPPER = (0.20, 0.50)      # haut du corps, en part de la hauteur de la boîte
LOWER = (0.55, 0.90)
MARGIN = 0.2
MIN_PIXELS = 50


def _band_histogram(hsv: np.ndarray) -> np.ndarray | None:
    if hsv.size == 0 or hsv.shape[0] * hsv.shape[1] < MIN_PIXELS:
        return None
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256]).ravel()
    total = hist.sum()
    return hist / total if total else None


def bands(box, keypoints=None) -> tuple[tuple[int, int], tuple[int, int]]:
    """Lignes (y) des bandes haute et basse d'une boîte (x1, y1, x2, y2).
    `keypoints` : COCO (17, 3) facultatifs ; épaules et hanches visibles les
    remplacent par celles du squelette."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    upper = (y1 + UPPER[0] * h, y1 + UPPER[1] * h)
    lower = (y1 + LOWER[0] * h, y1 + LOWER[1] * h)
    if keypoints is not None and len(keypoints) >= 13:
        shoulders = [k[1] for k in (keypoints[5], keypoints[6]) if k[2] > 0.5]
        hips = [k[1] for k in (keypoints[11], keypoints[12]) if k[2] > 0.5]
        if shoulders and hips:
            s, hp = float(np.mean(shoulders)), float(np.mean(hips))
            if hp > s:
                upper = (s + 0.1 * (hp - s), hp)
                lower = (hp + 0.1 * (y2 - hp), hp + 0.8 * (y2 - hp))
    return ((int(upper[0]), int(upper[1])), (int(lower[0]), int(lower[1])))


def descriptor(frame: np.ndarray, box, keypoints=None) -> np.ndarray | None:
    """Histogrammes haut et bas concaténés, ou None si la boîte est trop petite."""
    x1, y1, x2, y2 = (int(v) for v in box)
    width = x2 - x1
    xa, xb = max(0, int(x1 + MARGIN * width)), min(frame.shape[1], int(x2 - MARGIN * width))
    parts = []
    for top, bottom in bands((x1, y1, x2, y2), keypoints):
        top, bottom = max(0, top), min(frame.shape[0], bottom)
        if bottom <= top or xb <= xa:
            return None
        hist = _band_histogram(cv2.cvtColor(frame[top:bottom, xa:xb], cv2.COLOR_BGR2HSV))
        if hist is None:
            return None
        parts.append(hist)
    return np.concatenate(parts)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Coefficient de Bhattacharyya moyen des deux bandes, entre 0 et 1."""
    half = H_BINS * S_BINS
    return float((np.sqrt(a[:half] * b[:half]).sum() + np.sqrt(a[half:] * b[half:]).sum()) / 2)
