"""
eval_topology.py — La topologie aide-t-elle à associer les passages entre caméras ?

Topologie apprise sur des séquences d'apprentissage (tools/infer_topology.py),
jugée sur d'autres : chaque paire « apparition qui se termine sur la caméra A,
apparition qui commence sur B ≠ A entre MIN_DELTA_S et MAX_DELTA_S plus
tard » est une candidate ; c'est un vrai passage si c'est la même personne et
sa prochaine apparition. On mesure à quel point le score de core/topology.py
sépare les vrais passages des autres (AUC), et ce qu'un seuil garde et écarte,
contre un score fondé sur le seul délai (plus c'est proche, plus c'est
plausible), sans topologie.

Depuis la racine du projet (réglage juin-juillet, validation décembre, comme
docs/tracking.md) :

    uv run -m tools.eval_topology --annotations ~/datasets/CHIRLA/annotations \\
        --train seq_000 seq_001 seq_002 seq_004 seq_006 seq_007 seq_020 \\
        --test seq_024 seq_025 seq_026
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from core.topology import Topology
from tools.infer_topology import FPS, GAP_S, appearances, infer, load_sequence

MIN_DELTA_S = -5.0
MAX_DELTA_S = 60.0


def candidates(seq_dir: str) -> list[tuple[str, str, float, bool]]:
    """(caméra de sortie, caméra d'entrée, délai en s, vrai passage ?)."""
    cameras = load_sequence(seq_dir)
    apps = [(c, a) for c, frames in cameras.items() for a in appearances(frames, int(GAP_S * FPS))]
    following = {}                     # (identité, caméra, fin) → prochaine apparition
    by_id: dict = {}
    for c, a in apps:
        by_id.setdefault(a["id"], []).append((c, a))
    for items in by_id.values():
        items.sort(key=lambda ca: ca[1]["start"])
        for (c1, a1), (c2, a2) in zip(items, items[1:]):
            following[(a1["id"], c1, a1["end"])] = (c2, a2["start"])
    pairs = []
    for ca, a in apps:
        for cb, b in apps:
            if ca == cb:
                continue
            delta = (b["start"] - a["end"]) / FPS
            if MIN_DELTA_S <= delta <= MAX_DELTA_S:
                true = a["id"] == b["id"] and following.get((a["id"], ca, a["end"])) == (cb, b["start"])
                pairs.append((ca, cb, delta, true))
    return pairs


def auc(scores, labels) -> float:
    """Probabilité qu'un vrai passage ait un meilleur score qu'un faux (ex æquo : ½)."""
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + ties / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    args = parser.parse_args()

    topology = Topology.from_dict(infer(args.annotations, args.train))
    root = os.path.expanduser(args.annotations)
    pairs = [p for seq in args.test for p in candidates(os.path.join(root, seq))]
    labels = [p[3] for p in pairs]
    topo = [topology.score(a, b, d) for a, b, d, _ in pairs]
    delay = [1 / (1 + max(0.0, d)) for _a, _b, d, _ in pairs]
    n_true = sum(labels)
    print(f"{len(pairs)} paires candidates, dont {n_true} vrais passages "
          f"({len(topology.links)} liens appris sur {len(args.train)} séquences)")
    # Seuil de chaque score : celui qui garde 90 % des vrais passages, pour
    # comparer ce que chacun écarte à rappel égal.
    for name, scores in (("délai seul", delay), ("topologie", topo)):
        s, y = np.asarray(scores), np.asarray(labels)
        keep = s >= np.quantile(s[y], 0.1)
        print(f"  {name:<11} AUC {auc(scores, labels):.3f} ; en gardant 90 % des vrais "
              f"passages, faux écartés {(~keep[~y]).mean():.1%}")


if __name__ == "__main__":
    main()
