"""
eval_clothing.py — La couleur des vêtements aide-t-elle l'association entre caméras ? (roadmap 2.5)

Sur les séquences CHIRLA dont on a les vidéos de plusieurs caméras (format
MOT17, tools/convert_chirla.py) : chaque apparition d'une personne sur une
caméra reçoit un descripteur couleur haut / bas (core/clothing.py), moyen sur
ses SAMPLES dernières images si elle se termine, premières si elle commence.
Chaque paire « sortie sur A, entrée sur B ≠ A entre −5 et +60 s » est une
candidate ; la vraie est la prochaine apparition de la même personne.

Mesures : pour chaque entrée qui a une vraie sortie parmi ses candidates, le
score met-il cette sortie en tête (association juste) ? Et l'AUC sur toutes
les paires. Scores comparés : délai seul, topologie (core/topology.py,
apprise sur d'autres séquences), couleur seule, et leurs produits.

Depuis la racine du projet (topologie apprise en juin-juillet, jugée en décembre) :

    uv run -m tools.eval_clothing --mot ~/datasets/CHIRLA/mot \\
        --annotations ~/datasets/CHIRLA/annotations \\
        --train seq_000 seq_001 seq_002 seq_004 seq_006 seq_007 seq_020 \\
        --test seq_025_camera_2,seq_025_camera_3,seq_025_camera_5 seq_026_camera_3,seq_026_camera_5
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np

from core.clothing import descriptor, similarity
from core.topology import Topology
from tools.eval_identity import load_gt
from tools.eval_topology import auc
from tools.infer_topology import FPS, GAP_S, appearances, infer

MIN_DELTA_S, MAX_DELTA_S = -5.0, 60.0
SAMPLES = 8


def mean_descriptor(seq_dir, gt, pid, frames):
    descs = []
    for frame in frames:
        box = next((p[1:] for p in gt.get(frame, []) if p[0] == pid), None)
        image = cv2.imread(os.path.join(seq_dir, "img1", f"{frame:06d}.jpg")) if box else None
        if image is not None:
            d = descriptor(image, box)
            if d is not None:
                descs.append(d)
    return np.mean(descs, axis=0) if descs else None


def camera_appearances(seq_dir):
    gt = load_gt(seq_dir)
    apps = appearances(gt, int(GAP_S * FPS))
    for a in apps:
        span = range(a["start"], a["end"] + 1)
        step = max(1, len(span) // (2 * SAMPLES))
        a["desc_end"] = mean_descriptor(seq_dir, gt, a["id"], list(span)[::-1][:SAMPLES * step:step])
        a["desc_start"] = mean_descriptor(seq_dir, gt, a["id"], list(span)[:SAMPLES * step:step])
    return apps


def candidates(group):
    """group : {caméra (camera_N): apparitions}. (A, B, délai, vraie ?, sim, entrée)."""
    flat = [(c, a) for c, apps in group.items() for a in apps]
    nxt = {}
    by_id = defaultdict(list)
    for c, a in flat:
        by_id[a["id"]].append((c, a))
    for items in by_id.values():
        items.sort(key=lambda ca: ca[1]["start"])
        for (c1, a1), (c2, a2) in zip(items, items[1:]):
            nxt[(c1, a1["id"], a1["end"])] = (c2, a2["start"])
    pairs = []
    for ca, a in flat:
        for cb, b in flat:
            delta = (b["start"] - a["end"]) / FPS
            if ca == cb or not MIN_DELTA_S <= delta <= MAX_DELTA_S:
                continue
            true = a["id"] == b["id"] and nxt.get((ca, a["id"], a["end"])) == (cb, b["start"])
            sim = (similarity(a["desc_end"], b["desc_start"])
                   if a["desc_end"] is not None and b["desc_start"] is not None else 0.0)
            pairs.append((ca, cb, delta, true, sim, (cb, b["id"], b["start"])))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mot", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True,
                        help="groupes de séquences-caméras filmées ensemble, séparées par des virgules")
    args = parser.parse_args()

    topology = Topology.from_dict(infer(args.annotations, args.train))
    pairs = []
    for group in args.test:
        cams = {}
        for name in group.split(","):
            camera = "camera_" + name.rsplit("_", 1)[1]
            cams[camera] = camera_appearances(os.path.join(os.path.expanduser(args.mot), name))
        pairs += candidates(cams)

    scores = {
        "délai seul": [1 / (1 + max(0.0, d)) for _a, _b, d, *_ in pairs],
        "topologie": [topology.score(a, b, d) for a, b, d, *_ in pairs],
        "couleur seule": [p[4] for p in pairs],
    }
    scores["délai × couleur"] = [x * p[4] for x, p in zip(scores["délai seul"], pairs)]
    scores["topologie × couleur"] = [x * p[4] for x, p in zip(scores["topologie"], pairs)]
    labels = [p[3] for p in pairs]
    by_entry = defaultdict(list)
    for i, p in enumerate(pairs):
        by_entry[p[5]].append(i)
    entries = [ix for ix in by_entry.values() if any(labels[i] for i in ix)]
    print(f"{len(pairs)} paires, {sum(labels)} vraies ; {len(entries)} entrées à associer, "
          f"{np.mean([len(ix) for ix in entries]):.1f} candidates en moyenne")
    for name, s in scores.items():
        right = sum(labels[max(ix, key=lambda i: s[i])] for ix in entries)
        print(f"  {name:<20} association juste {right / len(entries):6.1%}   AUC {auc(s, labels):.3f}")


if __name__ == "__main__":
    main()
