"""
eval_body_keep.py — Le corps distingue-t-il « visage caché » d'« échange de pistes » ?

Une piste nommée dont le visage de face sort « Inconnu » est soit la même
personne au visage caché (masque, cagoule : garder le nom), soit une autre
personne après un échange de pistes (retirer le nom). FaceBodyTracker ne
compte l'« Inconnu » que si le corps ne ressemble plus à la référence de la
piste (BODY_KEEP_SIMILARITY). Cet outil choisit ce seuil sur la vérité terrain.

Pour chaque personne annotée, une image sur STRIDE : descripteur couleur
haut / bas (core/clothing.py, proportions de la boîte, comme l'application)
et référence = moyenne glissante (BODY_REFERENCE_ALPHA) de ses descripteurs.
À chaque image t où la personne A est visible :

- paire « même personne » : référence de A figée DELAY_S plus tôt, comparée
  au corps de A à t (le visage a disparu depuis DELAY_S) ;
- paire « échange » : la même référence comparée au corps de chaque autre
  personne B proche de A à t (centres à moins de NEAR fois la largeur de A),
  celle sur qui la piste peut sauter.

Pour chaque seuil : part des paires « même personne » qui gardent le nom
(nom gardé sous masque) et des paires « échange » qui le gardent aussi
(échange non détecté par la série d'« Inconnu »).

Depuis la racine du projet (seuil choisi sur juin-juillet, vérifié sur décembre) :

    uv run -m tools.eval_body_keep --mot ~/datasets/CHIRLA/mot \\
        --tune seq_000_camera_5 seq_004_camera_2 seq_004_camera_4 seq_020_camera_4 \\
               seq_020_camera_5 \\
        --test seq_025_camera_2 seq_025_camera_3 seq_025_camera_5 seq_026_camera_3 \\
               seq_026_camera_5
"""

from __future__ import annotations

import argparse
import bisect
import os
from collections import defaultdict

import cv2
import numpy as np

import config
from core.clothing import descriptor, similarity
from tools.eval_identity import _sequence_fps, load_gt
from tools.eval_topology import auc

STRIDE = 10
NEAR = 1.5
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def sequence_pairs(seq_dir: str, delays_s: tuple[float, ...], smooth: bool = False) -> dict:
    """{délai: (similarités même personne, similarités échange)} d'une séquence."""
    gt = load_gt(seq_dir)
    fps = _sequence_fps(seq_dir)
    alpha = config.BODY_REFERENCE_ALPHA
    images = sorted(os.listdir(os.path.join(seq_dir, "img1")))
    # { id: [(image, descripteur, référence après cette image)] }
    samples = defaultdict(list)
    descs = {}
    smoothed = {}
    boxes = {}
    for frame in sorted(gt)[::STRIDE]:
        image = cv2.imread(os.path.join(seq_dir, "img1", images[frame - 1]))
        boxes[frame] = {pid: box for pid, *box in gt[frame]}
        for pid, box in boxes[frame].items():
            desc = descriptor(image, box)
            if desc is None:
                continue
            history = samples[pid]
            ref = desc if not history else (1 - alpha) * history[-1][2] + alpha * desc
            history.append((frame, desc, ref))
            descs[frame, pid] = desc
            smoothed[frame, pid] = ref

    pairs = {}
    for delay_s in delays_s:
        delay = int(delay_s * fps)
        same, swap = [], []
        for pid, history in samples.items():
            frames = [f for f, _, _ in history]
            for frame, desc, _ in history:
                i = bisect.bisect_right(frames, frame - delay) - 1
                if i < 0:
                    continue
                ref = history[i][2]
                current = smoothed[frame, pid] if smooth else desc
                same.append(similarity(ref, current))
                x1, _, x2, _ = boxes[frame][pid]
                cx, width = (x1 + x2) / 2, x2 - x1
                for other, (ox1, _, ox2, _) in boxes[frame].items():
                    if other == pid or abs((ox1 + ox2) / 2 - cx) > NEAR * width:
                        continue
                    other_desc = (smoothed if smooth else descs).get((frame, other))
                    if other_desc is not None:
                        swap.append(similarity(ref, other_desc))
        pairs[delay_s] = (same, swap)
    return pairs


def report(title: str, sequences: list[str], mot: str, delays_s, smooth: bool) -> None:
    pooled = {d: ([], []) for d in delays_s}
    for name in sequences:
        for delay_s, (same, swap) in sequence_pairs(os.path.join(mot, name), delays_s,
                                                         smooth).items():
            pooled[delay_s][0].extend(same)
            pooled[delay_s][1].extend(swap)
    print(f"\n## {title} ({', '.join(sequences)})\n")
    print("Nom gardé (même personne / échange) selon le seuil :\n")
    header = " | ".join(f"{t:.2f}" for t in THRESHOLDS)
    print(f"| Délai | Paires (même / échange) | AUC | {header} |")
    print("|---|---|---|" + "---|" * len(THRESHOLDS))
    for delay_s, (same, swap) in pooled.items():
        same, swap = np.asarray(same), np.asarray(swap)
        score = auc(np.concatenate([same, swap]),
                    np.r_[np.ones(len(same)), np.zeros(len(swap))])
        cells = " | ".join(f"{(same >= t).mean():.1%} / {(swap >= t).mean():.1%}"
                           for t in THRESHOLDS)
        print(f"| {delay_s:g} s | {len(same)} / {len(swap)} | {score:.3f} | {cells} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mot", required=True, help="dossier des séquences MOT17")
    parser.add_argument("--tune", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", default=[])
    parser.add_argument("--delays", nargs="+", type=float, default=[2.0, 10.0, 60.0])
    parser.add_argument("--smooth", action="store_true",
                        help="corps actuel lissé (moyenne glissante) plutôt qu'une image")
    args = parser.parse_args()
    report("Réglage", args.tune, os.path.expanduser(args.mot), args.delays, args.smooth)
    if args.test:
        report("Validation", args.test, os.path.expanduser(args.mot), args.delays, args.smooth)


if __name__ == "__main__":
    main()
