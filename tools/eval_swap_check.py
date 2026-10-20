"""
eval_swap_check.py — Après un croisement, le corps dit-il qui est qui ?

Pour garder le nom d'une personne au visage caché quand d'autres sont visibles,
il faut savoir si sa piste a pu passer sur une autre personne. Un échange de
pistes n'arrive qu'au contact : deux personnes dont les boîtes se recouvrent.
Après le contact, on dispose de l'apparence des deux personnes d'avant ; le
test est relatif : le corps suivi ressemble-t-il plus à A d'avant qu'à B
d'avant ?

Sur la vérité terrain (format MOT17) : chaque contact entre A et B (IoU ≥
CONTACT_IOU, interruptions de moins de MERGE_S fusionnées) est un événement.
Galeries « avant » : vues propres (IoU < CLEAN_IOU avec toute autre boîte) de
A et B dans les PRE_S secondes avant le contact ; corps « après » : moyenne des
POST_SAMPLES premières vues propres dans les POST_S secondes après. Marge =
sim(après, A avant) − sim(après, B avant), pour la piste restée sur A (même
personne) et pour la piste passée sur B (échange), dans les deux sens.

Signaux : couleur des vêtements (core/clothing.py, Bhattacharyya), embedding
MobileNetV2 de DeepSORT (core/appearance_embedder.py, cosinus), et leur somme
réduite (chaque marge divisée par son écart-type sur le réglage). Seuil de
marge choisi sur le réglage pour laisser passer au plus 1 % des échanges, puis
appliqué tel quel à la validation.

Depuis la racine du projet (seuil choisi sur juin-juillet, vérifié sur
décembre) ; chaque argument est un dossier de séquence complet :

    D=~/datasets/CHIRLA/mot
    uv run -m tools.eval_swap_check --name CHIRLA \\
        --tune $D/seq_000_camera_5 $D/seq_004_camera_2 $D/seq_004_camera_4 \\
               $D/seq_020_camera_4 $D/seq_020_camera_5 \\
        --test $D/seq_025_camera_2 $D/seq_025_camera_3 $D/seq_025_camera_5 \\
               $D/seq_026_camera_3 $D/seq_026_camera_5
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np

from core.clothing import descriptor, similarity
from core.face_body_tracker import _iou
from tools.eval_identity import _sequence_fps, load_gt
from tools.eval_topology import auc

CONTACT_IOU = 0.2
CLEAN_IOU = 0.05
MERGE_S = 0.5
PRE_S = 2.0
POST_S = 1.5
STRIDE = 3
MIN_SAMPLES = 3
POST_SAMPLES = 5
MAX_SWAP_PASS = 0.01
SIGNALS = ("couleur", "mobilenet", "combiné")


def contact_events(gt: dict, fps: float) -> list[tuple[int, int, int, int]]:
    """[(A, B, première image, dernière image)] des contacts entre deux personnes."""
    contacts = defaultdict(list)
    for frame, people in gt.items():
        for i, (a, *box_a) in enumerate(people):
            for b, *box_b in people[i + 1:]:
                if _iou(box_a, box_b) >= CONTACT_IOU:
                    contacts[min(a, b), max(a, b)].append(frame)
    events, merge = [], int(MERGE_S * fps)
    for (a, b), frames in contacts.items():
        frames.sort()
        start = prev = frames[0]
        for f in frames[1:] + [None]:
            if f is None or f - prev > merge:
                events.append((a, b, start, prev))
                if f is not None:
                    start = f
            if f is not None:
                prev = f
    return events


def clean_frames(gt: dict, pid: int, frames) -> list[tuple[int, tuple]]:
    """Images de `frames` où `pid` est visible sans recouvrir personne, et sa boîte."""
    out = []
    for f in frames:
        people = {p: box for p, *box in gt.get(f, [])}
        if pid not in people:
            continue
        if all(_iou(people[pid], box) < CLEAN_IOU for p, box in people.items() if p != pid):
            out.append((f, tuple(people[pid])))
    return out


class Features:
    """Descripteurs couleur et embeddings, calculés à la demande et gardés."""

    def __init__(self, seq_dir: str, embedder) -> None:
        self.seq_dir = seq_dir
        self.images = sorted(os.listdir(os.path.join(seq_dir, "img1")))
        self.embedder = embedder
        self.cache: dict = {}

    def get(self, frame: int, box: tuple):
        key = (frame, box)
        if key not in self.cache:
            image = cv2.imread(os.path.join(self.seq_dir, "img1", self.images[frame - 1]))
            x1, y1, x2, y2 = (int(max(0, v)) for v in box)
            crop = image[y1:y2, x1:x2]
            colour = descriptor(image, box)
            emb = None
            if crop.size:
                emb = self.embedder.predict([crop])[0].astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-9
            self.cache[key] = (colour, emb)
        return self.cache[key]

    def mean(self, samples):
        feats = [self.get(f, box) for f, box in samples]
        colours = [c for c, _ in feats if c is not None]
        embs = [e for _, e in feats if e is not None]
        if not colours or not embs:
            return None
        emb = np.mean(embs, axis=0)
        return np.mean(colours, axis=0), emb / (np.linalg.norm(emb) + 1e-9)


def sims(post, pre) -> np.ndarray:
    return np.array([similarity(post[0], pre[0]), float(post[1] @ pre[1])])


def sequence_margins(seq_dir: str, embedder) -> tuple[list, list]:
    """Marges (couleur, mobilenet) des cas « même personne » et « échange »."""
    gt, fps = load_gt(seq_dir), _sequence_fps(seq_dir)
    feats = Features(seq_dir, embedder)
    same, swap = [], []
    for a, b, start, end in contact_events(gt, fps):
        pre_range = range(max(1, start - int(PRE_S * fps)), start, STRIDE)
        post_range = range(end + 1, end + 1 + int(POST_S * fps), STRIDE)
        pre = {p: clean_frames(gt, p, pre_range) for p in (a, b)}
        post = {p: clean_frames(gt, p, post_range)[:POST_SAMPLES] for p in (a, b)}
        if any(len(pre[p]) < MIN_SAMPLES or len(post[p]) < MIN_SAMPLES for p in (a, b)):
            continue
        g = {p: feats.mean(pre[p]) for p in (a, b)}
        q = {p: feats.mean(post[p]) for p in (a, b)}
        if any(v is None for v in (*g.values(), *q.values())):
            continue
        for named, other in ((a, b), (b, a)):
            # La piste portait `named` ; après le contact elle suit `named` (même
            # personne) ou `other` (échange). Marge = sim(·, named) − sim(·, other).
            same.append(sims(q[named], g[named]) - sims(q[named], g[other]))
            swap.append(sims(q[other], g[named]) - sims(q[other], g[other]))
    return same, swap


def pooled(sequences, embedder):
    same, swap, events = [], [], 0
    for seq in sequences:
        s, w = sequence_margins(os.path.expanduser(seq), embedder)
        same += s
        swap += w
        events += len(s) // 2
    return np.array(same).reshape(-1, 2), np.array(swap).reshape(-1, 2), events


def with_combined(margins: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.column_stack([margins, (margins / scale).sum(axis=1)])


def report(title, same, swap, events, thresholds) -> None:
    print(f"\n## {title} : {events} contacts, {len(same)} cas par hypothèse\n")
    print("| Signal | AUC | Échange repéré (marge < 0) | Seuil (≤ 1 % d'échanges sur le "
          "réglage) | Même personne vérifiée | Échanges qui passent |")
    print("|:--|---:|---:|---:|---:|---:|")
    for i, name in enumerate(SIGNALS):
        score = auc(np.r_[same[:, i], swap[:, i]], np.r_[np.ones(len(same)), np.zeros(len(swap))])
        t = thresholds[i]
        print(f"| {name} | {score:.3f} | {(swap[:, i] < 0).mean():.1%} | {t:.3f} | "
              f"{(same[:, i] >= t).mean():.1%} | {(swap[:, i] >= t).mean():.1%} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tune", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", default=[])
    parser.add_argument("--name", default="", help="titre des tableaux")
    args = parser.parse_args()

    from core.appearance_embedder import TorchEmbedder
    embedder = TorchEmbedder(gpu=True)

    same, swap, events = pooled(args.tune, embedder)
    scale = np.r_[same, swap].std(axis=0) + 1e-9
    same, swap = with_combined(same, scale), with_combined(swap, scale)
    thresholds = np.quantile(swap, 1 - MAX_SWAP_PASS, axis=0)
    report(f"{args.name} réglage", same, swap, events, thresholds)
    if args.test:
        same_t, swap_t, events_t = pooled(args.test, embedder)
        report(f"{args.name} validation", with_combined(same_t, scale),
               with_combined(swap_t, scale), events_t, thresholds)


if __name__ == "__main__":
    main()
