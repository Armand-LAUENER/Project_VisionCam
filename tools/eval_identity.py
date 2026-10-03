"""
eval_identity.py — Identité de bout en bout : le nom affiché est-il le bon ?

Le tracking (MOTA/IDF1, tools/eval_mot.py) et la reconnaissance (par visage,
tools/bench_face.py) se mesurent séparément ; ici, c'est ce que voit
l'utilisateur. Entrées :

- le journal des pistes de l'application (TRACKS_LOG_PATH), écrit en
  rejouant une séquence en `every_frame` (VIDEO_FILE) : les numéros d'image
  sont alors ceux de la séquence ;
- la vérité terrain de la séquence (`gt/gt.txt`, MOT17) ;
- le nom enrôlé de chaque identité annotée : `<préfixe><identité>`, s'il
  figure dans `--known-faces` (cf. tools/enroll_from_sequence.py), ou
  `--names id=nom …` pour une séquence annotée à la main.

Pour chaque personne annotée et chaque image où elle est visible, la piste
affichée qui la recouvre (IoU ≥ 0,5, un-pour-un) donne une issue : bon nom,
mauvais nom, « Inconnu », ou pas de piste. Pour une personne non enrôlée,
« Inconnu » est la bonne réponse et un nom une erreur.

Tracking des pistes affichées : MOTA, IDF1 et changements d'identité de ce que
montre l'application (YOLO, DeepSORT, pistes en roue libre masquées), là où
tools/eval_mot.py mesure DeepSORT seul sur les détections publiques. Une
personne suivie tour à tour par deux pistes compte un changement à chaque
bascule.

Délai avant le bon nom : pour chaque apparition d'une personne enrôlée (une
absence de plus d'une seconde en ouvre une nouvelle), temps entre sa première
image et la première où son nom est affiché.

Depuis la racine du projet :

    uv run -m tools.eval_identity --tracks data/tracks.csv \\
        --sequence ~/datasets/CHIRLA/mot/seq_025_camera_2 --known-faces /tmp/known_faces_chirla
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np

from core.face_body_tracker import match_tracks_to_detections

UNKNOWN = "Inconnu"
MIN_IOU = 0.5
GAP_S = 1.0


@dataclass
class Counts:
    frames: int = 0
    correct: int = 0
    wrong: int = 0
    unknown: int = 0
    untracked: int = 0

    def add(self, other: Counts) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def rates(self) -> dict:
        n = max(1, self.frames)
        return {k: v / n for k, v in self.as_dict().items() if k != "frames"}


@dataclass
class Report:
    per_id: dict           # {identité GT: Counts}
    enrolled: Counts
    unenrolled: Counts
    time_to_name: list     # secondes par apparition, None si jamais nommée


def load_tracks(path: str) -> dict:
    """{image: [(track_id, x, y, l, h, nom)]} depuis le journal TRACKS_LOG_PATH."""
    tracks = defaultdict(list)
    with open(path, newline="") as f:
        for frame, track_id, x, y, w, h, name in csv.reader(f):
            tracks[int(frame)].append((int(track_id), float(x), float(y), float(w), float(h), name))
    return tracks


def _matches(gt_boxes, track_boxes):
    return match_tracks_to_detections(gt_boxes, track_boxes, min_iou=MIN_IOU)


def evaluate(gt: dict, tracks: dict, gt_names: dict, fps: float) -> Report:
    """gt : {image: [(id, x1, y1, x2, y2)]} ; gt_names : {id: nom enrôlé ou None}."""
    per_id = defaultdict(Counts)
    first_seen: dict[int, int] = {}      # début de l'apparition en cours
    last_seen: dict[int, int] = {}
    named: dict[int, bool] = {}
    delays: list = []

    for frame in sorted(gt):
        people = gt[frame]
        shown = tracks.get(frame, [])
        matched = _matches([p[1:] for p in people], [t[1:5] for t in shown])
        for index, (gid, *_box) in enumerate(people):
            expected = gt_names.get(gid)
            counts = per_id[gid]
            counts.frames += 1
            if gid in last_seen and frame - last_seen[gid] > GAP_S * fps:
                if expected and not named[gid]:
                    delays.append(None)
                del first_seen[gid]
            if gid not in first_seen:
                first_seen[gid], named[gid] = frame, False
            last_seen[gid] = frame

            if index not in matched:
                counts.untracked += 1
                continue
            name = shown[matched[index]][5]
            if expected is None:
                if name == UNKNOWN:
                    counts.correct += 1
                else:
                    counts.wrong += 1
            elif name == expected:
                counts.correct += 1
                if not named[gid]:
                    named[gid] = True
                    delays.append((frame - first_seen[gid]) / fps)
            elif name == UNKNOWN:
                counts.unknown += 1
            else:
                counts.wrong += 1

    for gid in first_seen:
        if gt_names.get(gid) and not named[gid]:
            delays.append(None)

    enrolled, unenrolled = Counts(), Counts()
    for gid, counts in per_id.items():
        (enrolled if gt_names.get(gid) else unenrolled).add(counts)
    return Report(dict(per_id), enrolled, unenrolled, delays)


def tracking_metrics(gt: dict, tracks: dict) -> dict:
    """MOTA, IDF1 et changements d'ID (motmetrics, appariement IoU ≥ 0,5)."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(set(gt) | set(tracks)):
        people = gt.get(frame, [])
        shown = tracks.get(frame, [])
        gt_boxes = [(x1, y1, x2 - x1, y2 - y1) for _, x1, y1, x2, y2 in people]
        distances = mm.distances.iou_matrix(gt_boxes, [t[1:5] for t in shown],
                                            max_iou=1 - MIN_IOU)
        acc.update([p[0] for p in people], [t[0] for t in shown], distances, frameid=frame)
    summary = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches"])
    return {"mota": float(summary["mota"].iloc[0]), "idf1": float(summary["idf1"].iloc[0]),
            "switches": int(summary["num_switches"].iloc[0])}


def load_gt(seq_dir: str) -> dict:
    """{image: [(id, x1, y1, x2, y2)]} des piétons à évaluer (MOT17 : classe 1, drapeau 1).

    CHIRLA n'annote que des piétons à évaluer : rien n'y est écarté.
    """
    gt = defaultdict(list)
    with open(os.path.join(seq_dir, "gt", "gt.txt")) as f:
        for line in f:
            values = line.strip().split(",")
            frame, track_id, left, top, width, height = (float(v) for v in values[:6])
            if len(values) >= 8 and (float(values[6]) == 0 or int(float(values[7])) != 1):
                continue
            x1, y1 = left - 1, top - 1
            gt[int(frame)].append((int(track_id), x1, y1, x1 + width, y1 + height))
    return gt


def _sequence_fps(seq_dir: str) -> float:
    parser = configparser.ConfigParser()
    parser.read(os.path.join(seq_dir, "seqinfo.ini"))
    return float(parser["Sequence"]["frameRate"])


def main():
    from tools.recognition_threshold_study import wilson

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tracks", required=True, help="journal TRACKS_LOG_PATH")
    parser.add_argument("--sequence", required=True, help="séquence MOT17 avec gt/gt.txt")
    parser.add_argument("--known-faces", help="dossier known_faces/ utilisé par l'application")
    parser.add_argument("--prefix", default="id_", help="préfixe des noms enrôlés (défaut : id_)")
    parser.add_argument("--names", nargs="*", default=[], metavar="ID=NOM",
                        help="nom enrôlé d'une identité annotée, en plus de --known-faces")
    args = parser.parse_args()

    seq_dir = os.path.expanduser(args.sequence)
    gt = load_gt(seq_dir)
    enrolled_names = (set(os.listdir(os.path.expanduser(args.known_faces)))
                      if args.known_faces else set())
    explicit = dict(item.split("=", 1) for item in args.names)
    gt_names = {}
    for gid in {p[0] for people in gt.values() for p in people}:
        name = explicit.get(str(gid)) or f"{args.prefix}{gid}"
        gt_names[gid] = name if (str(gid) in explicit or name in enrolled_names) else None

    fps = _sequence_fps(seq_dir)
    tracks = load_tracks(args.tracks)
    if not tracks:
        # Rejeu raté (vidéo illisible, séquence sans images) : pas un tracking à 0 %.
        raise SystemExit(f"Aucune piste dans {args.tracks} : le rejeu a-t-il tourné ?")
    report = evaluate(gt, tracks, gt_names, fps)
    metrics = tracking_metrics(gt, tracks)
    print(f"Tracking des pistes affichées : MOTA {metrics['mota']:.1%}, "
          f"IDF1 {metrics['idf1']:.1%}, changements d'ID {metrics['switches']}")

    def line(label, counts, correct_label):
        n = counts.frames
        if not n:
            return
        rates = counts.rates()
        low, high = wilson(counts.wrong, n)
        print(f"{label:<28} │ {n:>7} │ {correct_label} {rates['correct']:6.1%}  "
              f"mauvais nom {rates['wrong']:5.1%} [IC 95 % {low:.1%}-{high:.1%}]  "
              f"Inconnu {rates['unknown']:5.1%}  sans piste {rates['untracked']:5.1%}")

    print(f"Séquence {os.path.basename(seq_dir.rstrip('/'))}, {fps:.0f} i/s ; part des "
          f"images-personnes annotées. IC : Wilson, images non indépendantes (indicatif).")
    print(f"{'':<28} │ {'images':>7} │")
    for gid in sorted(report.per_id):
        label = f"id {gid} ({gt_names[gid] or 'non enrôlé'})"
        line(label, report.per_id[gid], "bon nom" if gt_names[gid] else "Inconnu ")
    line("TOTAL enrôlés", report.enrolled, "bon nom")
    line("TOTAL non enrôlés", report.unenrolled, "Inconnu ")

    delays = [d for d in report.time_to_name if d is not None]
    never = sum(d is None for d in report.time_to_name)
    if report.time_to_name:
        p50 = f"{np.median(delays):.1f}" if delays else "—"
        p95 = f"{np.percentile(delays, 95):.1f}" if delays else "—"
        print(f"\nDélai avant le bon nom : {len(report.time_to_name)} apparitions, "
              f"médiane {p50} s, p95 {p95} s ; jamais nommées : {never}")


if __name__ == "__main__":
    main()
