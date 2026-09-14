"""
eval_mot.py — MOTA / IDF1 des backends de tracking sur une séquence MOT17.

Protocole « détections publiques » : le tracker est nourri des détections
fournies avec le jeu de données (`det/det.txt`), pas de YOLO. C'est ce qui
isole la qualité du tracking de celle du détecteur — sans ça, on mesurerait
surtout YOLO. Seuls les embeddings d'apparence sont calculés ici, avec le même
MobileNetV2 que le pipeline.

Les variantes à comparer sont passées en ligne de commande et écrasent la
configuration le temps du run, pour comparer sans toucher au `.env` :

    python -m tools.eval_mot --seq ~/datasets/MOT17/train/MOT17-04-SDP \\
        --variant rust:100 --variant rust:none --variant python:100

Réserve sur les chiffres : `motmetrics` est appliqué directement sur la vérité
terrain, sans le pré-traitement officiel de MOTChallenge (retrait des
détections appariées à un distracteur). Les valeurs absolues ne sont donc pas
comparables au classement en ligne ; les écarts entre variantes, si — toutes
passent par le même filtre.
"""

import os

# Avant numpy : sans ça, les threads OpenBLAS se disputent le CPU avec torch
# et ralentissent l'association (cf. app.py).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import configparser
import sys
import time

import cv2
import numpy as np

import config


def load_public_detections(seq_dir, min_conf):
    """det.txt → { frame: [([l, t, w, h], conf, 'person', None), ...] }."""
    path = os.path.join(seq_dir, "det", "det.txt")
    if not os.path.exists(path):
        sys.exit(f"Détections publiques introuvables : {path}")
    by_frame: dict[int, list] = {}
    for row in np.loadtxt(path, delimiter=","):
        frame, _, left, top, width, height, conf = row[:7]
        if conf < min_conf or width <= 0 or height <= 0:
            continue
        # MOT indexe les pixels à la manière de Matlab ; motmetrics ramène la
        # vérité terrain en base 0 au chargement, on fait pareil ici pour que
        # les boîtes comparées vivent dans le même repère.
        by_frame.setdefault(int(frame), []).append(
            ([float(left) - 1, float(top) - 1, float(width), float(height)],
             float(conf), "person", None)
        )
    return by_frame


def sequence_length(seq_dir):
    ini = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(ini):
        parser = configparser.ConfigParser()
        parser.read(ini)
        return int(parser["Sequence"]["seqLength"])
    return len(os.listdir(os.path.join(seq_dir, "img1")))


def run_variant(seq_dir, backend_name, nn_budget, n_frames, min_conf, overrides):
    """Rejoue la séquence avec un backend et retourne (résultats MOT, temps)."""
    config.TRACKER_BACKEND = backend_name
    config.DEEPSORT_NN_BUDGET = nn_budget
    for name, value in overrides.items():
        setattr(config, name, value)
    # Import tardif : les backends lisent la configuration à la construction.
    from core.tracker_backends import build_body_tracker

    tracker = build_body_tracker()
    detections = load_public_detections(seq_dir, min_conf)
    img_dir = os.path.join(seq_dir, "img1")

    rows, durations = [], []
    for frame_id in range(1, n_frames + 1):
        frame = cv2.imread(os.path.join(img_dir, f"{frame_id:06d}.jpg"))
        if frame is None:
            break
        start = time.perf_counter()
        tracks = tracker.update(detections.get(frame_id, []), frame)
        durations.append(time.perf_counter() - start)
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = [float(v) for v in track.to_ltrb()]
            rows.append([frame_id, int(track.track_id),
                         x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1])
    return np.array(rows), durations


def evaluate(gt, results, name):
    """MOTA / IDF1 d'un jeu de résultats contre la vérité terrain chargée."""
    import motmetrics as mm
    import pandas as pd

    # Les résultats sont déjà en base 0 : on construit le DataFrame à la main
    # plutôt que de repasser par loadtxt, qui retrancherait 1 une seconde fois.
    tracked = pd.DataFrame(
        {
            "X": results[:, 2], "Y": results[:, 3],
            "Width": results[:, 4], "Height": results[:, 5],
            "Confidence": 1.0,
        },
        index=pd.MultiIndex.from_arrays(
            [results[:, 0].astype(int), results[:, 1].astype(int)],
            names=["FrameId", "Id"],
        ),
    )
    acc = mm.utils.compare_to_groundtruth(gt, tracked, "iou", distth=0.5)
    return mm.metrics.create().compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "num_false_positives",
                 "num_misses", "num_fragmentations", "mostly_tracked",
                 "mostly_lost", "num_unique_objects"],
        name=name,
    )


def load_ground_truth(gt_path):
    """Vérité terrain réduite aux piétons annotés (classe 1).

    MOT17 annote aussi les personnes sur un véhicule, les personnes statiques,
    les distracteurs et les reflets. Le protocole officiel les traite comme des
    zones à ignorer ; ici elles sont simplement écartées, ce qui compte les
    détections qui tombent dessus comme des faux positifs et tire le MOTA
    absolu vers le bas. Uniforme sur toutes les variantes comparées.
    """
    import motmetrics as mm

    gt = mm.io.loadtxt(gt_path, fmt="mot16", min_confidence=1)
    return gt[gt["ClassId"] == 1] if "ClassId" in gt.columns else gt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", required=True, help="Dossier d'une séquence MOT17")
    parser.add_argument("--variant", action="append", required=True,
                        help="backend:budget, ex. rust:100 ou python:none")
    parser.add_argument("--frames", type=int, default=0, help="0 = toute la séquence")
    parser.add_argument("--det-conf", type=float, default=0.0,
                        help="Seuil sur la confiance des détections publiques")
    parser.add_argument("--max-cosine-distance", type=float,
                        help="Écrase DEEPSORT_MAX_COSINE_DISTANCE")
    parser.add_argument("--max-iou-distance", type=float,
                        help="Écrase DEEPSORT_MAX_IOU_DISTANCE")
    parser.add_argument("--sweep-cosine",
                        help="Liste de seuils cosinus à comparer, ex. 0.1,0.2,0.3 — "
                             "chaque variante est rejouée pour chacun")
    parser.add_argument("--max-age", type=int, help="Écrase DEEPSORT_MAX_AGE")
    parser.add_argument("--n-init", type=int, help="Écrase DEEPSORT_N_INIT")
    args = parser.parse_args()

    overrides = {
        name: value
        for name, value in (
            ("DEEPSORT_MAX_COSINE_DISTANCE", args.max_cosine_distance),
            ("DEEPSORT_MAX_IOU_DISTANCE", args.max_iou_distance),
            ("DEEPSORT_MAX_AGE", args.max_age),
            ("DEEPSORT_N_INIT", args.n_init),
        )
        if value is not None
    }

    import motmetrics as mm
    import pandas as pd

    seq_dir = os.path.expanduser(args.seq)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    if not os.path.exists(gt_path):
        sys.exit(f"Vérité terrain introuvable : {gt_path} "
                 "(séquence de test ? seules celles de train sont annotées)")

    ground_truth = load_ground_truth(gt_path)
    n_frames = args.frames or sequence_length(seq_dir)
    print(f"\nSéquence : {os.path.basename(seq_dir)} — {n_frames} frames, "
          f"détections publiques (conf >= {args.det_conf})")
    if overrides:
        print("  réglages écrasés : "
              + ", ".join(f"{k.removeprefix('DEEPSORT_').lower()}={v}"
                          for k, v in overrides.items()))

    cosines = ([float(v) for v in args.sweep_cosine.split(",")]
               if args.sweep_cosine else [None])

    summaries, timings = [], {}
    for variant in args.variant:
        backend, _, budget = variant.partition(":")
        nn_budget = None if budget.lower() in ("none", "") else int(budget)
        for cosine in cosines:
            run_overrides = dict(overrides)
            label = f"{backend}/budget={budget or 'none'}"
            if cosine is not None:
                run_overrides["DEEPSORT_MAX_COSINE_DISTANCE"] = cosine
                label += f"/cos={cosine:g}"
            print(f"  → {label} ...", flush=True)
            results, durations = run_variant(seq_dir, backend, nn_budget, n_frames,
                                             args.det_conf, run_overrides)
            if not len(results):
                print("    aucune piste confirmée, variante ignorée")
                continue
            summaries.append(evaluate(ground_truth, results, label))
            timings[label] = float(np.median(durations)) * 1e3

    summary = pd.concat(summaries)
    summary["ms/frame"] = [timings[i] for i in summary.index]
    print()
    print(mm.io.render_summary(
        summary,
        formatters={**mm.metrics.create().formatters, "ms/frame": "{:.2f}".format},
        namemap={**mm.io.motchallenge_metric_names, "ms/frame": "ms/frame"},
    ))
    print("\nMOTA et IDF1 plus hauts = mieux ; IDs (changements d'identité) plus bas = mieux.")


if __name__ == "__main__":
    main()
