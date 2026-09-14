"""
sweep_deepsort.py — Règle les seuils DeepSORT sur des séquences annotées.

Rejoue l'association de `deep_sort_realtime` (backend python) pour chaque
combinaison d'une grille de seuils, et classe les combinaisons sur les
séquences de réglage. La meilleure est ensuite rejouée sur des séquences de
validation qu'elle n'a jamais vues, face aux réglages actuels : un réglage qui
ne gagne que sur les séquences qui l'ont choisi est du sur-apprentissage.

Les embeddings d'apparence ne dépendent pas des seuils : ils sont calculés une
fois par image (embedder de la configuration), puis seule l'association est
rejouée, en parallèle sur les cœurs CPU.

Format attendu d'une séquence : celui de MOT17 (`img1/`, `det/det.txt`,
`gt/gt.txt`, `seqinfo.ini`). Protocole et réserves : cf. tools/eval_mot.py.

Depuis la racine du projet :

    uv run -m tools.sweep_deepsort \\
        --tune ~/datasets/MOT17/train/MOT17-05-FRCNN ~/datasets/MOT17/train/MOT17-11-FRCNN \\
        --validate ~/datasets/MOT17/train/MOT17-04-FRCNN

`--grid cos=0.1,0.15 age=20,30` remplace les valeurs balayées d'un paramètre
(cos, iou, age, init), par exemple pour élargir la grille quand la meilleure
combinaison tombe sur une de ses bornes.

`--only-updated` n'évalue que les pistes détectées sur l'image courante, sans
les boîtes prédites des pistes en roue libre (personne occultée). C'est ce
qu'afficherait un pipeline qui garde la mémoire des pistes pour la
ré-identification sans dessiner les fantômes : `max_age` ne pèse plus alors
que sur la ré-identification, et plus sur les faux positifs.
"""

import os

# Avant numpy : l'association tourne dans plusieurs processus, les threads
# OpenBLAS de chacun se disputeraient les cœurs (cf. app.py).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import itertools
import multiprocessing
import time

import cv2
import numpy as np

import config
from tools import eval_mot

DEFAULT_GRID = {
    "DEEPSORT_MAX_COSINE_DISTANCE": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    "DEEPSORT_MAX_IOU_DISTANCE": [0.5, 0.7, 0.9],
    "DEEPSORT_MAX_AGE": [30, 70, 150],
    "DEEPSORT_N_INIT": [1, 3, 5],
}
SHORT = {
    "DEEPSORT_MAX_COSINE_DISTANCE": "cos",
    "DEEPSORT_MAX_IOU_DISTANCE": "iou",
    "DEEPSORT_MAX_AGE": "age",
    "DEEPSORT_N_INIT": "init",
}
METRICS = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses"]

# Rempli avant la création du pool : les processus fils en héritent par fork
# au lieu de recevoir les embeddings sérialisés à chaque tâche.
_SEQUENCES: dict[str, dict] = {}
_OPTIONS = {"only_updated": False}


def load_sequence(seq_dir, embedder):
    """Détections publiques et embeddings de chaque image, calculés une fois."""
    from deep_sort_realtime.deepsort_tracker import DeepSort

    detections = eval_mot.load_public_detections(seq_dir, 0.0)
    frames = []
    for frame_id in range(1, eval_mot.sequence_length(seq_dir) + 1):
        dets = [d for d in detections.get(frame_id, []) if d[0][2] > 0 and d[0][3] > 0]
        embeds = []
        if dets:
            image = cv2.imread(eval_mot.frame_path(seq_dir, frame_id))
            embeds = list(embedder.predict(DeepSort.crop_bb(image, dets)[0]))
        frames.append((frame_id, dets, embeds))
    return {"frames": frames,
            "gt": eval_mot.load_ground_truth(os.path.join(seq_dir, "gt", "gt.txt"))}


def track(seq_name, params):
    """Accumulateur motmetrics d'une séquence rejouée avec `params`."""
    import motmetrics as mm
    import pandas as pd

    from core.tracker_backends import new_deep_sort

    for name, value in params.items():
        setattr(config, name, value)
    tracker = new_deep_sort()
    rows = []
    for frame_id, dets, embeds in _SEQUENCES[seq_name]["frames"]:
        for t in tracker.update_tracks(dets, embeds=embeds):
            if t.is_confirmed() and (not _OPTIONS["only_updated"] or t.time_since_update == 0):
                x1, y1, x2, y2 = (float(v) for v in t.to_ltrb())
                rows.append((frame_id, int(t.track_id), x1, y1, x2 - x1, y2 - y1))
    res = np.array(rows, dtype=float).reshape(-1, 6)
    tracked = pd.DataFrame(
        {"X": res[:, 2], "Y": res[:, 3], "Width": res[:, 4], "Height": res[:, 5],
         "Confidence": 1.0},
        index=pd.MultiIndex.from_arrays([res[:, 0].astype(int), res[:, 1].astype(int)],
                                        names=["FrameId", "Id"]),
    )
    return mm.utils.compare_to_groundtruth(_SEQUENCES[seq_name]["gt"], tracked, "iou",
                                           distth=0.5)


def evaluate(job):
    """(params, métriques globales, détail par séquence) sur un groupe de séquences."""
    import motmetrics as mm

    params, seq_names = job
    accs = [track(name, params) for name in seq_names]
    summary = mm.metrics.create().compute_many(accs, metrics=METRICS, names=seq_names,
                                               generate_overall=True)
    return params, summary.loc["OVERALL"].to_dict(), summary


def label(params):
    return " ".join(f"{SHORT[k]}={v:g}" for k, v in params.items())


def row(params, metrics):
    return (f"  {label(params):<36} MOTA {metrics['mota']:6.1%}  IDF1 {metrics['idf1']:6.1%}"
            f"  IDs {int(metrics['num_switches']):4d}  FP {int(metrics['num_false_positives']):6d}"
            f"  FN {int(metrics['num_misses']):6d}")


def current_params():
    return {name: getattr(config, name) for name in DEFAULT_GRID}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tune", nargs="+", required=True, help="séquences de réglage")
    parser.add_argument("--validate", nargs="*", default=[], help="séquences de validation")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--grid", nargs="*", default=[], metavar="PARAM=V1,V2",
                        help="valeurs balayées pour cos, iou, age ou init")
    parser.add_argument("--only-updated", action="store_true",
                        help="n'évaluer que les pistes détectées sur l'image courante")
    args = parser.parse_args()
    _OPTIONS["only_updated"] = args.only_updated

    grid_values = dict(DEFAULT_GRID)
    by_short = {short: name for name, short in SHORT.items()}
    for spec in args.grid:
        short, _, values = spec.partition("=")
        if short not in by_short or not values:
            parser.error(f"--grid {spec!r} : attendu cos|iou|age|init=v1,v2,…")
        cast = float if short in ("cos", "iou") else int
        grid_values[by_short[short]] = [cast(v) for v in values.split(",")]

    from core.appearance_embedder import build_embedder

    embedder = build_embedder()
    tune = [os.path.expanduser(p) for p in args.tune]
    validate = [os.path.expanduser(p) for p in args.validate]
    start = time.time()
    for seq_dir in tune + validate:
        _SEQUENCES[os.path.basename(seq_dir)] = load_sequence(seq_dir, embedder)
    del embedder
    tune_names = [os.path.basename(p) for p in tune]
    validate_names = [os.path.basename(p) for p in validate]
    print(f"Embeddings calculés en {time.time() - start:.0f} s "
          f"({config.DEEPSORT_EMBEDDER_ENGINE or 'PyTorch'})")

    grid = [dict(zip(grid_values, values))
            for values in itertools.product(*grid_values.values())]
    baseline = current_params()
    if baseline not in grid:
        grid.append(baseline)
    print(f"Réglage sur {', '.join(tune_names)} : {len(grid)} combinaisons, "
          f"{args.workers} processus...")

    start = time.time()
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(evaluate, [(params, tune_names) for params in grid], chunksize=1)
    print(f"Balayage terminé en {time.time() - start:.0f} s\n")

    # IDF1 d'abord : VisionCam doit garder une identité attachée à la bonne
    # piste. Le MOTA départage les égalités.
    results.sort(key=lambda r: (r[1]["idf1"], r[1]["mota"]), reverse=True)
    baseline_metrics = next(m for p, m, _ in results if p == baseline)
    best = results[0][0]
    on_edge = [SHORT[name] for name, values in grid_values.items()
               if len(values) > 1 and best[name] in (min(values), max(values))]
    print("Meilleures combinaisons (réglage, classées par IDF1 puis MOTA) :")
    for params, metrics, _ in results[:args.top]:
        print(row(params, metrics))
    print(f"\nRéglages actuels :\n{row(baseline, baseline_metrics)}")
    if on_edge:
        print(f"\nAttention : la meilleure combinaison est en bord de grille pour "
              f"{', '.join(on_edge)} — l'optimum peut se trouver au-delà (--grid).")
    if not validate_names or best == baseline:
        return
    print(f"\nValidation sur {', '.join(validate_names)} (séquences non vues) :")
    for name, params in (("actuels", baseline), ("meilleurs", best)):
        _, metrics, summary = evaluate((params, validate_names))
        print(f" {name} :\n{row(params, metrics)}")
        for seq in validate_names:
            m = summary.loc[seq]
            print(f"    {seq:<18} MOTA {m['mota']:6.1%}  IDF1 {m['idf1']:6.1%}"
                  f"  IDs {int(m['num_switches']):4d}")


if __name__ == "__main__":
    main()
