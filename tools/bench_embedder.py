"""
bench_embedder.py — Compare les embedders d'apparence sur les mêmes crops.

Les crops sont ceux que le pipeline enverrait à l'embedder : détections YOLO
(config.YOLO_MODEL) découpées par `DeepSort.crop_bb`. Pour chaque embedder :

  - vitesse : temps par image (tous les crops de l'image en un appel) ;
  - fidélité : écart des distances cosinus entre crops d'une même image,
    comparées à celles de la référence deep_sort_realtime. C'est ce que voit
    le gating d'apparence de DeepSORT (seuil DEEPSORT_MAX_COSINE_DISTANCE).

La fidélité ne suffit pas à juger un écart : l'effet sur le tracking se
mesure avec tools/eval_mot.py (DEEPSORT_EMBEDDER_ENGINE=… dans
l'environnement).

Depuis la racine du projet :

    uv run -m tools.bench_embedder --images "~/datasets/MOT17/train/MOT17-04-FRCNN/img1/%06d.jpg" \\
        --engine mobilenetv2-embedder.engine
"""

import os

# Avant numpy : sans ça, les threads OpenBLAS se disputent le CPU avec torch
# et faussent les temps mesurés (cf. app.py).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import statistics
import sys
import time

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
from ultralytics import YOLO

import config
from core.appearance_embedder import TensorRTEmbedder, TorchEmbedder
from core.face_body_tracker import FaceBodyTracker

WARMUP_FRAMES = 30


def collect_crops(pattern, count):
    yolo = YOLO(config.YOLO_MODEL, task="pose")
    cap = cv2.VideoCapture(os.path.expanduser(pattern))
    frames_crops = []
    while len(frames_crops) < count + WARMUP_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        detections = []
        for result in yolo.predict(frame, classes=[0], conf=config.YOLO_CONF_THRESHOLD,
                                   device=0, verbose=False):
            detections.extend(FaceBodyTracker._parse_result(result))
        detections = [d for d in detections if d[0][2] > 0 and d[0][3] > 0]
        frames_crops.append(DeepSort.crop_bb(frame, detections)[0])
    if len(frames_crops) <= WARMUP_FRAMES:
        sys.exit(f"Pas assez d'images lisibles : {pattern}")
    return frames_crops


def timed_run(embed, frames_crops):
    for crops in frames_crops[:WARMUP_FRAMES]:
        embed(crops)
    times, features = [], []
    for crops in frames_crops[WARMUP_FRAMES:]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = embed(crops)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        features.append(np.asarray(out, dtype=np.float64).reshape(len(crops), -1))
    return times, features


def cosine_distances(features):
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    return 1.0 - unit @ unit.T


def distance_gaps(reference, candidate):
    gaps = []
    for ref, cand in zip(reference, candidate):
        if len(ref) < 2:
            continue
        upper = np.triu_indices(len(ref), 1)
        gaps.extend(np.abs(cosine_distances(ref) - cosine_distances(cand))[upper])
    return np.array(gaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", required=True,
                        help="vidéo ou motif d'images (ex. .../img1/%%06d.jpg)")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--engine", help="moteur TensorRT à inclure dans la comparaison")
    args = parser.parse_args()

    frames_crops = collect_crops(args.images, args.frames)
    reference = MobileNetv2_Embedder(half=True, bgr=True, gpu=True)
    embedders = {
        "deep_sort_realtime (référence)": reference.predict,
        "PyTorch (TorchEmbedder)": TorchEmbedder(gpu=True).predict,
    }
    if args.engine:
        embedders["TensorRT (TensorRTEmbedder)"] = TensorRTEmbedder(args.engine).predict

    results = {name: timed_run(embed, frames_crops) for name, embed in embedders.items()}
    ref_name = next(iter(embedders))
    ref_times, ref_features = results[ref_name]
    crops_per_frame = statistics.mean(len(c) for c in frames_crops[WARMUP_FRAMES:])
    print(f"\n{len(ref_times)} images, {crops_per_frame:.1f} crops/image, "
          f"GPU {torch.cuda.get_device_name(0)}\n")
    print(f"{'embedder':<32}{'méd. ms':>9}{'p95 ms':>9}{'gain':>7}"
          f"{'|Δd| méd.':>11}{'|Δd| max':>10}{'> 1e-3':>8}")
    for name, (times, features) in results.items():
        ordered = sorted(times)
        p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        line = (f"{name:<32}{statistics.median(times):>9.2f}{p95:>9.2f}"
                f"{statistics.median(ref_times) / statistics.median(times):>6.2f}x")
        if name != ref_name:
            gaps = distance_gaps(ref_features, features)
            line += f"{np.median(gaps):>11.2e}{gaps.max():>10.2e}{(gaps > 1e-3).mean():>8.1%}"
        print(line)
    print(f"\nSeuil de gating d'apparence : DEEPSORT_MAX_COSINE_DISTANCE = "
          f"{config.DEEPSORT_MAX_COSINE_DISTANCE}")


if __name__ == "__main__":
    main()
