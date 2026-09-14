"""
bench_yolo.py — Compare des variantes du modèle YOLOv8-Pose sur les mêmes images.

Mesure, pour chaque modèle passé en argument, le coût de l'étape de détection
telle que le pipeline la paie (`predict` + `FaceBodyTracker._parse_result`,
qui rapatrie les tenseurs sur le CPU), et compare ses sorties à celles du
premier modèle, la référence :

  - détections : même nombre par image, IoU des boîtes appariées ;
  - keypoints  : écart en pixels des keypoints COCO 0-6 (visage + épaules),
    ceux qui alimentent le crop visage et l'orientation.

Un modèle plus rapide qui déplace les keypoints dégrade la reconnaissance :
les deux mesures vont ensemble.

Depuis la racine du projet :

    uv run -m tools.bench_yolo --images "~/datasets/MOT17/train/MOT17-04-FRCNN/img1/%06d.jpg" \\
        --frames 300 yolov8s-pose.pt yolov8s-pose.pt:half yolov8s-pose.engine

`modèle:half` lance le modèle PyTorch en FP16 (`predict(half=True)`).
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
from ultralytics import YOLO

import config
from core.face_body_tracker import FaceBodyTracker

WARMUP_FRAMES = 30
# Un keypoint n'est comparé que s'il est jugé visible par les deux modèles,
# au seuil qu'utilise le pipeline pour le crop visage.
KP_CONF = config.POSE_NOSE_CONF_THRESHOLD
MATCH_IOU = 0.5


def load_frames(pattern, count):
    cap = cv2.VideoCapture(os.path.expanduser(pattern))
    frames = []
    while len(frames) < count + WARMUP_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    if len(frames) <= WARMUP_FRAMES:
        sys.exit(f"Pas assez d'images lisibles : {pattern}")
    return frames


def run(spec, frames):
    """Détections par image et temps par image (ms) après warm-up."""
    path, _, option = spec.partition(":")
    model = YOLO(path, task="pose")
    kwargs = dict(classes=[0], conf=config.YOLO_CONF_THRESHOLD, device=0, verbose=False)
    if option == "half":
        kwargs["half"] = True

    outputs, times = [], []
    for i, frame in enumerate(frames):
        torch.cuda.synchronize()
        start = time.perf_counter()
        detections = []
        for result in model.predict(frame, **kwargs):
            detections.extend(FaceBodyTracker._parse_result(result))
        elapsed = (time.perf_counter() - start) * 1000
        if i >= WARMUP_FRAMES:
            times.append(elapsed)
            outputs.append(detections)
    return outputs, times


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def compare(reference, candidate):
    """Appariement glouton par IoU décroissante, image par image."""
    same_count = unmatched = 0
    ious, kp_errors = [], []
    for ref_dets, cand_dets in zip(reference, candidate):
        same_count += len(ref_dets) == len(cand_dets)
        pairs = sorted(((iou(r[0], c[0]), i, j)
                        for i, r in enumerate(ref_dets)
                        for j, c in enumerate(cand_dets)), reverse=True)
        used_ref, used_cand = set(), set()
        for score, i, j in pairs:
            if score < MATCH_IOU or i in used_ref or j in used_cand:
                continue
            used_ref.add(i)
            used_cand.add(j)
            ious.append(score)
            for (rx, ry, rc), (cx, cy, cc) in zip(ref_dets[i][3]['pose_kps'],
                                                  cand_dets[j][3]['pose_kps']):
                if rc >= KP_CONF and cc >= KP_CONF:
                    kp_errors.append(float(np.hypot(rx - cx, ry - cy)))
        unmatched += len(ref_dets) - len(used_ref) + len(cand_dets) - len(used_cand)
    return same_count, unmatched, ious, kp_errors


def percentile(values, ratio):
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * ratio), len(ordered) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", required=True,
                        help="vidéo ou motif d'images (ex. .../img1/%%06d.jpg)")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("models", nargs="+",
                        help="le premier sert de référence ; suffixe :half pour FP16 PyTorch")
    args = parser.parse_args()

    frames = load_frames(args.images, args.frames)
    h, w = frames[0].shape[:2]
    print(f"{len(frames) - WARMUP_FRAMES} images {w}x{h}, {WARMUP_FRAMES} de warm-up, "
          f"GPU {torch.cuda.get_device_name(0)}\n")

    results = {spec: run(spec, frames) for spec in args.models}
    reference_spec = args.models[0]
    reference, reference_times = results[reference_spec]
    reference_median = statistics.median(reference_times)
    frames_measured = len(reference)
    persons = statistics.mean(len(d) for d in reference)
    print(f"Référence : {reference_spec} — {persons:.1f} personnes/image\n")

    print(f"{'modèle':<28} {'méd. ms':>8} {'p95 ms':>8} {'gain':>6} "
          f"{'nb dét. =':>10} {'non app.':>9} {'IoU méd.':>9} {'kp méd. px':>11} {'kp p95 px':>10}")
    for spec in args.models:
        outputs, times = results[spec]
        median = statistics.median(times)
        line = (f"{spec:<28} {median:>8.2f} {percentile(times, 0.95):>8.2f} "
                f"{reference_median / median:>5.2f}x")
        if spec != reference_spec:
            same, unmatched, ious, kp = compare(reference, outputs)
            line += (f" {same:>5}/{frames_measured:<4} {unmatched:>9} "
                     f"{statistics.median(ious) if ious else float('nan'):>9.4f} "
                     f"{statistics.median(kp) if kp else float('nan'):>11.2f} "
                     f"{percentile(kp, 0.95) if kp else float('nan'):>10.2f}")
        print(line)


if __name__ == "__main__":
    main()
