"""
pose_threshold_study.py — Choisit le seuil de taille sous lequel l'orientation
issue des keypoints YOLO n'est plus fiable.

Le constat de départ : sur des personnes lointaines, KeypointPoseEstimator rend
un verdict là où Mediapipe déclare forfait, et les deux divergent fortement.
Plutôt que de fixer POSE_MIN_SHOULDER_DIST_PX au jugé, on mesure l'accord entre
les deux méthodes en fonction de l'écart d'épaules, et on lit le seuil sur la
courbe.

Mediapipe sert de référence faute de mieux : il travaille sur un crop zoomé,
donc dispose de plus d'information que YOLO sur les petits sujets. Les cas où il
ne rend aucun verdict sont comptés à part — ce ne sont pas des désaccords, ce
sont des absences de référence.

    python tools/pose_threshold_study.py videos/*.mp4 --frames 30

Écrit aussi un CSV des observations brutes pour rejouer l'analyse sans
recalculer les inférences.
"""

import os

# Avant numpy : sans ça, les threads OpenBLAS se disputent le CPU avec torch
# et ralentissent le pipeline (cf. app.py).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import sys
from collections import defaultdict

import cv2

import config
from core.face_body_tracker import FaceBodyTracker
from core.face_recognition import FaceRecognizer
from core.pose_estimation import PoseEstimator
from core.pose_from_keypoints import KeypointPoseEstimator

# Bornes basses des tranches d'écart d'épaules, en pixels
SPAN_BINS = [0, 10, 15, 20, 25, 30, 40, 60, 100]

# Accord minimal exigé pour considérer une tranche exploitable
TARGET_AGREEMENT = 0.90


def bin_for(span):
    chosen = SPAN_BINS[0]
    for low in SPAN_BINS:
        if span >= low:
            chosen = low
    return chosen


def bin_label(low):
    idx = SPAN_BINS.index(low)
    high = SPAN_BINS[idx + 1] if idx + 1 < len(SPAN_BINS) else None
    return f"{low}-{high} px" if high else f"{low}+ px"


def collect(video, frames, rows):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"  {video} : illisible, ignoré")
        return

    recognizer = FaceRecognizer(config.KNOWN_FACES_DIR,
                                threshold=config.RECOGNITION_THRESHOLD,
                                cache_path=config.EMBEDDINGS_CACHE_PATH)
    tracker = FaceBodyTracker(recognizer)
    mediapipe_est = PoseEstimator()
    keypoint_est = KeypointPoseEstimator()

    seen = 0
    for frame_count in range(frames):
        ok, frame = cap.read()
        if not ok:
            break

        for person in tracker.update(frame, frame_count):
            if not person.pose_kps:
                continue
            x1, y1, x2, y2 = person.body_bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            span = abs(person.pose_kps[5][0] - person.pose_kps[6][0])
            rows.append({
                'video': video.rsplit('/', 1)[-1],
                'shoulder_span_px': round(span, 1),
                'body_height_px': y2 - y1,
                'mediapipe': mediapipe_est.estimate(crop),
                'yolo': keypoint_est.estimate(person.pose_kps),
            })
            seen += 1

    cap.release()
    mediapipe_est.release()
    print(f"  {video.rsplit('/', 1)[-1]:16s} : {seen} observations")


def report(rows):
    per_bin = defaultdict(lambda: {'n': 0, 'mp_silent': 0, 'compared': 0, 'agree': 0})

    for r in rows:
        b = per_bin[bin_for(r['shoulder_span_px'])]
        b['n'] += 1
        if r['mediapipe'] is None:
            b['mp_silent'] += 1
            continue
        b['compared'] += 1
        if r['mediapipe'] == r['yolo']:
            b['agree'] += 1

    print("\nACCORD PAR ECART D'EPAULES")
    print(f"  {'tranche':<12} {'obs':>6} {'mediapipe muet':>16} {'comparables':>12} {'accord':>10}")
    usable = []
    for low in SPAN_BINS:
        b = per_bin.get(low)
        if not b or not b['n']:
            continue
        silent_pct = 100 * b['mp_silent'] / b['n']
        if b['compared']:
            rate = b['agree'] / b['compared']
            print(f"  {bin_label(low):<12} {b['n']:>6} {silent_pct:>15.0f}% "
                  f"{b['compared']:>12} {100*rate:>9.1f}%")
            if rate >= TARGET_AGREEMENT and b['compared'] >= 10:
                usable.append(low)
        else:
            print(f"  {bin_label(low):<12} {b['n']:>6} {silent_pct:>15.0f}% "
                  f"{0:>12} {'n/a':>10}")

    print(f"\nSeuil visé : {100*TARGET_AGREEMENT:.0f} % d'accord, au moins 10 comparaisons par tranche.")
    if usable:
        print(f"Tranches qui l'atteignent : {', '.join(bin_label(x) for x in usable)}")
        print(f"→ POSE_MIN_SHOULDER_DIST_PX = {min(usable)}")
    else:
        print("Aucune tranche n'atteint la cible — élargir l'échantillon avant de trancher.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", help="Vidéos à analyser")
    parser.add_argument("--frames", type=int, default=30, help="Frames par vidéo (défaut : 30)")
    parser.add_argument("--csv", default=None, help="Chemin du CSV des observations brutes")
    args = parser.parse_args()

    rows = []
    print(f"\nCollecte sur {len(args.videos)} vidéo(s), {args.frames} frames chacune")
    for video in args.videos:
        collect(video, args.frames, rows)

    if not rows:
        sys.exit("Aucune observation collectée.")

    print(f"\n{len(rows)} observations au total")
    report(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nObservations brutes : {args.csv}")


if __name__ == "__main__":
    main()
