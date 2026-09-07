"""
bench_pose.py — Compare les deux sources d'estimation d'orientation.

Mesure, sur le même flux et les mêmes personnes, ce que donnent Mediapipe
(inférence CPU sur un crop) et la relecture des keypoints YOLO (déjà calculés
sur GPU) :

  - latence par appel : moyenne, médiane, p95
  - coût par frame : latence × nombre de personnes suivies
  - taux d'accord entre les deux verdicts, avec matrice de confusion
  - accord ventilé par taille de personne — c'est le point sensible, les
    keypoints YOLO étant estimés sur la frame entière là où Mediapipe travaille
    sur un crop zoomé, l'écart attendu est sur les personnes lointaines

Les deux estimateurs tournent sur les MÊMES frames et les MÊMES tracks : la
comparaison ne dépend pas d'un aléa de détection entre deux exécutions.

    python tools/bench_pose.py --frames 200
"""

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict

import cv2

import config
from core.face_body_tracker import FaceBodyTracker
from core.face_recognition import FaceRecognizer
from core.pose_estimation import PoseEstimator
from core.pose_from_keypoints import KeypointPoseEstimator

# Bornes de hauteur de corps (px) pour la ventilation par distance
BUCKETS = [("proche (>400px)", 400), ("moyen (200-400px)", 200), ("loin (<200px)", 0)]


def bucket_for(height):
    for label, low in BUCKETS:
        if height >= low:
            return label
    return BUCKETS[-1][0]


def fmt_ms(values):
    if not values:
        return "n/a"
    p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values)
    return (f"moy {statistics.mean(values):6.2f} ms | "
            f"med {statistics.median(values):6.2f} ms | p95 {p95:6.2f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=150, help="Frames à analyser (défaut : 150)")
    parser.add_argument("--source", default=config.CAMERA_SOURCE,
                        help="Source vidéo (défaut : config.CAMERA_SOURCE)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.CAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.CAM_READ_TIMEOUT_MS)
    if not cap.isOpened():
        sys.exit(f"Source vidéo inaccessible : {args.source}")

    recognizer = FaceRecognizer(config.KNOWN_FACES_DIR,
                                threshold=config.RECOGNITION_THRESHOLD,
                                cache_path=config.EMBEDDINGS_CACHE_PATH)
    tracker = FaceBodyTracker(recognizer)
    mediapipe_est = PoseEstimator()
    keypoint_est = KeypointPoseEstimator()

    mp_times, kp_times = [], []
    persons_per_frame = []
    agree = disagree = 0
    confusion = Counter()
    by_bucket = defaultdict(lambda: [0, 0])   # label -> [accords, total]
    frames_done = 0

    print(f"\nMesure sur {args.frames} frames — source : {args.source}\n")
    for frame_count in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            print(f"Flux interrompu après {frames_done} frames.")
            break

        persons = tracker.update(frame, frame_count)
        frames_done += 1
        persons_per_frame.append(len(persons))

        for person in persons:
            x1, y1, x2, y2 = person.body_bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            t0 = time.perf_counter()
            mp_verdict = mediapipe_est.estimate(crop)
            mp_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            kp_verdict = keypoint_est.estimate(person.pose_kps)
            kp_times.append((time.perf_counter() - t0) * 1000)

            confusion[(mp_verdict, kp_verdict)] += 1
            label = bucket_for(y2 - y1)
            by_bucket[label][1] += 1
            if mp_verdict == kp_verdict:
                agree += 1
                by_bucket[label][0] += 1
            else:
                disagree += 1

    cap.release()
    mediapipe_est.release()

    # ── Restitution ──────────────────────────────────────────────────────────
    total = agree + disagree
    if not total:
        sys.exit("Aucune personne détectée : impossible de comparer.")

    avg_persons = statistics.mean(persons_per_frame) if persons_per_frame else 0
    print(f"{frames_done} frames analysées, {total} observations, "
          f"{avg_persons:.2f} personne(s)/frame en moyenne\n")

    print("LATENCE PAR APPEL")
    print(f"  Mediapipe (CPU)     : {fmt_ms(mp_times)}")
    print(f"  Keypoints YOLO      : {fmt_ms(kp_times)}")
    if kp_times and statistics.mean(kp_times) > 0:
        print(f"  Rapport             : ×{statistics.mean(mp_times)/statistics.mean(kp_times):.0f}")

    print("\nCOUT PAR FRAME (latence × personnes suivies)")
    print(f"  Mediapipe           : {statistics.mean(mp_times)*avg_persons:6.2f} ms")
    print(f"  Keypoints YOLO      : {statistics.mean(kp_times)*avg_persons:6.2f} ms")
    print(f"  Economie            : {(statistics.mean(mp_times)-statistics.mean(kp_times))*avg_persons:6.2f} ms/frame")

    print(f"\nACCORD GLOBAL : {agree}/{total} ({100*agree/total:.1f} %)")

    print("\nACCORD PAR DISTANCE")
    for label, _ in BUCKETS:
        ok_count, seen = by_bucket[label]
        if seen:
            print(f"  {label:20s} : {ok_count}/{seen} ({100*ok_count/seen:.1f} %)")
        else:
            print(f"  {label:20s} : aucune observation")

    print("\nMATRICE DE CONFUSION (mediapipe → yolo), désaccords d'abord")
    for (mp_v, kp_v), n in sorted(confusion.items(), key=lambda kv: (kv[0][0] == kv[0][1], -kv[1])):
        flag = "  " if mp_v == kp_v else "≠ "
        print(f"  {flag}{str(mp_v):16s} → {str(kp_v):16s} : {n}")


if __name__ == "__main__":
    main()
