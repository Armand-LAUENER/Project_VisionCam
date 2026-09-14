"""
bench_tracker.py — Compare les deux backends d'association de tracks.

Fait tourner `deep_sort_realtime` et `deepsort-rs` sur EXACTEMENT les mêmes
détections YOLO, frame par frame, et rapporte deux choses distinctes :

  - parité : les deux backends produisent-ils les mêmes pistes (mêmes IDs,
    mêmes boîtes) ? Sans ça, un gain de vitesse ne veut rien dire.
  - vitesse : temps par frame de chaque backend, et part de ce temps qui va à
    l'embedder MobileNetV2 — commun aux deux, donc incompressible. C'est ce qui
    borne le gain réellement observable dans VisionCam (loi d'Amdahl).

YOLO ne tourne qu'une fois par frame : la comparaison ne dépend pas d'un aléa
de détection entre deux exécutions.

Depuis la racine du projet (le `-m` met celle-ci sur le sys.path) :

    python -m tools.bench_tracker --video ma_video.mp4 --frames 300
    python -m tools.bench_tracker --webcam --frames 200
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

import config
from core.tracker_backends import PythonDeepSortBackend, RustDeepSortBackend

# Écart de boîte au-delà duquel deux pistes ne sont plus considérées identiques.
# 1e-3 px : on compare deux implémentations du même filtre, pas deux trackers.
BOX_TOLERANCE = 1e-3


def open_source(video, webcam):
    source = config.CAMERA_SOURCE if webcam else video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Source vidéo illisible : {source}")
    return cap


def detections_from(yolo, frame):
    """Mêmes détections que `FaceBodyTracker._detect_bodies`, sans les
    keypoints : aucun des deux backends ne les utilise pour l'association."""
    results = yolo.predict(
        frame,
        classes=[0],
        conf=config.YOLO_CONF_THRESHOLD,
        device=0,
        verbose=False,
    )
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), "person", None)
            )
    return detections


def snapshot(tracks):
    """{track_id: boîte} des pistes confirmées — ce que le pipeline consomme."""
    return {
        int(t.track_id): np.asarray(t.to_ltrb(), dtype=float)
        for t in tracks
        if t.is_confirmed()
    }


def compare(python_tracks, rust_tracks):
    """Retourne (identiques, écart_max_px, motif si divergence)."""
    ref, rust = snapshot(python_tracks), snapshot(rust_tracks)

    if set(ref) != set(rust):
        only_ref = sorted(set(ref) - set(rust))
        only_rust = sorted(set(rust) - set(ref))
        return False, float("inf"), f"IDs python-seuls={only_ref} rust-seuls={only_rust}"

    if not ref:
        return True, 0.0, ""

    gap = max(float(np.abs(ref[tid] - rust[tid]).max()) for tid in ref)
    if gap > BOX_TOLERANCE:
        return False, gap, f"écart de boîte {gap:.4f} px"
    return True, gap, ""


def summarise(label, samples):
    if not samples:
        return f"  {label:<28} (aucune mesure)"
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return (
        f"  {label:<28} médiane {statistics.median(samples) * 1e3:7.3f} ms"
        f"   p95 {p95 * 1e3:7.3f} ms"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Fichier vidéo à rejouer")
    source.add_argument("--webcam", action="store_true", help="Utiliser config.CAMERA_SOURCE")
    parser.add_argument("--frames", type=int, default=300, help="Frames à traiter")
    parser.add_argument("--warmup", type=int, default=50, help="Frames ignorées dans les temps")
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Chargement YOLO-Pose ({config.YOLO_MODEL})...")
    yolo = YOLO(config.YOLO_MODEL)

    print("Instanciation des deux backends (deux embedders MobileNetV2 chargés)...")
    backends = {
        "python (deep_sort_realtime)": PythonDeepSortBackend(),
        "rust (deepsort-rs)": RustDeepSortBackend(),
    }

    cap = open_source(args.video, args.webcam)
    timings: dict[str, list[float]] = {name: [] for name in backends}
    detections_per_frame = []
    divergences = []
    frames_compared = 0

    print(f"\nTraitement de {args.frames} frames ({args.warmup} de warm-up)...")
    for frame_index in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            print(f"  flux terminé à la frame {frame_index}")
            break

        detections = detections_from(yolo, frame)
        outputs = {}
        for name, backend in backends.items():
            start = time.perf_counter()
            outputs[name] = backend.update(detections, frame)
            elapsed = time.perf_counter() - start
            if frame_index >= args.warmup:
                timings[name].append(elapsed)

        if frame_index >= args.warmup:
            detections_per_frame.append(len(detections))
            frames_compared += 1
            same, _, reason = compare(*outputs.values())
            if not same:
                divergences.append((frame_index, reason))

    cap.release()

    print("\n─── Parité ────────────────────────────────────────────────────")
    if not frames_compared:
        print("  aucune frame comparée")
    elif not divergences:
        print(f"  {frames_compared} frames : pistes identiques (IDs et boîtes)")
    else:
        rate = 100 * (frames_compared - len(divergences)) / frames_compared
        print(f"  {frames_compared - len(divergences)}/{frames_compared} frames identiques ({rate:.1f} %)")
        for frame_index, reason in divergences[:10]:
            print(f"    frame {frame_index} : {reason}")
        if len(divergences) > 10:
            print(f"    ... et {len(divergences) - 10} autres")

    print("\n─── Vitesse (tracker complet : embedder + association) ────────")
    if detections_per_frame:
        print(
            f"  {statistics.mean(detections_per_frame):.1f} personnes/frame en moyenne"
            f" (max {max(detections_per_frame)})"
        )
    for name in backends:
        print(summarise(name, timings[name]))

    medians = {name: statistics.median(t) for name, t in timings.items() if t}
    if len(medians) == 2:
        python_ms, rust_ms = (medians[n] * 1e3 for n in backends)
        print(
            f"\n  Étape tracker, python → rust : {python_ms:.3f} ms → {rust_ms:.3f} ms"
            f"  ({python_ms - rust_ms:+.3f} ms gagnés, ×{python_ms / rust_ms:.2f})"
        )
        print(
            "  Les deux chiffres incluent le même embedder MobileNetV2 sur les mêmes\n"
            "  crops. L'écart ne mesure pas pour autant l'association seule : côté\n"
            "  rust, il inclut aussi la conversion des embeddings en float32 et le\n"
            "  passage des tableaux vers le crate."
        )


if __name__ == "__main__":
    main()
