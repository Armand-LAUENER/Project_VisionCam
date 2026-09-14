"""
recognition_threshold_study.py — Choisit RECOGNITION_THRESHOLD sur des visages annotés.

Simule l'usage de VisionCam sur des séquences au format MOT17 dont les
identités sont les mêmes d'une séquence à l'autre (CHIRLA, par exemple) :

1. Enrôlement : pour chaque identité des séquences `--gallery`, un embedding
   moyen normalisé de `--photos` visages répartis dans le temps, comme
   FaceRecognizer.enroll_person_average.
2. Reconnaissance : chaque visage des séquences `--probe`, prises à une autre
   période (autres jours, tenues, éclairage), est comparé à la base avec la
   règle de FaceRecognizer._identify (meilleur score cosinus, puis seuil).

Pour chaque seuil, trois issues pour une personne enrôlée — bon nom, mauvais
nom, « Inconnu » — et deux pour une personne absente de la base : rejetée ou
reconnue à tort. Le mauvais nom est l'erreur qui compte : il attache une
identité à la mauvaise personne.

Les visages viennent du même InsightFace que l'application (modèle et det_size
de la configuration). Un visage est attribué à une personne annotée quand son
centre tombe dans le haut de sa boîte et qu'il est le seul dans ce cas.

Limite : chaque visage est jugé seul. L'application vote sur plusieurs images
par piste (FaceBodyTracker.VOTE_WINDOW), ce qui réduit encore les erreurs
isolées.

Depuis la racine du projet :

    uv run -m tools.recognition_threshold_study \\
        --gallery ~/datasets/CHIRLA/mot/seq_004_camera_2 ~/datasets/CHIRLA/mot/seq_020_camera_4 \\
        --probe ~/datasets/CHIRLA/mot/seq_025_camera_2 ~/datasets/CHIRLA/mot/seq_026_camera_3
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import statistics
from collections import defaultdict

import cv2
import numpy as np

import config
from tools import eval_mot

# Part haute de la boîte d'une personne où doit tomber le centre de son visage.
HEAD_FRACTION = 0.35


def load_gt_boxes(seq_dir):
    """{image: [(identité, x1, y1, x2, y2)]}, pixels en base 0."""
    boxes = defaultdict(list)
    with open(os.path.join(seq_dir, "gt", "gt.txt")) as f:
        for line in f:
            frame, track_id, left, top, width, height = (float(v) for v in line.split(",")[:6])
            x1, y1 = left - 1, top - 1
            boxes[int(frame)].append((int(track_id), x1, y1, x1 + width, y1 + height))
    return boxes


def extract_faces(face_app, seq_dir, step, min_face):
    """[(identité, image, embedding normalisé, taille du visage px)] d'une séquence."""
    faces = []
    gt = load_gt_boxes(seq_dir)
    for frame in range(1, eval_mot.sequence_length(seq_dir) + 1, step):
        if not gt.get(frame):
            continue
        image = cv2.imread(eval_mot.frame_path(seq_dir, frame))
        detected = [f for f in face_app.get(image)
                    if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= min_face]
        for track_id, x1, y1, x2, y2 in gt[frame]:
            head_bottom = y1 + HEAD_FRACTION * (y2 - y1)
            inside = [f for f in detected
                      if x1 <= (f.bbox[0] + f.bbox[2]) / 2 <= x2
                      and y1 <= (f.bbox[1] + f.bbox[3]) / 2 <= head_bottom]
            if len(inside) != 1:
                continue
            face = inside[0]
            embedding = face.embedding / np.linalg.norm(face.embedding)
            size = float(min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]))
            faces.append((track_id, frame, embedding, size))
    return faces


def enroll(faces, photos):
    """{identité: embedding moyen normalisé} de `photos` visages répartis dans le temps."""
    by_id = defaultdict(list)
    for track_id, frame, embedding, _size in faces:
        by_id[track_id].append((frame, embedding))
    gallery = {}
    for track_id, samples in by_id.items():
        if len(samples) < photos:
            continue
        samples.sort(key=lambda s: s[0])
        picks = np.linspace(0, len(samples) - 1, photos).round().astype(int)
        mean = np.mean([samples[i][1] for i in picks], axis=0)
        gallery[track_id] = mean / np.linalg.norm(mean)
    return gallery


def outcomes(gallery, probes, thresholds):
    ids = sorted(gallery)
    matrix = np.stack([gallery[i] for i in ids])
    known, unknown = [], []
    for track_id, _frame, embedding, _size in probes:
        scores = matrix @ embedding
        best = int(np.argmax(scores))
        (known if track_id in gallery else unknown).append(
            (ids[best] == track_id, float(scores[best]),
             float(scores[ids.index(track_id)]) if track_id in gallery else None))
    rows = []
    for t in thresholds:
        right = sum(ok and s >= t for ok, s, _ in known)
        wrong = sum((not ok) and s >= t for ok, s, _ in known)
        false_accept = sum(s >= t for _, s, _ in unknown)
        rows.append((t, right, wrong, len(known) - right - wrong, false_accept))
    return known, unknown, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gallery", nargs="+", required=True, help="séquences d'enrôlement")
    parser.add_argument("--probe", nargs="+", required=True, help="séquences de test")
    parser.add_argument("--step", type=int, default=15, help="une image sur N")
    parser.add_argument("--min-face", type=int, default=24, help="taille minimale du visage (px)")
    parser.add_argument("--photos", type=int, default=5, help="photos par personne enrôlée")
    args = parser.parse_args()

    from core.face_recognition import FaceRecognizer

    face_app = FaceRecognizer._new_face_analysis(None)
    face_app.prepare(ctx_id=0, det_size=config.INSIGHTFACE_DET_SIZE)

    gallery_faces = [f for s in args.gallery
                     for f in extract_faces(face_app, os.path.expanduser(s), args.step, args.min_face)]
    probe_faces = [f for s in args.probe
                   for f in extract_faces(face_app, os.path.expanduser(s), args.step, args.min_face)]
    gallery = enroll(gallery_faces, args.photos)

    thresholds = [round(0.25 + 0.05 * i, 2) for i in range(11)]
    known, unknown, rows = outcomes(gallery, probe_faces, thresholds)
    probe_ids = sorted({f[0] for f in probe_faces})
    print(f"\nBase : {len(gallery)} personnes ({args.photos} photos chacune) — "
          f"visages de test : {len(known)} de personnes enrôlées, "
          f"{len(unknown)} de personnes absentes de la base "
          f"({sorted(set(probe_ids) - set(gallery))})")
    print(f"Taille des visages de test : médiane "
          f"{statistics.median(f[3] for f in probe_faces):.0f} px")

    own = [s for _, _, s in known]
    impostor_best = [s for ok, s, _ in known if not ok] + [s for _, s, _ in unknown]
    print(f"Score de la bonne personne : médiane {statistics.median(own):.2f}, "
          f"5e centile {np.percentile(own, 5):.2f}")
    if impostor_best:
        print(f"Meilleur score d'une autre personne : médiane "
              f"{statistics.median(impostor_best):.2f}, 95e centile "
              f"{np.percentile(impostor_best, 95):.2f}")

    print(f"\n{'seuil':>6} │ {'bon nom':>8} {'mauvais nom':>12} {'Inconnu':>8} │ "
          f"{'absent reconnu à tort':>22}")
    for t, right, wrong, rejected, false_accept in rows:
        n = max(1, len(known))
        fa = f"{false_accept / len(unknown):.1%}" if unknown else "—"
        print(f"{t:>6.2f} │ {right / n:>8.1%} {wrong / n:>12.1%} {rejected / n:>8.1%} │ {fa:>22}")
    print(f"\nSeuil actuel : RECOGNITION_THRESHOLD = {config.RECOGNITION_THRESHOLD}")


if __name__ == "__main__":
    main()
