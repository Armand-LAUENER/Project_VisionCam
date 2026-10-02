"""
enroll_from_sequence.py — Enrôle les identités d'une séquence annotée (MOT17).

Pour chaque identité de la vérité terrain (`gt/gt.txt`) des séquences
données, écrit `--photos` crops de tête répartis dans le temps dans
`<sortie>/<préfixe><identité>/`, au format de `known_faces/`. Les crops sont
ceux de l'application (YOLOv8-Pose + FaceBodyTracker.head_crop_box, comme
tools/bench_face.py), et seulement ceux où InsightFace trouve un visage d'au
moins RECOGNITION_MIN_FACE_PX : l'application fait la moyenne des embeddings
de toutes les photos d'une personne, et un visage minuscule ou flou la
dégrade (mesuré sur CHIRLA : la bonne personne plafonnait alors à 0,3-0,4 de
similarité, sous le seuil).

Sert à mesurer l'identité de bout en bout (tools/eval_identity.py) sur un jeu
public dont les identités sont communes aux séquences, comme CHIRLA : enrôler
sur certaines séquences, rejouer les autres. Écrire dans un dossier à part,
jamais dans le `known_faces/` de l'application.

Depuis la racine du projet :

    uv run -m tools.enroll_from_sequence --output /tmp/known_faces_chirla \\
        ~/datasets/CHIRLA/mot/seq_004_camera_2 ~/datasets/CHIRLA/mot/seq_020_camera_4
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from collections import defaultdict

import cv2
import numpy as np

import config

DEFAULT_PREFIX = "id_"


def pick_spread(samples, count):
    """`count` éléments répartis régulièrement dans `samples` (déjà triés), sans doublon."""
    if len(samples) <= count:
        return list(samples)
    picks = np.linspace(0, len(samples) - 1, count).round().astype(int)
    return [samples[i] for i in sorted(set(picks))]


def usable_face(face_app, crop) -> bool:
    """Le plus grand visage du crop, celui que retient l'enrôlement, est-il assez grand ?"""
    faces = face_app.get(crop)
    if not faces:
        return False
    x1, y1, x2, y2 = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).bbox
    return min(x2 - x1, y2 - y1) >= config.RECOGNITION_MIN_FACE_PX


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequences", nargs="+", help="séquences MOT17 avec gt/gt.txt")
    parser.add_argument("--output", required=True, help="dossier au format known_faces/ à créer")
    parser.add_argument("--photos", type=int, default=10, help="crops par identité")
    parser.add_argument("--step", type=int, default=15, help="une image sur N")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"préfixe du nom de chaque identité (défaut : {DEFAULT_PREFIX})")
    args = parser.parse_args()

    output = os.path.expanduser(args.output)
    if os.path.realpath(output) == os.path.realpath(config.KNOWN_FACES_DIR):
        raise SystemExit("Refusé : --output est le known_faces/ de l'application.")

    from ultralytics import YOLO

    from core.face_recognition import FaceRecognizer
    from tools.bench_face import extract_crops

    face_app = FaceRecognizer._new_face_analysis(None)
    face_app.prepare(ctx_id=0, det_size=config.INSIGHTFACE_DET_SIZE)
    yolo = YOLO(config.YOLO_MODEL, task="pose")
    by_id = defaultdict(list)
    rejected = 0
    for seq in args.sequences:
        seq_dir = os.path.expanduser(seq)
        name = os.path.basename(seq_dir.rstrip("/"))
        for track_id, frame, crop in extract_crops(yolo, seq_dir, args.step):
            if not usable_face(face_app, crop):
                rejected += 1
                continue
            by_id[track_id].append((name, frame, crop))
    print(f"{rejected} crops écartés (pas de visage ≥ {config.RECOGNITION_MIN_FACE_PX} px)")

    for track_id, samples in sorted(by_id.items()):
        person_dir = os.path.join(output, f"{args.prefix}{track_id}")
        os.makedirs(person_dir, exist_ok=True)
        picked = pick_spread(samples, args.photos)
        for seq_name, frame, crop in picked:
            cv2.imwrite(os.path.join(person_dir, f"{seq_name}_{frame:06d}.jpg"), crop)
        print(f"{args.prefix}{track_id} : {len(picked)} crops (sur {len(samples)})")
    print(f"{len(by_id)} identités écrites dans {output}")


if __name__ == "__main__":
    main()
