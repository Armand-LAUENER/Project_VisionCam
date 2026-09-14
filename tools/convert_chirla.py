"""
convert_chirla.py — Convertit des vidéos CHIRLA en séquences au format MOT17.

CHIRLA (bureau filmé à 30 i/s, identités annotées sur chaque image, CC-BY 4.0 :
https://huggingface.co/datasets/bdager/CHIRLA) fournit une vidéo et un JSON
par caméra. Pour chaque vidéo passée en argument, écrit dans
`<sortie>/<séquence>_<caméra>/` :

    img1/000001.jpg …   images de la vidéo
    gt/gt.txt           annotations CHIRLA, format gt MOT16 (pixels base 1)
    det/det.txt         détections YOLO (config.YOLO_MODEL), comme le pipeline
    seqinfo.ini

CHIRLA ne fournit pas de détections publiques : les détections sont celles de
VisionCam, ce qui mesure l'association dans les conditions du pipeline.

Téléchargement (vidéos et annotations des caméras voulues) :

    uvx --from huggingface_hub hf download bdager/CHIRLA --repo-type dataset \\
        --local-dir ~/datasets/CHIRLA --include "annotations/*" \\
        --include "videos/seq_025/camera_2_*"

Depuis la racine du projet :

    uv run -m tools.convert_chirla --root ~/datasets/CHIRLA --output ~/datasets/CHIRLA/mot \\
        seq_025/camera_2 seq_026/camera_3
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import glob
import json
import sys

import cv2

from tools.record_sequence import detect, write_seqinfo

FPS = 30


def find(root, kind, name, extension):
    """Fichier CHIRLA d'une caméra : les noms portent l'horodatage d'enregistrement."""
    sequence, camera = name.split("/")
    matches = glob.glob(os.path.join(root, kind, sequence, f"{camera}_*{extension}"))
    if len(matches) != 1:
        sys.exit(f"{kind} de {name} : {len(matches)} fichier(s) trouvé(s) dans {root}")
    return matches[0]


def write_gt(annotations, path):
    """JSON CHIRLA {image: [{id, BboxP: [x1, y1, x2, y2]}]} → gt MOT16, pixels base 1."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boxes = 0
    with open(path, "w") as out:
        for frame in sorted(annotations, key=int):
            for person in annotations[frame]:
                x1, y1, x2, y2 = person["BboxP"]
                out.write(f"{int(frame)},{person['id']},{x1 + 1},{y1 + 1},"
                          f"{x2 - x1},{y2 - y1},1,1,1\n")
                boxes += 1
    return boxes


def extract_frames(video, img_dir):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Vidéo illisible : {video}")
    os.makedirs(img_dir, exist_ok=True)
    count, size = 0, (0, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        size = (frame.shape[1], frame.shape[0])
        cv2.imwrite(os.path.join(img_dir, f"{count:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()
    return count, *size


def convert(root, output, name):
    sequence, camera = name.split("/")
    seq_dir = os.path.join(output, f"{sequence}_{camera}")
    with open(find(root, "annotations", name, ".json")) as f:
        annotations = json.load(f)

    count, width, height = extract_frames(find(root, "videos", name, ".avi"),
                                           os.path.join(seq_dir, "img1"))
    annotated = max(int(k) for k in annotations)
    # Un écart d'une ou deux images en fin de vidéo arrive (dernière image
    # incomplète) ; au-delà, les annotations seraient décalées.
    if abs(count - annotated) > 2:
        sys.exit(f"{name} : {count} images dans la vidéo, {annotated} annotées")
    annotations = {k: v for k, v in annotations.items() if int(k) <= count}

    boxes = write_gt(annotations, os.path.join(seq_dir, "gt", "gt.txt"))
    write_seqinfo(os.path.join(seq_dir, "seqinfo.ini"), f"{sequence}_{camera}",
                  FPS, count, width, height)
    os.makedirs(os.path.join(seq_dir, "det"), exist_ok=True)
    detect(seq_dir, count, os.path.join(seq_dir, "det", "det.txt"))
    print(f"{seq_dir} : {count} images {width}x{height}, {boxes} boîtes annotées", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="dossier CHIRLA téléchargé")
    parser.add_argument("--output", required=True, help="dossier des séquences MOT")
    parser.add_argument("cameras", nargs="+", help="seq_XXX/camera_N")
    args = parser.parse_args()

    root, output = os.path.expanduser(args.root), os.path.expanduser(args.output)
    for name in args.cameras:
        convert(root, output, name)


if __name__ == "__main__":
    main()
