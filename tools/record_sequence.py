"""
record_sequence.py — Enregistre une séquence de test au format MOT17.

Capture la caméra configurée (config.CAMERA_SOURCE, donc le pont webcam
Windows si USE_LOCAL_CAM=false) ou une vidéo, puis écrit :

    <sortie>/img1/000001.jpg …   images, à la cadence demandée
    <sortie>/det/det.txt         détections YOLO (config.YOLO_MODEL), format MOT
    <sortie>/seqinfo.ini         nom, cadence, taille, nombre d'images

La vérité terrain (`gt/gt.txt`) se construit ensuite avec
tools/annotate_sequence.py, puis la séquence se passe à tools/sweep_deepsort.py.

Scènes utiles pour régler DeepSORT : deux à quatre personnes qui se croisent,
sortent du champ et reviennent, passent devant la caméra, se retournent.

Depuis la racine du projet :

    uv run -m tools.record_sequence --seconds 90 --output ~/datasets/visioncam/croisements-01
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import configparser
import sys
import time

import cv2

import config


def capture(source, seconds, fps, img_dir):
    """Écrit les images à `fps` pendant `seconds` ; retourne (nombre, largeur, hauteur)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Source vidéo illisible : {source}")
    is_file = isinstance(source, str) and os.path.isfile(source)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, round(source_fps / fps)) if is_file else 1
    period = 1.0 / fps
    count = read = 0
    size = (0, 0)
    next_shot = time.monotonic()
    print(f"Enregistrement de {seconds} s à {fps} images/s… (Ctrl+C pour arrêter)")
    try:
        while count < seconds * fps:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            if is_file:
                if (read - 1) % step:
                    continue
            else:
                # Flux temps réel : on lit en continu pour vider le tampon,
                # et on ne garde qu'une image par période.
                now = time.monotonic()
                if now < next_shot:
                    continue
                next_shot += period
            count += 1
            size = (frame.shape[1], frame.shape[0])
            cv2.imwrite(os.path.join(img_dir, f"{count:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if count % fps == 0:
                print(f"  {count // fps} s", flush=True)
    except KeyboardInterrupt:
        print("  arrêt demandé")
    cap.release()
    return count, *size


def detect(img_dir, count, det_path):
    """Détections YOLO au format MOT : pixels en base 1, comme les det.txt de MOT17."""
    from ultralytics import YOLO

    from core.face_body_tracker import FaceBodyTracker

    yolo = YOLO(config.YOLO_MODEL, task="pose")
    with open(det_path, "w") as out:
        for frame_id in range(1, count + 1):
            image = cv2.imread(os.path.join(img_dir, f"{frame_id:06d}.jpg"))
            for result in yolo.predict(image, classes=[0], conf=config.YOLO_CONF_THRESHOLD,
                                       device=0, verbose=False):
                for (left, top, width, height), conf, *_ in FaceBodyTracker._parse_result(result):
                    out.write(f"{frame_id},-1,{left + 1:.2f},{top + 1:.2f},"
                              f"{width:.2f},{height:.2f},{conf:.4f},-1,-1,-1\n")


def write_seqinfo(path, name, fps, count, width, height):
    info = configparser.ConfigParser()
    info.optionxform = str
    info["Sequence"] = {
        "name": name, "imDir": "img1", "frameRate": str(fps), "seqLength": str(count),
        "imWidth": str(width), "imHeight": str(height), "imExt": ".jpg",
    }
    with open(path, "w") as f:
        info.write(f, space_around_delimiters=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, help="dossier de la séquence à créer")
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--fps", type=int, default=15,
                        help="cadence enregistrée ; proche de celle du pipeline")
    parser.add_argument("--video", help="fichier vidéo au lieu de la caméra configurée")
    args = parser.parse_args()

    output = os.path.expanduser(args.output)
    img_dir = os.path.join(output, "img1")
    if os.path.exists(img_dir) and os.listdir(img_dir):
        sys.exit(f"{img_dir} existe déjà et n'est pas vide : choisir un autre --output")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(output, "det"), exist_ok=True)

    source = args.video or config.CAMERA_SOURCE
    count, width, height = capture(source, args.seconds, args.fps, img_dir)
    if not count:
        sys.exit("Aucune image enregistrée.")
    write_seqinfo(os.path.join(output, "seqinfo.ini"), os.path.basename(output.rstrip("/")),
                  args.fps, count, width, height)
    print(f"{count} images {width}x{height}. Détections YOLO…")
    detect(img_dir, count, os.path.join(output, "det", "det.txt"))
    print(f"Séquence prête : {output}\n"
          f"Étape suivante : uv run -m tools.annotate_sequence {output}")


if __name__ == "__main__":
    main()
