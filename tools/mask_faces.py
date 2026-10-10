"""
mask_faces.py — Copie d'une séquence MOT17 où les visages portent un masque.

Pour mesurer le maintien du nom quand le visage est caché (masque, cagoule ;
cf. BODY_KEEP_SIMILARITY), sans jeu public de personnes masquées : chaque
visage que SCRFD trouve dans la boîte annotée d'une personne (haut de la boîte)
reçoit un masque chirurgical, de juste sous les yeux au menton, pendant les
MASKED_S dernières secondes de chaque cycle de CYCLE_S. Les premières
secondes du cycle laissent le visage visible : la personne peut être nommée,
puis se masque, comme devant la webcam. Les yeux restent visibles, comme avec
un vrai masque : SCRFD trouve encore le visage, ArcFace ne le reconnaît plus.

Sortie : `<sortie>/img1/*.jpg`, `gt/` et `seqinfo.ini` recopiés, et
`masked.txt` (image,identité) des personnes-images en période masquée.

Depuis la racine du projet :

    uv run -m tools.mask_faces ~/datasets/CHIRLA/mot/seq_026_camera_3 \\
        --output /tmp/seq_026_camera_3_masked --exclude 9 14
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import shutil

import cv2
import numpy as np

from tools.eval_identity import _sequence_fps, load_gt

CYCLE_S = 30.0
MASKED_S = 20.0
MASK_BGR = (215, 190, 140)     # bleu clair de masque chirurgical
HEAD_PART = 0.35               # le visage doit être dans ce haut de la boîte
DET_SIZE = 640                 # image entière : visages de 30 à 60 px


def masked_now(frame: int, fps: float) -> bool:
    return (frame / fps) % CYCLE_S >= CYCLE_S - MASKED_S


def draw_mask(image: np.ndarray, bbox, kps) -> None:
    """Masque de juste sous les yeux au bas de la boîte SCRFD, sur sa largeur."""
    x1, _, x2, y2 = bbox
    eyes_y = (kps[0][1] + kps[1][1]) / 2
    top = eyes_y + 0.35 * (kps[2][1] - eyes_y)
    bottom = y2 + 0.05 * (y2 - top)
    cv2.rectangle(image, (int(x1), int(top)), (int(x2), int(bottom)), MASK_BGR, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude", nargs="*", type=int, default=[],
                        help="identités jamais masquées (non enrôlées)")
    args = parser.parse_args()

    from core.face_recognition import FaceRecognizer

    seq_dir, out = os.path.expanduser(args.sequence), os.path.expanduser(args.output)
    os.makedirs(os.path.join(out, "img1"), exist_ok=True)
    shutil.copytree(os.path.join(seq_dir, "gt"), os.path.join(out, "gt"), dirs_exist_ok=True)
    shutil.copy(os.path.join(seq_dir, "seqinfo.ini"), out)

    face_app = FaceRecognizer._new_face_analysis(None)
    face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
    detector = face_app.det_model
    gt, fps = load_gt(seq_dir), _sequence_fps(seq_dir)
    images = sorted(os.listdir(os.path.join(seq_dir, "img1")))

    masked_rows, faces_masked = [], 0
    for index, name in enumerate(images, start=1):
        image = cv2.imread(os.path.join(seq_dir, "img1", name))
        people = [p for p in gt.get(index, []) if p[0] not in args.exclude]
        if people and masked_now(index, fps):
            bboxes, kpss = detector.detect(image, max_num=0, metric="default")
            for pid, x1, y1, x2, y2 in people:
                head_bottom = y1 + HEAD_PART * (y2 - y1)
                for bbox, kps in zip(bboxes, kpss):
                    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    if x1 <= cx <= x2 and y1 <= cy <= head_bottom:
                        draw_mask(image, bbox[:4], kps)
                        faces_masked += 1
                        break
                masked_rows.append(f"{index},{pid}")
        cv2.imwrite(os.path.join(out, "img1", name), image)
    with open(os.path.join(out, "masked.txt"), "w") as f:
        f.write("\n".join(masked_rows) + "\n")
    print(f"{len(images)} images, {len(masked_rows)} personnes-images en période masquée, "
          f"{faces_masked} visages masqués -> {out}")


if __name__ == "__main__":
    main()
