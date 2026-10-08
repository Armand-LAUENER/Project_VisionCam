"""
convert_chokepoint.py — Séquences ChokePoint au format MOT17, et galerie de visages.

ChokePoint (NICTA, CC BY-NC 4.0, https://zenodo.org/record/815657) : caméras
au-dessus de portes, 800×600 à 30 i/s, personnes identifiées sur toutes les
caméras, position des deux yeux annotée image par image, et deux photos posées
par personne (`Still/Neutral`, `Still/Smile`). C'est le protocole de
VisionCam : enrôler à partir de photos, reconnaître dans la vidéo.

Pour chaque séquence-caméra (`P1E_S1_C1`…), écrit dans `<sortie>/<nom>/` :

    img1/000001.jpg …   liens vers les images, renumérotées de 1 à N (les
                        fichiers ChokePoint portent le numéro d'image, avec
                        quelques sauts entre deux passages)
    gt/gt_eyes.txt      image,personne,œil_g_x,œil_g_y,œil_d_x,œil_d_y (base 1)
    seqinfo.ini

Avec `--gallery`, écrit aussi `<galerie>/ID0001/{neutral,smile}.jpg` au format
de `known_faces/`, sans les personnes de `--hold-out` (pour mesurer aussi les
non enrôlés). Les photos (4000×2672) sont réduites à 1600 px de large.
Jamais dans le `known_faces/` de l'application.

Archives à extraire d'abord (chacune, puis chaque sous-archive par caméra,
dans son propre dossier) sous `<racine>` : `groundtruth/groundtruth/*.xml`,
`<séquence>/<séquence>_C<n>/<séquence>_C<n>/*.jpg`, `Still/{Neutral,Smile}`.

Depuis la racine du projet :

    uv run -m tools.convert_chokepoint --root ~/datasets/ChokePoint/raw \\
        --output ~/datasets/ChokePoint/mot \\
        --gallery /tmp/known_faces_chokepoint --hold-out 23 24 25 26 27
"""

import argparse
import glob
import os
import re
import sys

import cv2

from tools.record_sequence import write_seqinfo

FRAME = re.compile(r'<frame number="(\d+)"\s*(/?)>')
PERSON = re.compile(r'<person id="(\d+)">\s*<leftEye x="(\d+)" y="(\d+)"/>\s*'
                    r'<rightEye x="(\d+)" y="(\d+)"/>')
GALLERY_WIDTH = 1600


def parse_groundtruth(xml_text: str) -> dict:
    """{numéro d'image ChokePoint: [(personne, lx, ly, rx, ry)]}."""
    eyes: dict = {}
    for block in re.split(r'(?=<frame number=)', xml_text):
        m = FRAME.match(block)
        if not m:
            continue
        frame = int(m.group(1))
        eyes[frame] = [tuple(int(v) for v in p) for p in PERSON.findall(block)]
    return eyes


def convert(root: str, output: str, name: str) -> int:
    """Une séquence-caméra ; renvoie le nombre d'images."""
    sequence = name.rsplit("_", 1)[0]
    frames_dir = os.path.join(root, sequence, name, name)
    images = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not images:
        sys.exit(f"Aucune image dans {frames_dir}")
    numbers = [int(os.path.splitext(os.path.basename(p))[0]) for p in images]
    index = {number: i + 1 for i, number in enumerate(numbers)}

    seq_dir = os.path.join(output, name)
    img_dir = os.path.join(seq_dir, "img1")
    os.makedirs(img_dir, exist_ok=True)
    for i, path in enumerate(images, start=1):
        link = os.path.join(img_dir, f"{i:06d}.jpg")
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(path), link)

    with open(os.path.join(root, "groundtruth", "groundtruth", f"{name}.xml")) as f:
        eyes = parse_groundtruth(f.read())
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
    with open(os.path.join(seq_dir, "gt", "gt_eyes.txt"), "w") as out:
        for number in sorted(eyes):
            if number not in index:
                continue
            for pid, lx, ly, rx, ry in eyes[number]:
                out.write(f"{index[number]},{pid},{lx + 1},{ly + 1},{rx + 1},{ry + 1}\n")

    height, width = cv2.imread(images[0]).shape[:2]
    write_seqinfo(os.path.join(seq_dir, "seqinfo.ini"), name, 30, len(images), width, height)
    return len(images)


def build_gallery(root: str, gallery: str, hold_out: set[int]) -> list[str]:
    """known_faces/ ChokePoint : une entrée par personne, photos réduites."""
    names = []
    for path in sorted(glob.glob(os.path.join(root, "Still", "Neutral", "ID*.JPG"))):
        pid = int(os.path.basename(path)[2:6])
        if pid in hold_out:
            continue
        name = f"ID{pid:04d}"
        os.makedirs(os.path.join(gallery, name), exist_ok=True)
        for expression in ("Neutral", "Smile"):
            image = cv2.imread(os.path.join(root, "Still", expression, f"{name}.JPG"))
            if image is None:
                continue
            scale = GALLERY_WIDTH / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(gallery, name, f"{expression.lower()}.jpg"), image)
        names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="dossier des archives extraites")
    parser.add_argument("--output", required=True, help="dossier des séquences MOT17")
    parser.add_argument("--gallery", help="dossier known_faces/ à écrire depuis Still/")
    parser.add_argument("--hold-out", type=int, nargs="*", default=[],
                        help="personnes laissées hors de la galerie (non enrôlées)")
    parser.add_argument("names", nargs="*",
                        help="séquences-caméras (P1E_S1_C1 …) ; défaut : toutes celles extraites")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    names = args.names or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(root, "P*_S*", "P*_S*_C*"))
        if os.path.isdir(p))
    for name in names:
        count = convert(root, os.path.expanduser(args.output), name)
        print(f"{name} : {count} images")
    if args.gallery:
        enrolled = build_gallery(root, os.path.expanduser(args.gallery), set(args.hold_out))
        print(f"{len(enrolled)} personnes dans {args.gallery} "
              f"(hors galerie : {', '.join(map(str, args.hold_out)) or 'aucune'})")


if __name__ == "__main__":
    main()
