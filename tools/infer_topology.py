"""
infer_topology.py — Topologie d'un réseau de caméras déduite des annotations (roadmap 2.4).

CHIRLA ne fournit pas le plan de ses 7 caméras, mais ses identités sont
communes à toutes les caméras d'une séquence, filmées en même temps. Pour
chaque personne annotée, ses apparitions sur chaque caméra (suites d'images
sans trou de plus de GAP_S) donnent :

  - le recouvrement des champs : la même personne visible au même instant sur
    deux caméras (part des images-personnes de A aussi vues sur B) ;
  - les passages : une apparition qui se termine sur A, suivie sans
    recouvrement d'une apparition sur B dans les MAX_TRANSIT_S secondes, avec
    la distribution des temps de transit ;
  - les zones d'entrée et de sortie de chaque caméra : où, dans l'image, les
    apparitions commencent et finissent (pied de la boîte, grille 3 × 3 sur
    l'étendue des boîtes annotées, `observed_extent` : la taille des images
    n'est pas dans les annotations).

Écrit un YAML d'agrégats (aucune image, aucune trajectoire individuelle), lu
par core/topology.py.

Depuis la racine du projet :

    uv run -m tools.infer_topology --annotations ~/datasets/CHIRLA/annotations \\
        --output topologies/chirla.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import yaml

FPS = 30
GAP_S = 1.0
MAX_TRANSIT_S = 60.0
MIN_PASSAGES = 3
CAMERA = re.compile(r"(camera_\d+)_")
COLUMNS = ("gauche", "centre", "droite")
ROWS = ("haut", "milieu", "bas")


def load_sequence(seq_dir: str) -> dict:
    """{caméra: {image: [(identité, x1, y1, x2, y2)]}} d'une séquence CHIRLA."""
    cameras = {}
    for path in sorted(glob.glob(os.path.join(seq_dir, "camera_*.json"))):
        camera = CAMERA.match(os.path.basename(path)).group(1)
        with open(path) as f:
            raw = json.load(f)
        cameras[camera] = {int(frame): [(p["id"], *p["BboxP"]) for p in people]
                           for frame, people in raw.items()}
    return cameras


def appearances(frames: dict, gap: int) -> list[dict]:
    """Apparitions d'une caméra : identité, première et dernière image, pied de
    la boîte au début et à la fin."""
    by_id = defaultdict(list)
    for frame, people in frames.items():
        for pid, x1, y1, x2, y2 in people:
            by_id[pid].append((frame, ((x1 + x2) / 2, y2)))
    result = []
    for pid, points in by_id.items():
        points.sort()
        start = points[0]
        previous = points[0]
        for point in points[1:] + [None]:
            if point is None or point[0] - previous[0] > gap:
                result.append({"id": pid, "start": start[0], "end": previous[0],
                               "foot_start": start[1], "foot_end": previous[1]})
                if point is not None:
                    start = point
            if point is not None:
                previous = point
    return result


def zone(foot, width, height) -> str:
    """Case d'une grille 3 × 3 de l'image (« bas-gauche »…)."""
    col = COLUMNS[min(2, int(3 * foot[0] / width))]
    row = ROWS[min(2, int(3 * foot[1] / height))]
    return f"{row}-{col}"


def infer(annotation_root: str) -> dict:
    overlap_frames = Counter()      # (A, B) → images-personnes de A aussi vues sur B
    person_frames = Counter()       # A → images-personnes
    transits = defaultdict(list)    # (A, B) → secondes
    entries = defaultdict(Counter)
    exits = defaultdict(Counter)
    sizes = defaultdict(lambda: [0, 0])
    gap = int(GAP_S * FPS)
    sequences = sorted(glob.glob(os.path.join(os.path.expanduser(annotation_root), "seq_*")))

    for seq_dir in sequences:
        cameras = load_sequence(seq_dir)
        for camera, frames in cameras.items():
            for people in frames.values():
                for _pid, _x1, _y1, x2, y2 in people:
                    sizes[camera][0] = max(sizes[camera][0], x2)
                    sizes[camera][1] = max(sizes[camera][1], y2)
        # Recouvrement : même identité, même image, deux caméras.
        visible = {c: {(f, p[0]) for f, ps in frames.items() for p in ps}
                   for c, frames in cameras.items()}
        for a in cameras:
            person_frames[a] += len(visible[a])
            for b in cameras:
                if a != b:
                    overlap_frames[(a, b)] += len(visible[a] & visible[b])
        # Passages : apparitions d'une même identité, toutes caméras, dans l'ordre.
        per_id = defaultdict(list)
        for camera, frames in cameras.items():
            for app in appearances(frames, gap):
                per_id[app["id"]].append((camera, app))
        for apps in per_id.values():
            apps.sort(key=lambda ca: ca[1]["start"])
            for camera, app in apps:
                entries[camera][zone(app["foot_start"], *sizes[camera])] += 1
                exits[camera][zone(app["foot_end"], *sizes[camera])] += 1
            for (a, app_a), (b, app_b) in zip(apps, apps[1:]):
                delta = (app_b["start"] - app_a["end"]) / FPS
                if a != b and 0 <= delta <= MAX_TRANSIT_S:
                    transits[(a, b)].append(delta)

    cams = sorted(person_frames)
    root = os.path.normpath(os.path.expanduser(annotation_root))
    topology = {"source": os.path.basename(os.path.dirname(root)) or os.path.basename(root),
                "sequences": len(sequences), "fps": FPS, "cameras": {}, "overlaps": [],
                "links": []}
    for c in cams:
        topology["cameras"][c] = {
            "observed_extent": [int(v) for v in sizes[c]],
            "person_frames": int(person_frames[c]),
            "entries": dict(entries[c].most_common()),
            "exits": dict(exits[c].most_common()),
        }
    for a in cams:
        for b in cams:
            if a < b:
                shared = overlap_frames[(a, b)]
                ratio = shared / max(1, min(person_frames[a], person_frames[b]))
                if ratio >= 0.01:
                    topology["overlaps"].append({"cameras": [a, b], "shared_person_frames":
                                                 int(shared), "ratio": round(ratio, 3)})
    for (a, b), deltas in sorted(transits.items()):
        if len(deltas) < MIN_PASSAGES:
            continue
        d = np.array(deltas)
        topology["links"].append({
            "from": a, "to": b, "passages": len(d),
            "transit_s": {"p10": round(float(np.percentile(d, 10)), 1),
                          "median": round(float(np.median(d)), 1),
                          "p90": round(float(np.percentile(d, 90)), 1)},
        })
    return topology


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations", required=True, help="dossier annotations/ de CHIRLA")
    parser.add_argument("--output", required=True, help="fichier YAML à écrire")
    args = parser.parse_args()

    topology = infer(args.annotations)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Topologie déduite des annotations (tools/infer_topology.py) : agrégats seulement.\n")
        yaml.safe_dump(topology, f, allow_unicode=True, sort_keys=False)
    print(f"{len(topology['cameras'])} caméras, {len(topology['overlaps'])} recouvrements, "
          f"{len(topology['links'])} liens → {args.output}")
    for o in sorted(topology["overlaps"], key=lambda o: -o["ratio"]):
        print(f"  recouvrement {o['cameras'][0]} ↔ {o['cameras'][1]} : {o['ratio']:.0%}")
    for link in sorted(topology["links"], key=lambda link: -link["passages"]):
        t = link["transit_s"]
        print(f"  passage {link['from']} → {link['to']} : {link['passages']} fois, "
              f"{t['median']} s (p10 {t['p10']}, p90 {t['p90']})")


if __name__ == "__main__":
    main()
