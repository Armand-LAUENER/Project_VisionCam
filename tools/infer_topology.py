"""
infer_topology.py — Topologie d'un réseau de caméras déduite des annotations (roadmap 2.4).

CHIRLA ne fournit pas le plan de ses 7 caméras, mais ses identités sont
communes à toutes les caméras d'une séquence, filmées en même temps. Pour
chaque personne annotée, ses apparitions sur chaque caméra (suites d'images
sans trou de plus de GAP_S) donnent :

  - le recouvrement des champs : la même personne visible au même instant sur
    deux caméras (part des images-personnes de A aussi vues sur B) ;
  - les passages : l'apparition suivante d'une personne commence sur B ≠ A
    entre MIN_TRANSIT_S et MAX_TRANSIT_S après la fin de celle sur A, avec
    l'histogramme des temps de transit (pas de HIST_BIN_S), que
    core/topology.py lit comme une distribution ;
  - les zones d'entrée et de sortie de chaque caméra : où, dans l'image, les
    apparitions commencent et finissent (pied de la boîte, grille 3 × 3 sur
    l'étendue des boîtes annotées, `observed_extent` : la taille des images
    n'est pas dans les annotations).

  - pour les champs qui se recouvrent, l'homographie du sol entre deux
    caméras, estimée (RANSAC) sur les pieds d'une même personne vue au même
    instant (bas-centre des boîtes). Validation dans le temps, comme le
    protocole de docs/tracking.md : estimée sur les premières séquences
    (HOMOGRAPHY_FIT_SHARE, dans l'ordre chronologique des noms ; CHIRLA :
    juin-juillet), jugée sur les suivantes (décembre) par l'association par position
    (la personne de B dont le pied projeté est le plus proche est-elle la
    bonne ?) ; utilisable au-dessus de HOMOGRAPHY_MIN_MATCH. Échoue quand les
    pieds ne sont pas visibles (personnes assises derrière un bureau).

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

import cv2
import numpy as np
import yaml

FPS = 30
GAP_S = 1.0
MAX_TRANSIT_S = 60.0
# Une apparition sur B peut commencer un peu avant la fin de celle sur A
# (champs qui se recouvrent) : délais comptés à partir de MIN_TRANSIT_S.
MIN_TRANSIT_S = -5.0
HIST_BIN_S = 0.5
MIN_PASSAGES = 2
HOMOGRAPHY_MIN_OVERLAP = 0.01
HOMOGRAPHY_MIN_POINTS = 50
HOMOGRAPHY_RANSAC_PX = 20.0
HOMOGRAPHY_MIN_MATCH = 0.8
HOMOGRAPHY_FIT_SHARE = 0.7


def foot(box) -> tuple[float, float]:
    _pid, x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def fit_homography(points: list) -> np.ndarray | None:
    """Homographie A → B (RANSAC) depuis des paires de pieds ((xa, ya), (xb, yb))."""
    if len(points) < HOMOGRAPHY_MIN_POINTS:
        return None
    src = np.float32([p[0] for p in points])
    dst = np.float32([p[1] for p in points])
    matrix, _mask = cv2.findHomography(src, dst, cv2.RANSAC, HOMOGRAPHY_RANSAC_PX)
    return matrix


def position_match(matrix: np.ndarray, frames: list) -> tuple[float, int]:
    """Part des personnes de A dont le pied projeté tombe au plus près du leur
    sur B, parmi les images où B voit au moins deux personnes."""
    right = total = 0
    for people_a, people_b in frames:
        ids_b = list(people_b)
        points_b = np.float32([people_b[i] for i in ids_b])
        for pid, point in people_a.items():
            if pid not in people_b:
                continue
            projected = cv2.perspectiveTransform(np.float32([[point]]), matrix)[0, 0]
            right += ids_b[int(np.argmin(np.linalg.norm(points_b - projected, axis=1)))] == pid
            total += 1
    return (right / total if total else 0.0), total
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


def homography_entry(points_by_seq: dict, frames_by_seq: dict, sequences: list[str]) -> dict:
    """Homographie de sol d'une paire, estimée sur les premières séquences et
    jugée sur les suivantes."""
    cut = max(1, round(HOMOGRAPHY_FIT_SHARE * len(sequences)))
    fit_half, test_half = sequences[:cut], sequences[cut:]
    matrix = fit_homography([p for s in fit_half for p in points_by_seq.get(s, [])])
    if matrix is None:
        return {}
    match, cases = position_match(matrix, [f for s in test_half for f in frames_by_seq.get(s, [])])
    final = fit_homography([p for s in sequences for p in points_by_seq.get(s, [])])
    return {"homography": {"matrix": [[round(float(v), 6) for v in row] for row in final],
                           "position_match": round(match, 3), "validation_cases": cases,
                           "usable": bool(match >= HOMOGRAPHY_MIN_MATCH and cases >= 100)}}


def infer(annotation_root: str, sequences: list[str] | None = None) -> dict:
    """Topologie des séquences `sequences` (noms seq_*), toutes par défaut."""
    overlap_frames = Counter()      # (A, B) → images-personnes de A aussi vues sur B
    person_frames = Counter()       # A → images-personnes
    transits = defaultdict(list)    # (A, B) → secondes
    entries = defaultdict(Counter)
    exits = defaultdict(Counter)
    sizes = defaultdict(lambda: [0, 0])
    gap = int(GAP_S * FPS)
    # Pieds d'une même personne au même instant, par paire de caméras et par
    # séquence (homographies) ; images où B voit au moins deux personnes.
    feet_pairs = defaultdict(lambda: defaultdict(list))
    feet_frames = defaultdict(lambda: defaultdict(list))
    sequences = sorted(p for p in glob.glob(os.path.join(os.path.expanduser(annotation_root), "seq_*"))
                       if sequences is None or os.path.basename(p) in sequences)

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
                if a < b:
                    for f in cameras[a].keys() & cameras[b].keys():
                        pa = {p[0]: foot(p) for p in cameras[a][f]}
                        pb = {p[0]: foot(p) for p in cameras[b][f]}
                        shared = pa.keys() & pb.keys()
                        feet_pairs[(a, b)][seq_dir].extend((pa[i], pb[i]) for i in shared)
                        if shared and len(pb) >= 2:
                            feet_frames[(a, b)][seq_dir].append((pa, pb))
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
                if a != b and MIN_TRANSIT_S <= delta <= MAX_TRANSIT_S:
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
                    entry = {"cameras": [a, b], "shared_person_frames": int(shared),
                             "ratio": round(ratio, 3)}
                    entry.update(homography_entry(feet_pairs[(a, b)], feet_frames[(a, b)],
                                                  sequences))
                    topology["overlaps"].append(entry)
    for (a, b), deltas in sorted(transits.items()):
        if len(deltas) < MIN_PASSAGES:
            continue
        d = np.array(deltas)
        edges = np.arange(MIN_TRANSIT_S, MAX_TRANSIT_S + HIST_BIN_S, HIST_BIN_S)
        counts, _ = np.histogram(d, bins=edges)
        topology["links"].append({
            "from": a, "to": b, "passages": len(d),
            "transit_s": {"p10": round(float(np.percentile(d, 10)), 1),
                          "median": round(float(np.median(d)), 1),
                          "p90": round(float(np.percentile(d, 90)), 1)},
            "transit_histogram": {"start_s": MIN_TRANSIT_S, "bin_s": HIST_BIN_S,
                                  "counts": [int(c) for c in counts]},
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
        h = o.get("homography")
        homography = (f", homographie {'utilisable' if h['usable'] else 'écartée'} "
                      f"(association par position {h['position_match']:.0%})" if h else "")
        print(f"  recouvrement {o['cameras'][0]} ↔ {o['cameras'][1]} : {o['ratio']:.0%}{homography}")
    for link in sorted(topology["links"], key=lambda link: -link["passages"]):
        t = link["transit_s"]
        print(f"  passage {link['from']} → {link['to']} : {link['passages']} fois, "
              f"{t['median']} s (p10 {t['p10']}, p90 {t['p90']})")


if __name__ == "__main__":
    main()
