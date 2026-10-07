"""
eval_global_identity.py — Cohérence des noms entre caméras (roadmap 2.3).

tools/eval_identity.py juge chaque caméra seule. Ici, plusieurs caméras
filment les mêmes personnes en même temps (CHIRLA : identités annotées
communes à toutes les caméras d'une séquence, vidéos démarrées ensemble) ; à
chaque image, chaque personne annotée reçoit sur chaque caméra où elle est
visible le nom de la piste affichée qui la recouvre (IoU ≥ 0,5), comme dans
eval_identity. On en tire, par moment (image, personne) :

  - nommée quelque part : son nom est affiché sur au moins une caméra ;
  - mauvais nom quelque part : un autre nom est affiché sur au moins une ;
  - noms en conflit : deux caméras affichent deux noms différents ;
  - clone : un même nom est affiché au même instant sur deux personnes
    différentes (comptés par image et par nom) ;
  - propagation possible : une caméra suit la personne en « Inconnu » alors
    qu'une autre affiche son nom au même instant. C'est le gain maximal d'un
    passage de nom entre caméras (roadmap 2.4, 2.5), qui demanderait de savoir
    que les deux pistes sont la même personne.

Le numéro d'image fait office d'horloge commune : les vidéos d'une séquence
CHIRLA démarrent ensemble (quelques images d'écart sur ~8 000).

Depuis la racine du projet, après un rejeu multi-caméras en `every_frame`
(CAMERAS, TRACKS_LOG_PATH : un journal par caméra) :

    uv run -m tools.eval_global_identity --known-faces /tmp/known_faces_chirla \\
        --camera ~/datasets/CHIRLA/mot/seq_025_camera_2=data/tracks-a.csv \\
        --camera ~/datasets/CHIRLA/mot/seq_025_camera_3=data/tracks-b.csv
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field

from tools.eval_identity import UNKNOWN, _matches, load_gt, load_tracks

# Personne annotée mais sans piste affichée qui la recouvre.
UNTRACKED = None


def shown_names(gt: dict, tracks: dict) -> dict:
    """{image: {identité: nom affiché, ou UNTRACKED}} pour une caméra."""
    shown = {}
    for frame, people in gt.items():
        displayed = tracks.get(frame, [])
        matched = _matches([p[1:] for p in people], [t[1:5] for t in displayed])
        shown[frame] = {gid: displayed[matched[i]][5] if i in matched else UNTRACKED
                        for i, (gid, *_box) in enumerate(people)}
    return shown


@dataclass
class GlobalReport:
    moments: int = 0                 # (image, personne enrôlée) visible sur ≥ 1 caméra
    named_somewhere: int = 0
    wrong_somewhere: int = 0
    conflicts: int = 0
    unenrolled_moments: int = 0
    unenrolled_named: int = 0
    clones: int = 0                  # (image, nom) affiché sur ≥ 2 personnes
    tracked_unknown: int = 0         # (image, caméra, enrôlée) suivie en « Inconnu »
    propagatable: int = 0            # … alors qu'une autre caméra affiche son nom
    per_camera_named: dict = field(default_factory=dict)  # caméra → moments nommés


def evaluate_global(cameras: dict, gt_names: dict) -> GlobalReport:
    """cameras : {caméra: {image: {identité: nom ou UNTRACKED}}} ;
    gt_names : {identité: nom enrôlé, ou None si non enrôlée}."""
    report = GlobalReport(per_camera_named={camera: 0 for camera in cameras})
    frames = sorted(set().union(*(views.keys() for views in cameras.values())))
    for frame in frames:
        views = {camera: shown.get(frame, {}) for camera, shown in cameras.items()}
        people = set().union(*(view.keys() for view in views.values()))

        bearers = defaultdict(set)   # nom → identités qui le portent à cet instant
        for view in views.values():
            for gid, name in view.items():
                if name not in (UNTRACKED, UNKNOWN):
                    bearers[name].add(gid)
        report.clones += sum(1 for gids in bearers.values() if len(gids) > 1)

        for gid in people:
            names = {camera: view[gid] for camera, view in views.items() if gid in view}
            displayed = {n for n in names.values() if n not in (UNTRACKED, UNKNOWN)}
            expected = gt_names.get(gid)
            if expected is None:
                report.unenrolled_moments += 1
                report.unenrolled_named += bool(displayed)
                continue
            report.moments += 1
            report.named_somewhere += expected in displayed
            report.wrong_somewhere += bool(displayed - {expected})
            report.conflicts += len(displayed) > 1
            for camera, name in names.items():
                report.per_camera_named[camera] += name == expected
                if name == UNKNOWN:
                    report.tracked_unknown += 1
                    report.propagatable += expected in displayed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", action="append", required=True, metavar="SÉQUENCE=JOURNAL",
                        help="séquence MOT17 (gt/gt.txt) et journal des pistes d'une caméra")
    parser.add_argument("--known-faces", required=True,
                        help="dossier known_faces/ utilisé par l'application")
    parser.add_argument("--prefix", default="id_", help="préfixe des noms enrôlés (défaut : id_)")
    args = parser.parse_args()

    enrolled = set(os.listdir(os.path.expanduser(args.known_faces)))
    cameras, gids = {}, set()
    for item in args.camera:
        seq_dir, tracks_path = (os.path.expanduser(s) for s in item.split("=", 1))
        gt = load_gt(seq_dir)
        tracks = load_tracks(tracks_path)
        if not tracks:
            raise SystemExit(f"Aucune piste dans {tracks_path} : le rejeu a-t-il tourné ?")
        cameras[os.path.basename(seq_dir.rstrip("/"))] = shown_names(gt, tracks)
        gids |= {p[0] for people in gt.values() for p in people}
    gt_names = {gid: (f"{args.prefix}{gid}" if f"{args.prefix}{gid}" in enrolled else None)
                for gid in gids}

    r = evaluate_global(cameras, gt_names)
    n = max(1, r.moments)
    print(f"{len(cameras)} caméras, {r.moments} moments (image, personne enrôlée) "
          f"visibles sur au moins une caméra.")
    for camera, named in r.per_camera_named.items():
        print(f"  nommée sur {camera:<20} {named / n:6.1%}")
    print(f"  nommée quelque part          {r.named_somewhere / n:6.1%}")
    print(f"  mauvais nom quelque part     {r.wrong_somewhere / n:6.1%}")
    print(f"  noms en conflit              {r.conflicts / n:6.1%}")
    print(f"Clones (même nom sur deux personnes au même instant) : {r.clones} (image, nom)")
    if r.unenrolled_moments:
        print(f"Non enrôlées nommées quelque part : "
              f"{r.unenrolled_named / r.unenrolled_moments:.1%} de {r.unenrolled_moments} moments")
    if r.tracked_unknown:
        print(f"Propagation possible : {r.propagatable / r.tracked_unknown:.1%} des "
              f"{r.tracked_unknown} suivis en « Inconnu » (image, caméra) ont leur nom "
              f"affiché par une autre caméra au même instant")


if __name__ == "__main__":
    main()
