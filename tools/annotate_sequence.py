"""
annotate_sequence.py — Construit et corrige la vérité terrain d'une séquence.

Pour une séquence au format MOT17 (cf. tools/record_sequence.py) :

1. Si `gt/gt.txt` n'existe pas, il est pré-rempli : DeepSORT tourne sur
   `det/det.txt` et chaque piste devient une identité. Seules les boîtes
   réellement détectées sont gardées, pas les prédictions de Kalman.
2. Une fenêtre montre chaque image avec ses boîtes et leurs identités. On y
   corrige ce que le tracker a raté, surtout les changements d'identité.

Commandes :

    clic sur une boîte    la sélectionner
    chiffres + Entrée     donner ce numéro à la boîte sélectionnée, à partir de
                          cette image ; si le numéro existe déjà, les deux
                          identités sont échangées (c'est la correction d'un
                          changement d'identité)
    x                     supprimer l'identité sélectionnée à partir d'ici
                          (fausse détection qui dure)
    c                     supprimer la boîte sélectionnée sur cette image seule
    d / a                 image suivante / précédente
    e / q                 25 images en avant / en arrière
    n                     prochaine image où une identité apparaît ou disparaît
    u                     annuler
    s                     enregistrer
    Échap                 enregistrer et quitter

Limite assumée : seules les personnes détectées par YOLO sont annotées. Les
oublis du détecteur ne comptent donc pas, ce qui est sans effet sur le réglage
de l'association (le détecteur est le même pour toutes les combinaisons).

Depuis la racine du projet :

    uv run -m tools.annotate_sequence ~/datasets/visioncam/croisements-01
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import sys

import cv2
import numpy as np

# Une ligne de vérité terrain : [image, identité, gauche, haut, largeur, hauteur],
# pixels en base 1 comme dans les fichiers MOT.
Row = list


def load_gt(path) -> list[Row]:
    rows = []
    with open(path) as f:
        for line in f:
            fields = line.strip().split(",")
            if len(fields) >= 6:
                rows.append([int(float(fields[0])), int(float(fields[1])),
                             *(float(v) for v in fields[2:6])])
    return rows


def save_gt(path, rows: list[Row]) -> None:
    """Format gt MOT16 : conf=1 (compté), classe 1 (piéton), visibilité 1."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for frame, track_id, left, top, width, height in sorted(rows, key=lambda r: (r[0], r[1])):
            f.write(f"{frame},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,1,1\n")
    os.replace(tmp, path)


def swap_ids(rows: list[Row], a: int, b: int, from_frame: int) -> list[Row]:
    """À partir de `from_frame`, `a` devient `b` et `b` devient `a`.

    Renommer une piste en un numéro libre est le cas particulier où `b`
    n'apparaît plus ensuite. Échanger plutôt qu'écraser évite deux boîtes de
    même identité sur une image.
    """
    swapped = []
    for frame, track_id, *box in rows:
        if frame >= from_frame and track_id in (a, b):
            track_id = b if track_id == a else a
        swapped.append([frame, track_id, *box])
    return swapped


def delete_track(rows: list[Row], track_id: int, from_frame: int,
                 only_this_frame: bool = False) -> list[Row]:
    def removed(frame, tid):
        if tid != track_id:
            return False
        return frame == from_frame if only_this_frame else frame >= from_frame

    return [row for row in rows if not removed(row[0], row[1])]


def box_at(rows: list[Row], frame: int, x: float, y: float) -> int | None:
    """Identité de la plus petite boîte de l'image qui contient le point (base 0)."""
    hits = [(width * height, track_id)
            for f, track_id, left, top, width, height in rows
            if f == frame
            and left - 1 <= x <= left - 1 + width and top - 1 <= y <= top - 1 + height]
    return min(hits)[1] if hits else None


def next_event(rows: list[Row], frame: int, last_frame: int) -> int:
    """Prochaine image dont l'ensemble d'identités diffère de l'image précédente."""
    ids_by_frame: dict[int, set] = {}
    for f, track_id, *_ in rows:
        ids_by_frame.setdefault(f, set()).add(track_id)
    for f in range(frame + 1, last_frame + 1):
        if ids_by_frame.get(f, set()) != ids_by_frame.get(f - 1, set()):
            return f
    return last_frame


def bootstrap(seq_dir) -> list[Row]:
    """Vérité terrain initiale : pistes DeepSORT sur les détections de la séquence."""
    from deep_sort_realtime.deepsort_tracker import DeepSort

    import config
    from core.appearance_embedder import build_embedder
    from core.tracker_backends import new_deep_sort
    from tools import eval_mot

    # n_init=1 : une personne doit être annotée dès sa première détection,
    # pas seulement une fois sa piste confirmée.
    config.DEEPSORT_N_INIT = 1
    tracker = new_deep_sort()
    embedder = build_embedder()
    detections = eval_mot.load_public_detections(seq_dir, 0.0)
    rows = []
    for frame in range(1, eval_mot.sequence_length(seq_dir) + 1):
        dets = [d for d in detections.get(frame, []) if d[0][2] > 0 and d[0][3] > 0]
        embeds = []
        if dets:
            image = cv2.imread(os.path.join(seq_dir, "img1", f"{frame:06d}.jpg"))
            embeds = list(embedder.predict(DeepSort.crop_bb(image, dets)[0]))
        for t in tracker.update_tracks(dets, embeds=embeds):
            if t.is_confirmed() and t.time_since_update == 0:
                x1, y1, x2, y2 = (float(v) for v in t.to_ltrb(orig=True))
                rows.append([frame, int(t.track_id), x1 + 1, y1 + 1, x2 - x1, y2 - y1])
    return rows


def colour(track_id):
    hsv = np.uint8([[[(track_id * 47) % 180, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])


def render(image, rows, frame, last_frame, selected, typed, dirty):
    canvas = image.copy()
    for f, track_id, left, top, width, height in rows:
        if f != frame:
            continue
        p1 = (int(left - 1), int(top - 1))
        p2 = (int(left - 1 + width), int(top - 1 + height))
        is_selected = track_id == selected
        cv2.rectangle(canvas, p1, p2, (255, 255, 255) if is_selected else colour(track_id),
                      4 if is_selected else 2)
        cv2.putText(canvas, f"#{track_id}", (p1[0] + 4, p1[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour(track_id), 2)
    status = f"image {frame}/{last_frame}"
    if selected is not None:
        status += f"  |  #{selected} -> {typed or '_'}"
    if dirty:
        status += "  |  non enregistre"
    cv2.putText(canvas, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(canvas, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequence", help="dossier de la séquence (format MOT17)")
    args = parser.parse_args()

    from tools import eval_mot

    seq_dir = os.path.expanduser(args.sequence)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    if os.path.exists(gt_path):
        rows = load_gt(gt_path)
    else:
        print("Pas de gt/gt.txt : pré-remplissage avec DeepSORT…")
        rows = bootstrap(seq_dir)
        save_gt(gt_path, rows)
    last_frame = eval_mot.sequence_length(seq_dir)
    if not last_frame:
        sys.exit(f"Séquence vide : {seq_dir}")

    state = {"frame": 1, "selected": None, "typed": "", "dirty": False}
    history: list[list[Row]] = []
    window = "VisionCam - annotation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["selected"] = box_at(rows, state["frame"], x, y)
            state["typed"] = ""

    def edit(new_rows):
        history.append(copy.deepcopy(rows))
        rows[:] = new_rows
        state["dirty"] = True

    cv2.setMouseCallback(window, on_mouse)
    image_cache: dict[int, np.ndarray] = {}
    steps = {ord("d"): 1, ord("a"): -1, ord("e"): 25, ord("q"): -25}

    while True:
        frame = state["frame"]
        if frame not in image_cache:
            image_cache.clear()
            image_cache[frame] = cv2.imread(os.path.join(seq_dir, "img1", f"{frame:06d}.jpg"))
        cv2.imshow(window, render(image_cache[frame], rows, frame, last_frame,
                                  state["selected"], state["typed"], state["dirty"]))
        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue
        if key == 27:
            save_gt(gt_path, rows)
            break
        if key == ord("s"):
            save_gt(gt_path, rows)
            state["dirty"] = False
        elif key in steps:
            state["frame"] = min(last_frame, max(1, frame + steps[key]))
        elif key == ord("n"):
            state["frame"] = next_event(rows, frame, last_frame)
        elif key == ord("u") and history:
            rows[:] = history.pop()
            state["dirty"] = True
        elif chr(key).isdigit() and state["selected"] is not None:
            state["typed"] += chr(key)
        elif key == 8:
            state["typed"] = state["typed"][:-1]
        elif key in (10, 13) and state["selected"] is not None and state["typed"]:
            new_id = int(state["typed"])
            edit(swap_ids(rows, state["selected"], new_id, frame))
            state["selected"], state["typed"] = new_id, ""
        elif key in (ord("x"), ord("c")) and state["selected"] is not None:
            edit(delete_track(rows, state["selected"], frame, only_this_frame=key == ord("c")))
            state["selected"] = None

    cv2.destroyAllWindows()
    print(f"Vérité terrain enregistrée : {gt_path} ({len(rows)} boîtes, "
          f"{len({r[1] for r in rows})} identités)")


if __name__ == "__main__":
    main()
