"""
cameras.py — Configuration des caméras (roadmap 2.2).

CAMERAS="hall=~/videos/hall.mp4;porte=rtsp://10.0.0.5/flux;bureau=0" : un
identifiant par caméra (lettres, chiffres, `-`, `_`), puis sa source :

- un nombre : une webcam locale (index OpenCV) ;
- une URL (http://, https://, rtsp://) : une caméra IP ;
- sinon un chemin : vidéo, motif d'images ou séquence MOT17, rejoué comme une
  caméra (core/video_source.py, avec VIDEO_MODE et VIDEO_LOOP).

Sans CAMERAS, une seule caméra, CAMERA_ID, dont la source est lue dans la
configuration au moment de l'ouverture (VIDEO_FILE, sinon CAMERA_SOURCE) :
l'usage à une caméra ne change pas.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_URL = ("http://", "https://", "rtsp://")


@dataclass(frozen=True)
class CameraSpec:
    id: str
    # None : caméra unique, source lue dans la configuration à l'ouverture.
    source: str | int | None
    is_file: bool


def worker_groups(count: int, per_worker: str, available_mb: float, worker_mb: int,
                  reserve_mb: int) -> list[int]:
    """Nombre de caméras de chaque processus de caméras.

    `per_worker` numérique : des groupes de cette taille (le dernier peut être
    plus petit). "auto" : autant de processus que la RAM disponible le permet
    (`worker_mb` chacun, `reserve_mb` gardés libres), au moins un, au plus un
    par caméra, et les caméras réparties à un près.
    """
    if count <= 0:
        return []
    if per_worker != "auto":
        size = max(1, int(per_worker))
        return [min(size, count - i) for i in range(0, count, size)]
    processes = int((available_mb - reserve_mb) // worker_mb)
    processes = max(1, min(count, processes))
    base, extra = divmod(count, processes)
    return [base + (i < extra) for i in range(processes)]


def parse_cameras(spec: str, default_id: str) -> list[CameraSpec]:
    if not spec.strip():
        return [CameraSpec(default_id, None, False)]
    cameras: list[CameraSpec] = []
    for item in (part.strip() for part in spec.split(";")):
        if not item:
            continue
        camera_id, sep, source = (s.strip() for s in item.partition("="))
        if not sep or not source or not _ID.match(camera_id):
            raise ValueError(f"CAMERAS : entrée invalide {item!r} (attendu id=source)")
        if any(c.id == camera_id for c in cameras):
            raise ValueError(f"CAMERAS : identifiant en double {camera_id!r}")
        if source.isdigit():
            cameras.append(CameraSpec(camera_id, int(source), False))
        elif source.startswith(_URL):
            cameras.append(CameraSpec(camera_id, source, False))
        else:
            cameras.append(CameraSpec(camera_id, os.path.expanduser(source), True))
    if not cameras:
        raise ValueError("CAMERAS : aucune caméra")
    return cameras
