"""
video_source.py — Source vidéo « fichier » : rejoue une vidéo comme une caméra.

Sert à faire passer une séquence enregistrée dans l'application complète, au
lieu de la caméra. Trois formes de chemin :

- un dossier de séquence MOT17 (`img1/` + `seqinfo.ini`, cadence lue dedans),
  tel que l'écrit tools/record_sequence.py ;
- un motif d'images OpenCV (`…/img1/%06d.jpg`) ;
- un fichier vidéo.

Deux modes :

- `realtime` : les images sont livrées à leur heure, à la cadence d'origine.
  Un pipeline trop lent reçoit la plus récente et les autres sont sautées,
  comme avec une vraie caméra. Réaliste, non déterministe.
- `every_frame` : chaque image est livrée, dans l'ordre, sans attendre.
  Déterministe : c'est le mode des tests et des mesures.

L'objet imite cv2.VideoCapture (isOpened, read, get, release) pour que la
boucle caméra de l'application n'ait pas à les distinguer.
"""

from __future__ import annotations

import configparser
import os
import time

import cv2

MODES = ("realtime", "every_frame")
DEFAULT_FPS = 30.0
# Tolérance sur le calcul de l'image due : 0,1 × 10 vaut 0,9999… en flottant.
_EPSILON = 1e-6


def _resolve(path: str) -> tuple[str, float | None]:
    """(chemin à passer à OpenCV, cadence de seqinfo.ini ou None)."""
    img_dir = os.path.join(path, "img1")
    if not os.path.isdir(img_dir):
        return path, None
    names = sorted(n for n in os.listdir(img_dir) if n.endswith(".jpg"))
    # MOT17 numérote sur 6 chiffres, DanceTrack sur 8.
    width = len(os.path.splitext(names[0])[0]) if names else 6
    fps = None
    ini = os.path.join(path, "seqinfo.ini")
    if os.path.exists(ini):
        parser = configparser.ConfigParser()
        parser.read(ini)
        fps = float(parser["Sequence"].get("frameRate", 0)) or None
    return os.path.join(img_dir, f"%0{width}d.jpg"), fps


class FileSource:
    def __init__(self, path: str, mode: str = "realtime", loop: bool = False,
                 fps: float | None = None, clock=time.monotonic, sleep=time.sleep) -> None:
        if mode not in MODES:
            raise ValueError(f"mode inconnu : {mode!r} (attendu : {', '.join(MODES)})")
        self.path = path
        self.mode = mode
        self.loop = loop
        self._clock = clock
        self._sleep = sleep
        self._target, seq_fps = _resolve(os.path.expanduser(path))
        self._cap = cv2.VideoCapture(self._target)
        self.fps = float(fps or seq_fps or self._cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS)

        self._start: float | None = None   # heure de l'image 0
        self._next = 0                      # numéro de la prochaine image, boucles comprises
        self.skipped = 0                    # images sautées en mode realtime
        self.loops = 0                      # retours au début
        self.ended = False

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return self._cap.get(prop)

    def release(self) -> None:
        self._cap.release()

    def read(self):
        """(True, image) ; (False, None) à la fin du fichier, sauf en boucle."""
        if self.ended:
            return False, None
        if self._start is None:
            self._start = self._clock()
        last = self._next
        if self.mode == "realtime":
            due = self._start + self._next / self.fps
            wait = due - self._clock()
            if wait > 0:
                self._sleep(wait)
            # Images dues depuis : seule la plus récente est livrée.
            last = max(self._next,
                       int((self._clock() - self._start) * self.fps + _EPSILON))

        frame = None
        restarted = False
        while self._next <= last:
            ok, image = self._cap.read()
            if not ok:
                # Une seule réouverture par lecture : un fichier devenu vide
                # ne doit pas faire tourner la boucle sans fin.
                if self.loop and not restarted:
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self._target)
                    self.loops += 1
                    restarted = True
                    continue
                break
            if frame is not None:
                self.skipped += 1
            frame = image
            self._next += 1

        if frame is None:
            self.ended = True
            return False, None
        return True, frame
