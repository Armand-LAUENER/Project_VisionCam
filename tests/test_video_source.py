"""
tests/test_video_source.py

Couvre la source vidéo « fichier » (simulation de caméra) : dossier de
séquence MOT17, motif d'images, modes every_frame et realtime, lecture en
boucle. Horloge simulée, images générées dans un dossier temporaire.

Lancer : pytest tests/test_video_source.py -v
"""

import configparser

import cv2
import numpy as np
import pytest

from core.video_source import FileSource

N_FRAMES = 10


def make_sequence(tmp_path, n=N_FRAMES, fps=10, width=6):
    """Séquence MOT17 dont l'image i (base 1) est unie, de valeur 20 × i."""
    img_dir = tmp_path / "seq" / "img1"
    img_dir.mkdir(parents=True)
    for i in range(1, n + 1):
        cv2.imwrite(str(img_dir / f"{i:0{width}d}.jpg"),
                    np.full((48, 64, 3), 20 * i, dtype=np.uint8))
    ini = configparser.ConfigParser()
    ini["Sequence"] = {"name": "seq", "imDir": "img1", "frameRate": str(fps),
                       "seqLength": str(n), "imWidth": "64", "imHeight": "48", "imExt": ".jpg"}
    with open(tmp_path / "seq" / "seqinfo.ini", "w") as f:
        ini.write(f)
    return str(tmp_path / "seq")


def index_of(frame):
    """Numéro (base 1) d'une image générée par make_sequence."""
    return round(float(frame.mean()) / 20)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def read_all(source, limit=100):
    frames = []
    for _ in range(limit):
        ok, frame = source.read()
        if not ok:
            break
        frames.append(index_of(frame))
    return frames


class TestOpening:

    def test_mot_sequence_directory(self, tmp_path):
        source = FileSource(make_sequence(tmp_path, fps=25))

        assert source.isOpened()
        assert source.fps == 25
        assert (source.get(cv2.CAP_PROP_FRAME_WIDTH), source.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (64, 48)

    def test_eight_digit_numbering(self, tmp_path):
        """DanceTrack numérote sur 8 chiffres."""
        source = FileSource(make_sequence(tmp_path, width=8))

        assert read_all(source) == list(range(1, N_FRAMES + 1))

    def test_image_pattern(self, tmp_path):
        seq = make_sequence(tmp_path)
        source = FileSource(f"{seq}/img1/%06d.jpg", fps=12)

        assert source.fps == 12
        assert read_all(source) == list(range(1, N_FRAMES + 1))

    def test_missing_file_does_not_open(self, tmp_path):
        assert not FileSource(str(tmp_path / "absent.mp4")).isOpened()

    def test_unknown_mode_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            FileSource(make_sequence(tmp_path), mode="fast")


class TestEveryFrame:

    def test_every_frame_in_order_then_end(self, tmp_path):
        source = FileSource(make_sequence(tmp_path), mode="every_frame")

        assert read_all(source) == list(range(1, N_FRAMES + 1))
        assert source.ended

    def test_does_not_wait_for_the_clock(self, tmp_path):
        clock = FakeClock()
        source = FileSource(make_sequence(tmp_path), mode="every_frame",
                            clock=clock, sleep=clock.sleep)
        read_all(source)

        assert clock.now == 100.0

    def test_loop_restarts_from_the_first_frame(self, tmp_path):
        source = FileSource(make_sequence(tmp_path), mode="every_frame", loop=True)

        frames = read_all(source, limit=2 * N_FRAMES + 3)

        assert frames == list(range(1, N_FRAMES + 1)) * 2 + [1, 2, 3]
        assert not source.ended
        assert source.loops == 2


class TestRealtime:

    def test_waits_for_the_frame_time(self, tmp_path):
        clock = FakeClock()
        source = FileSource(make_sequence(tmp_path, fps=10), mode="realtime",
                            clock=clock, sleep=clock.sleep)

        assert read_all(source) == list(range(1, N_FRAMES + 1))
        # La fin du fichier ne se voit qu'en lisant l'image suivante, à son heure.
        assert clock.now == pytest.approx(100.0 + N_FRAMES / 10)
        assert source.skipped == 0

    def test_slow_consumer_skips_frames(self, tmp_path):
        """Un pipeline deux fois trop lent ne voit qu'une image sur deux, à l'heure."""
        clock = FakeClock()
        source = FileSource(make_sequence(tmp_path, fps=10), mode="realtime",
                            clock=clock, sleep=clock.sleep)
        frames = []
        for _ in range(N_FRAMES):
            ok, frame = source.read()
            if not ok:
                break
            frames.append(index_of(frame))
            clock.now += 0.2   # traitement : deux périodes d'image

        # La 10e, déjà due quand le pipeline revient, est la plus récente : livrée.
        assert frames == [1, 3, 5, 7, 9, 10]
        assert source.skipped == 4

    def test_loop_keeps_the_pace(self, tmp_path):
        clock = FakeClock()
        source = FileSource(make_sequence(tmp_path, fps=10), mode="realtime", loop=True,
                            clock=clock, sleep=clock.sleep)

        frames = read_all(source, limit=N_FRAMES + 2)

        assert frames == list(range(1, N_FRAMES + 1)) + [1, 2]
        assert clock.now == pytest.approx(100.0 + (N_FRAMES + 1) / 10)
