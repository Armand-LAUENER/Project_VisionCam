"""
tests/test_cameras.py

Couvre la configuration des caméras (core/cameras.py, roadmap 2.2) :
CAMERAS="id=source;…", une source par caméra (webcam, URL, fichier), et la
caméra unique d'avant quand CAMERAS est vide.

Lancer : pytest tests/test_cameras.py -v
"""

import os

import pytest

from core.cameras import CameraSpec, parse_cameras


class TestParseCameras:

    def test_empty_gives_the_single_configured_camera(self):
        """Sans CAMERAS, la source reste lue dans la config à l'ouverture (VIDEO_FILE, CAMERA_SOURCE)."""
        assert parse_cameras("", "cam0") == [CameraSpec("cam0", None, False)]

    def test_several_sources(self):
        specs = parse_cameras("hall=~/videos/hall.mp4 ; porte=rtsp://10.0.0.5/flux ; bureau=0",
                              "cam0")

        assert specs == [
            CameraSpec("hall", os.path.expanduser("~/videos/hall.mp4"), True),
            CameraSpec("porte", "rtsp://10.0.0.5/flux", False),
            CameraSpec("bureau", 0, False),
        ]

    def test_trailing_separator_is_ignored(self):
        assert [s.id for s in parse_cameras("a=0;b=1;", "cam0")] == ["a", "b"]

    @pytest.mark.parametrize("spec", ["a=0;a=1", "sans-source", "=0", "a b=0", "a="])
    def test_invalid_specs_are_refused(self, spec):
        with pytest.raises(ValueError):
            parse_cameras(spec, "cam0")


# ─────────────────────────────────────────────────────────────────────────────
# Caméras par processus (worker_groups)
# ─────────────────────────────────────────────────────────────────────────────

from core.cameras import worker_groups  # noqa: E402


def test_auto_uses_as_many_processes_as_ram_allows():
    # 10,4 Go disponibles, 2 Go par processus, 1,5 Go de réserve : 4 processus.
    assert worker_groups(16, "auto", 10_400, 2000, 1500) == [4, 4, 4, 4]


def test_auto_spreads_cameras_within_one():
    assert worker_groups(10, "auto", 10_400, 2000, 1500) == [3, 3, 2, 2]


def test_auto_never_starts_more_processes_than_cameras():
    assert worker_groups(2, "auto", 30_000, 2000, 1500) == [1, 1]


def test_auto_keeps_one_process_when_ram_is_short():
    assert worker_groups(4, "auto", 1_000, 2000, 1500) == [4]


def test_a_number_sets_the_group_size():
    assert worker_groups(5, "2", 30_000, 2000, 1500) == [2, 2, 1]


def test_no_camera_no_process():
    assert worker_groups(0, "auto", 30_000, 2000, 1500) == []
