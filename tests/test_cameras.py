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
