"""
tests/test_convert_chokepoint.py

Conversion ChokePoint → MOT17 (tools/convert_chokepoint.py) sur une petite
arborescence synthétique : XML des yeux, images numérotées avec un saut,
photos de galerie.
"""

import os

import cv2
import numpy as np

from tools.convert_chokepoint import build_gallery, convert, parse_groundtruth

XML = """<!DOCTYPE GroundTruth>
<dataset name="P1E_S1_C1">
 <frame number="00000010"/>
 <frame number="00000011">
  <person id="0003">
   <leftEye x="626" y="210"/>
   <rightEye x="643" y="214"/>
  </person>
 </frame>
 <frame number="00000080">
  <person id="0003">
   <leftEye x="600" y="200"/>
   <rightEye x="620" y="202"/>
  </person>
  <person id="0012">
   <leftEye x="100" y="50"/>
   <rightEye x="120" y="52"/>
  </person>
 </frame>
</dataset>
"""


def make_tree(tmp_path):
    root = tmp_path / "raw"
    frames = root / "P1E_S1" / "P1E_S1_C1" / "P1E_S1_C1"
    frames.mkdir(parents=True)
    for number in (10, 11, 80):           # saut entre 11 et 80
        cv2.imwrite(str(frames / f"{number:08d}.jpg"), np.zeros((60, 80, 3), np.uint8))
    gt_dir = root / "groundtruth" / "groundtruth"
    gt_dir.mkdir(parents=True)
    (gt_dir / "P1E_S1_C1.xml").write_text(XML)
    for expression in ("Neutral", "Smile"):
        still = root / "Still" / expression
        still.mkdir(parents=True)
        for pid in (3, 12):
            cv2.imwrite(str(still / f"ID{pid:04d}.JPG"), np.zeros((400, 3200, 3), np.uint8))
    return root


def test_groundtruth_lists_eyes_per_frame():
    eyes = parse_groundtruth(XML)

    assert eyes[10] == []
    assert eyes[11] == [(3, 626, 210, 643, 214)]
    assert [p[0] for p in eyes[80]] == [3, 12]


def test_frames_are_renumbered_and_eyes_follow(tmp_path):
    root = make_tree(tmp_path)

    count = convert(str(root), str(tmp_path / "mot"), "P1E_S1_C1")

    seq = tmp_path / "mot" / "P1E_S1_C1"
    assert count == 3
    assert sorted(os.listdir(seq / "img1")) == ["000001.jpg", "000002.jpg", "000003.jpg"]
    assert os.path.realpath(seq / "img1" / "000003.jpg").endswith("00000080.jpg")
    lines = (seq / "gt" / "gt_eyes.txt").read_text().split()
    assert lines == ["2,3,627,211,644,215", "3,3,601,201,621,203", "3,12,101,51,121,53"]
    assert "seqLength=3" in (seq / "seqinfo.ini").read_text()


def test_gallery_leaves_held_out_people_out(tmp_path):
    root = make_tree(tmp_path)
    gallery = tmp_path / "gallery"

    names = build_gallery(str(root), str(gallery), hold_out={12})

    assert names == ["ID0003"]
    assert sorted(os.listdir(gallery / "ID0003")) == ["neutral.jpg", "smile.jpg"]
    assert cv2.imread(str(gallery / "ID0003" / "neutral.jpg")).shape[1] == 1600
