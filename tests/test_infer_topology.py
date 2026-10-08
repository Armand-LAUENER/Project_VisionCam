"""
tests/test_infer_topology.py

Topologie déduite des annotations (tools/infer_topology.py) sur une séquence
synthétique : une personne passe de la caméra 1 à la caméra 2 (3 s plus tard),
une autre est vue en même temps par les caméras 2 et 3.
"""

import json

import pytest

from tools.infer_topology import appearances, infer, zone


def write_sequence(root, name, cameras):
    seq = root / name
    seq.mkdir(parents=True)
    for camera, frames in cameras.items():
        (seq / f"{camera}_2023-01-01-00:00:00.json").write_text(json.dumps(
            {str(f): [{"id": pid, "BboxP": box} for pid, box in people]
             for f, people in frames.items()}))


BOX_LEFT = [10, 300, 60, 700]       # pied en bas à gauche d'une image ~1000 × 700
BOX_RIGHT = [900, 300, 990, 700]


@pytest.fixture
def annotations(tmp_path):
    cams = {"camera_1": {}, "camera_2": {}, "camera_3": {}}
    for f in range(1, 61):                       # personne 1 sur la caméra 1, 2 s
        cams["camera_1"][f] = [(1, BOX_RIGHT)]
    for f in range(151, 211):                    # puis sur la caméra 2, 3 s après
        cams["camera_2"].setdefault(f, []).append((1, BOX_LEFT))
    for f in range(300, 400):                    # personne 2 vue par 2 et 3 ensemble
        cams["camera_2"].setdefault(f, []).append((2, BOX_RIGHT))
        cams["camera_3"][f] = [(2, BOX_LEFT)]
    for seq in ("seq_000", "seq_001", "seq_002"):  # 3 passages : seuil d'un lien
        write_sequence(tmp_path, seq, cams)
    return tmp_path


def test_a_hand_off_becomes_a_link_with_its_transit_time(annotations):
    topology = infer(str(annotations))

    links = {(link["from"], link["to"]): link for link in topology["links"]}
    assert links[("camera_1", "camera_2")]["passages"] == 3
    assert links[("camera_1", "camera_2")]["transit_s"]["median"] == pytest.approx(3.0, abs=0.1)
    assert sum(links[("camera_1", "camera_2")]["transit_histogram"]["counts"]) == 3


def test_simultaneous_views_also_give_a_negative_transit(annotations):
    """Champs qui se recouvrent : l'apparition suivante commence avant la fin de
    la précédente ; le délai négatif fait partie de la distribution du lien."""
    links = {(link["from"], link["to"]): link for link in infer(str(annotations))["links"]}
    overlap = [link for pair, link in links.items() if set(pair) == {"camera_2", "camera_3"}]

    assert len(overlap) == 1 and overlap[0]["transit_s"]["median"] < 0


def test_simultaneous_views_are_an_overlap(annotations):
    topology = infer(str(annotations))

    overlaps = {tuple(o["cameras"]): o["ratio"] for o in topology["overlaps"]}
    assert overlaps == {("camera_2", "camera_3"): 1.0}


def test_entry_and_exit_zones(annotations):
    cam1 = infer(str(annotations))["cameras"]["camera_1"]

    assert cam1["entries"] == {"bas-droite": 3} and cam1["exits"] == {"bas-droite": 3}


def test_appearances_split_on_gaps():
    frames = {f: [(7, 0, 0, 10, 10)] for f in list(range(1, 11)) + list(range(100, 111))}

    apps = appearances(frames, gap=30)

    assert [(a["start"], a["end"]) for a in apps] == [(1, 10), (100, 110)]


def test_zone_grid():
    assert zone((10, 690), 1000, 700) == "bas-gauche"
    assert zone((500, 350), 1000, 700) == "milieu-centre"
