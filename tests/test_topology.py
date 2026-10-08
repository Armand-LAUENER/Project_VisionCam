"""
tests/test_topology.py

Plausibilité d'une association entre caméras (core/topology.py) sur une
topologie construite à la main : lien A → B à ~2 s, lien A → C à ~20 s, rare.
"""

import pytest

from core.topology import FLOOR, Topology


def hist(deltas, start=-5.0, bin_s=0.5, end=60.0):
    counts = [0] * int((end - start) / bin_s)
    for d in deltas:
        counts[int((d - start) / bin_s)] += 1
    return {"start_s": start, "bin_s": bin_s, "counts": counts}


@pytest.fixture
def topology():
    return Topology.from_dict({
        "cameras": {"A": {}, "B": {}, "C": {}},
        "links": [
            {"from": "A", "to": "B", "passages": 9, "transit_histogram": hist([2.0] * 9)},
            {"from": "A", "to": "C", "passages": 1, "transit_histogram": hist([20.0])},
        ],
    })


def test_a_typical_delay_scores_higher_than_an_odd_one(topology):
    assert topology.score("A", "B", 2.0) > 10 * topology.score("A", "B", 10.0)


def test_a_frequent_link_outweighs_a_rare_one_at_its_typical_delay(topology):
    assert topology.score("A", "B", 2.0) > topology.score("A", "C", 20.0)


def test_an_unseen_link_keeps_a_small_non_zero_score(topology):
    assert topology.score("B", "C", 2.0) == FLOOR


def test_plausible_sources_are_ranked_and_thresholded(topology):
    sources = topology.plausible_sources("B", {"A": 2.0, "C": 2.0}, threshold=0.01)

    assert [c for c, _ in sources] == ["A"]


def test_the_chirla_topology_file_loads():
    topology = Topology.load("topologies/chirla.yaml")

    assert len(topology.cameras) == 7
    assert topology.score("camera_6", "camera_5", 1.4) > topology.score("camera_6", "camera_5", 30)


# ─────────────────────────────────────────────────────────────────────────────
# Homographie de sol entre champs qui se recouvrent
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np  # noqa: E402

from tools.infer_topology import fit_homography, position_match  # noqa: E402

H = np.array([[0.9, 0.05, 30.0], [0.02, 1.1, -20.0], [0.0001, 0.0002, 1.0]])


def apply(matrix, point):
    x, y, w = matrix @ np.array([point[0], point[1], 1.0])
    return (x / w, y / w)


def test_a_homography_is_recovered_from_paired_feet():
    rng = np.random.default_rng(0)
    points = [(tuple(p), apply(H, p)) for p in rng.uniform(0, 700, (200, 2))]

    matrix = fit_homography(points)

    assert np.allclose(apply(matrix, (350, 600)), apply(H, (350, 600)), atol=1)


def test_position_match_finds_the_right_person():
    frames = [({1: (100, 600), 2: (500, 600)}, {1: apply(H, (100, 600)), 2: apply(H, (500, 600))})]

    assert position_match(H, frames) == (1.0, 2)


def test_only_validated_homographies_project():
    def overlap(a, b, usable):
        return {"cameras": [a, b], "ratio": 0.2,
                "homography": {"matrix": H.tolist(), "usable": usable}}

    topology = Topology.from_dict({"cameras": {"A": {}, "B": {}, "C": {}},
                                   "overlaps": [overlap("A", "B", True), overlap("A", "C", False)]})

    assert np.allclose(topology.project("A", "B", (100, 600)), apply(H, (100, 600)))
    assert np.allclose(topology.project("B", "A", apply(H, (100, 600))), (100, 600))
    assert topology.project("A", "C", (100, 600)) is None
