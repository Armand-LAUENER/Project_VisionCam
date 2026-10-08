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
