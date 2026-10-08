"""
tests/test_capture.py

Cadence maximale par caméra (core/capture.py, Throttle) : horloge simulée,
source à 30 i/s.
"""

import pytest

from core.capture import Throttle


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def allowed(throttle, clock, seconds, source_fps=30):
    count = 0
    for _ in range(int(seconds * source_fps)):
        count += throttle.allow()
        clock.now += 1 / source_fps
    return count


@pytest.mark.parametrize("target", [5, 10, 15, 30])
def test_average_rate_matches_the_target(target):
    clock = Clock()

    count = allowed(Throttle(target, clock), clock, seconds=10)

    assert count == pytest.approx(10 * target, abs=1)


def test_zero_lets_every_frame_through():
    clock = Clock()

    assert allowed(Throttle(0, clock), clock, seconds=2) == 60


def test_a_pause_in_the_source_does_not_cause_a_burst():
    """Après une coupure de la source, pas de rattrapage : la cadence reprend."""
    clock = Clock()
    throttle = Throttle(10, clock)
    allowed(throttle, clock, seconds=1)

    clock.now += 5.0
    count = allowed(throttle, clock, seconds=1)

    assert count == pytest.approx(10, abs=1)


def test_the_rate_can_change_while_running():
    clock = Clock()
    throttle = Throttle(30, clock)
    assert allowed(throttle, clock, seconds=1) == pytest.approx(30, abs=1)

    throttle.set_rate(5)

    assert allowed(throttle, clock, seconds=2) == pytest.approx(10, abs=1)
