"""
tests/test_adaptive.py

Réglages adaptés au nombre de caméras actives (core/adaptive.py) : la
politique (fonction pure), la priorité des réglages fixés, et le contrôleur
quand une caméra tombe ou revient (horloge simulée).
"""

from types import SimpleNamespace

import pytest

from core import adaptive
from core.adaptive import AdaptiveController, Tuning, plan, update_in_place

PROCESS_FPS, GPU_FPS = 72, 150


@pytest.fixture(autouse=True)
def nothing_fixed(monkeypatch):
    monkeypatch.setattr(adaptive.config, "CAMERA_MAX_FPS_FIXED", False)
    monkeypatch.setattr(adaptive.config, "FACE_RECOGNITION_SKIP_FIXED", False)


@pytest.mark.parametrize("active, expected", [
    ([1], Tuning(0, 2)),        # 30 i/s sur ~65 : reconnaissance plus fréquente
    ([2], Tuning(0, 5)),        # 60 i/s : tenu, mais pas avec le coût de « toutes les 2 »
    ([3], Tuning(23, 10)),      # 90 i/s : même toutes les 10 ne suffit pas
    ([4], Tuning(17, 10)),      # 120 i/s dans un seul processus : cadence réduite
])
def test_one_process_gets_lighter_as_cameras_arrive(active, expected):
    assert plan(active, PROCESS_FPS, GPU_FPS) == [expected]


def test_spreading_cameras_over_processes_keeps_full_rate():
    """4 caméras en 2 processus : 60 i/s chacun, toutes les images restent traitées."""
    assert plan([2, 2], PROCESS_FPS, GPU_FPS) == [Tuning(0, 5), Tuning(0, 5)]


def test_sixteen_cameras_in_two_processes_drop_to_a_low_rate():
    tunings = plan([8, 8], PROCESS_FPS, GPU_FPS)

    assert all(t.recognition_skip == 10 for t in tunings)
    assert all(0 < t.max_fps <= 7 for t in tunings)


def test_the_gpu_share_caps_each_process():
    """Avec un GPU plus faible, la part du GPU borne avant le GIL."""
    unbounded = plan([2, 2], PROCESS_FPS, gpu_fps=1000)
    bounded = plan([2, 2], PROCESS_FPS, gpu_fps=60)

    assert unbounded == [Tuning(0, 5), Tuning(0, 5)]
    assert all(t.max_fps and t.max_fps < 30 for t in bounded)


def test_a_process_without_active_camera_gets_defaults():
    assert plan([0, 1], PROCESS_FPS, GPU_FPS)[1] == Tuning(0, 2)


def test_fixed_settings_win(monkeypatch):
    monkeypatch.setattr(adaptive.config, "CAMERA_MAX_FPS_FIXED", True)
    monkeypatch.setattr(adaptive.config, "CAMERA_MAX_FPS", 12.0)

    assert adaptive.with_overrides(Tuning(0, 2)) == Tuning(12.0, 2)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def camera(cid, last_result):
    return SimpleNamespace(id=cid, last_result=last_result, tuning=Tuning(0, 5))


def make_controller(groups, clock):
    applied = []

    def apply(cam, tuning):
        update_in_place(cam.tuning, tuning)
        applied.append((cam.id, tuning.max_fps, tuning.recognition_skip))

    return AdaptiveController(groups, apply, clock), applied


def test_a_camera_that_stops_frees_capacity_for_the_others(monkeypatch):
    monkeypatch.setattr(adaptive.config, "ADAPTIVE_PROCESS_FPS", PROCESS_FPS)
    monkeypatch.setattr(adaptive.config, "ADAPTIVE_GPU_FPS", GPU_FPS)
    clock = Clock()
    cams = [camera(f"c{i}", clock.now) for i in range(4)]
    controller, applied = make_controller([cams], clock)

    controller.step()
    assert {t[1:] for t in applied} == {(17, 10)}

    # c3 tombe : ses images n'arrivent plus ; les trois autres continuent.
    clock.now += 10
    for c in cams[:3]:
        c.last_result = clock.now
    applied.clear()
    assert controller.step()
    assert sorted(applied) == [("c0", 23, 10), ("c1", 23, 10), ("c2", 23, 10)]

    # c3 revient : retour au partage à quatre. c3 a gardé ses réglages d'avant
    # la coupure, déjà les bons : seules les trois autres redescendent.
    cams[3].last_result = clock.now
    applied.clear()
    controller.step()
    assert sorted(applied) == [("c0", 17, 10), ("c1", 17, 10), ("c2", 17, 10)]
    assert all(c.tuning == Tuning(17, 10) for c in cams)


def test_nothing_is_pushed_while_the_active_set_is_unchanged(monkeypatch):
    clock = Clock()
    cams = [camera("c0", clock.now)]
    controller, applied = make_controller([cams], clock)
    controller.step()
    applied.clear()

    assert not controller.step()
    assert applied == []


def test_a_camera_that_never_sent_anything_is_inactive():
    clock = Clock()
    controller, applied = make_controller([[camera("c0", None)]], clock)

    controller.step()

    assert applied == []


# ─────────────────────────────────────────────────────────────────────────────
# Boucle de retour : cadence visée non tenue
# ─────────────────────────────────────────────────────────────────────────────

def running_cameras(clock, count, fps):
    cams = [camera(f"c{i}", clock.now) for i in range(count)]
    for c in cams:
        c.state = SimpleNamespace(fps=fps)
    return cams


def keep_sending(cams, clock):
    for c in cams:
        c.last_result = clock.now


def test_rates_not_held_lower_the_capacity_after_the_grace_period(monkeypatch):
    monkeypatch.setattr(adaptive.config, "ADAPTIVE_PROCESS_FPS", PROCESS_FPS)
    monkeypatch.setattr(adaptive.config, "ADAPTIVE_GPU_FPS", GPU_FPS)
    clock = Clock()
    cams = running_cameras(clock, 4, fps=0)
    controller, applied = make_controller([cams], clock)
    controller.step()                                  # 4 caméras : 17 i/s visés
    for c in cams:
        c.state.fps = 14.0                             # 82 % de la cible

    clock.now += 10
    keep_sending(cams, clock)
    applied.clear()
    assert not controller.step()                       # chauffe : pas d'évaluation

    clock.now += AdaptiveController.GRACE_S
    for _ in range(AdaptiveController.SHORTFALL_STEPS - 1):
        keep_sending(cams, clock)
        assert not controller.step()
        clock.now += 2
    keep_sending(cams, clock)
    assert controller.step()

    assert controller.scale == pytest.approx(14 / 17, abs=0.01)
    assert {t[1] for t in applied} == {14}             # floor(17 × 0,82 …)


def test_held_rates_change_nothing():
    clock = Clock()
    cams = running_cameras(clock, 4, fps=0)
    controller, applied = make_controller([cams], clock)
    controller.step()
    for c in cams:
        c.state.fps = c.tuning.max_fps or 30

    for _ in range(10):
        clock.now += AdaptiveController.GRACE_S
        keep_sending(cams, clock)
        controller.step()

    assert controller.scale == 1.0
