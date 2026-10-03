"""
tests/test_endurance.py

Couvre le test d'endurance : le journal périodique (core/endurance.py) et son
analyse (tools/endurance_report.py), qui dit si mémoire, FPS et structures
internes restent stables. Horloge simulée, mesures synthétiques.

Lancer : pytest tests/test_endurance.py -v
"""

import csv

import numpy as np

from core.endurance import EnduranceLog
from tools.endurance_report import analyze

HOUR = 3600.0


class FakeClock:
    def __init__(self):
        self.now = 1_790_000_000.0

    def __call__(self):
        return self.now


class TestEnduranceLog:

    def test_one_row_per_interval_with_a_header(self, tmp_path):
        clock = FakeClock()
        samples = iter(range(100))
        log = EnduranceLog(str(tmp_path / "e.csv"), 60, lambda: {"rss_mb": next(samples)},
                           clock=clock)

        written = []
        for _ in range(10):   # une image toutes les 20 s pendant 200 s
            written.append(log.maybe_write())
            clock.now += 20
        log.close()

        assert written == [True, False, False, True, False, False, True, False, False, True]
        with open(tmp_path / "e.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["elapsed_s"] for r in rows] == ["0.0", "60.0", "120.0", "180.0"]
        assert [r["rss_mb"] for r in rows] == ["0", "1", "2", "3"]

    def test_columns_appearing_later_extend_the_header(self, tmp_path):
        """Compteurs Windows, étapes de reconnaissance : absents de la première ligne.

        Bug d'origine : l'en-tête était figé à la première ligne, et la deuxième,
        plus large, levait ValueError — dans le thread de traitement, arrêté net.
        """
        clock = FakeClock()
        rows = iter([{"fps": 30}, {"fps": 29, "ctx_host_cpu_pct": 12},
                     {"fps": 28, "stage_recognition_p95_ms": 9.5}, {"fps": 30}])
        log = EnduranceLog(str(tmp_path / "e.csv"), 60, lambda: next(rows), clock=clock)
        for _ in range(4):
            log.maybe_write()
            clock.now += 60
        log.close()

        with open(tmp_path / "e.csv", newline="") as f:
            reader = csv.DictReader(f)
            written = list(reader)
        assert reader.fieldnames == ["time", "elapsed_s", "fps", "ctx_host_cpu_pct",
                                     "stage_recognition_p95_ms"]
        assert [(r["fps"], r["ctx_host_cpu_pct"], r["stage_recognition_p95_ms"]) for r in written] == [
            ("30", "", ""), ("29", "12", ""), ("28", "", "9.5"), ("30", "", "")]

    def test_probe_is_not_called_between_rows(self, tmp_path):
        calls = []
        clock = FakeClock()
        log = EnduranceLog(str(tmp_path / "e.csv"), 60, lambda: calls.append(1) or {},
                           clock=clock)
        for _ in range(30):
            log.maybe_write()
        log.close()

        assert len(calls) == 1


def make_rows(hours=8, step_s=60, **columns):
    """Une ligne par minute ; chaque colonne est une fonction du temps écoulé (s)."""
    times = np.arange(0, hours * HOUR + 1, step_s)
    return [{"elapsed_s": t, **{name: f(t) for name, f in columns.items()}} for t in times]


def verdicts(rows, warmup_s=1800):
    return {r.column: r.stable for r in analyze(rows, warmup_s)}


class TestAnalyze:

    def test_flat_run_is_stable(self):
        rng = np.random.default_rng(0)
        rows = make_rows(rss_mb=lambda t: 2400 + rng.normal(0, 5),
                         fps=lambda t: 18 + rng.normal(0, 0.5),
                         size_identity_map=lambda t: rng.integers(0, 4))

        assert verdicts(rows) == {"rss_mb": True, "fps": True, "size_identity_map": True}

    def test_memory_leak_is_detected(self):
        """100 Mo/h : 750 Mo sur 7 h 30 après le warm-up."""
        rows = make_rows(rss_mb=lambda t: 2400 + 100 * t / HOUR)

        assert verdicts(rows) == {"rss_mb": False}

    def test_fps_decline_is_detected(self):
        rows = make_rows(fps=lambda t: 20 - 0.5 * t / HOUR)

        assert verdicts(rows) == {"fps": False}

    def test_growing_structure_is_detected(self):
        rows = make_rows(size_last_seen=lambda t: int(t // 60))

        assert verdicts(rows) == {"size_last_seen": False}

    def test_warmup_growth_is_ignored(self):
        """Moteurs TensorRT et caches remplis au démarrage : pas une fuite."""
        rows = make_rows(rss_mb=lambda t: 1500 + min(t, 1200) / 1200 * 900)

        assert verdicts(rows) == {"rss_mb": True}

    def test_frame_counter_is_not_judged(self):
        rows = make_rows(frames=lambda t: int(t * 30), rss_mb=lambda t: 2400)

        assert verdicts(rows) == {"rss_mb": True}

    def test_too_short_run_is_not_judged(self):
        rows = make_rows(hours=0.5, rss_mb=lambda t: 2400)

        assert verdicts(rows, warmup_s=1500) == {"rss_mb": None}
