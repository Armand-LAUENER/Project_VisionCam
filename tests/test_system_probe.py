"""
tests/test_system_probe.py

Couvre les mesures système du journal d'endurance : statistiques par étape sur
tout l'intervalle (et pas seulement la fenêtre glissante), lecture des
compteurs Windows (VRAM et utilisation de WSL, applications Windows), NVML
absent ou présent, CPU et RAM du processus et du système.

Lancer : pytest tests/test_system_probe.py -v
"""

import json

import pytest

from core.events import StageTimer
from core.system_probe import Nvml, parse_windows_counters, process_stats, system_stats


class TestIntervalStats:

    def test_covers_every_sample_since_the_last_drain(self):
        """La fenêtre glissante ne garde que 300 échantillons ; l'intervalle les garde tous."""
        timer = StageTimer(window=10)
        timer.track_intervals()
        for i in range(1000):
            timer.record("detection", 0.200 if i == 3 else 0.005)

        stats = timer.drain_intervals()

        assert stats["detection"]["samples"] == 1000
        assert stats["detection"]["max_ms"] == pytest.approx(200.0)
        assert stats["detection"]["median_ms"] == pytest.approx(5.0)
        assert timer.drain_intervals() == {}

    def test_nothing_kept_unless_enabled(self):
        timer = StageTimer()
        for _ in range(100):
            timer.record("detection", 0.005)

        assert timer.drain_intervals() == {}
        assert not timer._interval


WINDOWS_JSON = json.dumps({
    "memory": [{"pid": 13172, "name": "vmwp", "mb": 1517.0},
               {"pid": 31960, "name": "dwm", "mb": 2059.0},
               {"pid": 28620, "name": "wallpaper32", "mb": 252.0}],
    "engines": [{"pid": 13172, "name": "vmwp", "engine": "Compute_0", "pct": 41.5},
                {"pid": 13172, "name": "vmwp", "engine": "3D", "pct": 2.0},
                {"pid": 28620, "name": "wallpaper32", "engine": "3D", "pct": 18.0},
                {"pid": 31960, "name": "dwm", "engine": "3D", "pct": 3.0}],
    "cpu_pct": 23.4, "ram_available_mb": 9120.0})


class TestWindowsCounters:

    def test_wsl_and_windows_apps_are_separated(self):
        stats = parse_windows_counters(WINDOWS_JSON)

        assert stats == {
            "wsl_vram_mb": 1517.0,
            "ctx_wsl_gpu_pct": 41.5,          # moteur le plus chargé, comme le Gestionnaire des tâches
            "ctx_windows_gpu_pct": 21.0,      # applications Windows, moteur le plus chargé chacune
            "ctx_windows_vram_mb": 2311.0,
            "ctx_windows_top_gpu": "wallpaper32",
            "ctx_host_cpu_pct": 23.4,
            "ctx_host_ram_available_mb": 9120.0,
        }

    def test_several_vm_workers_are_summed(self):
        data = json.loads(WINDOWS_JSON)
        data["memory"].append({"pid": 777, "name": "vmwp", "mb": 100.0})

        assert parse_windows_counters(json.dumps(data))["wsl_vram_mb"] == 1617.0

    def test_unreadable_output_gives_nothing(self):
        assert parse_windows_counters("PowerShell n'a pas pu…") == {}


class TestNvml:

    def test_missing_library_gives_no_columns(self):
        def missing(_name):
            raise OSError("libnvidia-ml.so.1 introuvable")

        assert Nvml(loader=missing).stats() == {}


class TestLocalStats:

    def test_process_stats(self):
        import psutil

        stats = process_stats(psutil.Process())

        assert set(stats) == {"proc_cpu_pct", "size_threads", "size_fds"}
        assert stats["size_threads"] >= 1

    def test_system_stats(self):
        stats = system_stats()

        assert set(stats) == {"ctx_wsl_cpu_pct", "ctx_wsl_ram_used_mb", "ctx_wsl_ram_available_mb",
                              "ctx_wsl_swap_used_mb", "ctx_wsl_load_1m"}
        assert stats["ctx_wsl_ram_available_mb"] > 0
