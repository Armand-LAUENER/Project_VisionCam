"""
system_probe.py — Mesures système pour le journal d'endurance (core/endurance.py).

Trois sources, sans dépendance ajoutée :

- psutil (déjà installé avec ultralytics) : CPU, threads et descripteurs du
  processus ; CPU, RAM et swap vus par WSL ;
- NVML, par ctypes sur libnvidia-ml : utilisation, mémoire, température,
  puissance, fréquence et causes de bridage du GPU entier. Sous WSL2, NVML ne
  donne ni la mémoire par processus (« non disponible ») ni des PID WSL ;
- les compteurs de performance de Windows, lus par powershell.exe depuis WSL :
  par processus Windows, mémoire GPU dédiée et utilisation des moteurs. Tout
  WSL y apparaît sous un seul processus, `vmwp` : pendant un run où VisionCam
  est le seul programme GPU de WSL, c'est donc lui. S'y ajoutent le CPU et la
  RAM disponible de la machine.

Les colonnes `ctx_*` décrivent l'environnement (GPU entier, Windows, WSL) :
le rapport les affiche sans les juger. Une source absente (pas de GPU NVIDIA,
pas de Windows) ne donne simplement pas ses colonnes.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import shutil
import subprocess
import threading
from collections import defaultdict

import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Processus et WSL (psutil)
# ─────────────────────────────────────────────────────────────────────────────


def process_stats(proc: psutil.Process) -> dict:
    """CPU (% d'un cœur, depuis l'appel précédent), threads et descripteurs ouverts."""
    with proc.oneshot():
        fds = proc.num_fds() if hasattr(proc, "num_fds") else len(proc.open_files())
        return {"proc_cpu_pct": round(proc.cpu_percent(None), 1),
                "size_threads": proc.num_threads(), "size_fds": fds}


def system_stats() -> dict:
    """CPU (% de toute la VM, depuis l'appel précédent), RAM et swap vus par WSL."""
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    return {"ctx_wsl_cpu_pct": psutil.cpu_percent(None),
            "ctx_wsl_ram_used_mb": round(memory.used / 2**20),
            "ctx_wsl_ram_available_mb": round(memory.available / 2**20),
            "ctx_wsl_swap_used_mb": round(swap.used / 2**20),
            "ctx_wsl_load_1m": round(os.getloadavg()[0], 2)}


# ─────────────────────────────────────────────────────────────────────────────
# GPU entier (NVML)
# ─────────────────────────────────────────────────────────────────────────────

class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Nvml:
    """GPU 0 par NVML ; stats() vide si la bibliothèque ou le GPU manque."""

    NVML_CLOCK_SM = 1

    def __init__(self, loader=ctypes.CDLL) -> None:
        self._lib = self._handle = None
        try:
            lib = loader("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return
            self._lib, self._handle = lib, handle
        except (OSError, AttributeError):
            pass

    def stats(self) -> dict:
        if self._lib is None:
            return {}
        lib, h, out = self._lib, self._handle, {}
        util, memory = _Utilization(), _Memory()
        value, wide = ctypes.c_uint(), ctypes.c_ulonglong()
        if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util)) == 0:
            out["ctx_gpu_util_pct"] = util.gpu
        if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(memory)) == 0:
            out["ctx_gpu_mem_used_mb"] = round(memory.used / 2**20)
        if lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(value)) == 0:
            out["ctx_gpu_temp_c"] = value.value
        if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(value)) == 0:
            out["ctx_gpu_power_w"] = round(value.value / 1000, 1)
        if lib.nvmlDeviceGetClockInfo(h, self.NVML_CLOCK_SM, ctypes.byref(value)) == 0:
            out["ctx_gpu_sm_clock_mhz"] = value.value
        # Masque des causes de bridage (0x1 : GPU au repos ; 0x4 : limite de
        # puissance ; 0x20/0x40 : thermique). Nom changé dans les pilotes récents.
        for name in ("nvmlDeviceGetCurrentClocksEventReasons",
                     "nvmlDeviceGetCurrentClocksThrottleReasons"):
            fn = getattr(lib, name, None)
            if fn is not None and fn(h, ctypes.byref(wide)) == 0:
                out["ctx_gpu_throttle_mask"] = wide.value
                break
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Compteurs Windows (powershell.exe depuis WSL)
# ─────────────────────────────────────────────────────────────────────────────

_WINDOWS_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$names = @{}; Get-Process | ForEach-Object { $names[[string]$_.Id] = $_.ProcessName }
function PidOf($i) { [regex]::Match($i, 'pid_(\d+)').Groups[1].Value }
$memory = @((Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples |
  Where-Object { $_.CookedValue -gt 0 } | ForEach-Object {
    $p = PidOf $_.InstanceName
    [pscustomobject]@{ pid = [int]$p; name = $names[$p]; mb = [math]::Round($_.CookedValue / 1MB, 1) } })
$sets = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 2
$engines = @($sets[-1].CounterSamples | Where-Object { $_.CookedValue -gt 0 } | ForEach-Object {
    $p = PidOf $_.InstanceName
    $e = [regex]::Match($_.InstanceName, 'engtype_(\w+)').Groups[1].Value
    [pscustomobject]@{ pid = [int]$p; name = $names[$p]; engine = $e; pct = [math]::Round($_.CookedValue, 2) } })
# CIM plutôt que Get-Counter : les noms des compteurs CPU et mémoire sont
# traduits selon la langue de Windows, ceux du GPU non.
$cpu = (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average
$ram = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB)
@{ memory = $memory; engines = $engines; cpu_pct = $cpu; ram_available_mb = $ram } |
  ConvertTo-Json -Depth 4 -Compress
"""


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_windows_counters(text: str) -> dict:
    """Sortie JSON du script → colonnes ctx_* ; {} si elle est illisible."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}

    def is_wsl(entry):
        return str(entry.get("name") or "").lower() == "vmwp"

    out = {}
    memory = _as_list(data.get("memory"))
    # Seule mesure de la VRAM de VisionCam (seul programme GPU de WSL pendant un
    # run) : jugée par le rapport, contrairement au reste du contexte.
    out["wsl_vram_mb"] = round(sum(m["mb"] for m in memory if is_wsl(m)), 1)
    out["ctx_windows_vram_mb"] = round(sum(m["mb"] for m in memory if not is_wsl(m)), 1)

    # Par processus : somme des instances d'un même type de moteur, puis le
    # moteur le plus chargé (ce qu'affiche le Gestionnaire des tâches).
    per_engine = defaultdict(lambda: defaultdict(float))
    names, wsl_pids = {}, set()
    for e in _as_list(data.get("engines")):
        per_engine[e["pid"]][e["engine"]] += e["pct"]
        names[e["pid"]] = e.get("name") or str(e["pid"])
        if is_wsl(e):
            wsl_pids.add(e["pid"])
    busiest = {pid: max(engines.values()) for pid, engines in per_engine.items()}
    out["ctx_wsl_gpu_pct"] = round(sum(v for p, v in busiest.items() if p in wsl_pids), 1)
    windows = {p: v for p, v in busiest.items() if p not in wsl_pids}
    out["ctx_windows_gpu_pct"] = round(sum(windows.values()), 1)
    out["ctx_windows_top_gpu"] = names[max(windows, key=windows.get)] if windows else ""
    if data.get("cpu_pct") is not None:
        out["ctx_host_cpu_pct"] = round(data["cpu_pct"], 1)
    if data.get("ram_available_mb") is not None:
        out["ctx_host_ram_available_mb"] = data["ram_available_mb"]
    return out


class WindowsCounters:
    """Lit les compteurs Windows dans un thread, toutes les `interval_s` : un appel prend ~3 s."""

    def __init__(self, interval_s: float = 60.0) -> None:
        self.interval_s = interval_s
        self.available = shutil.which("powershell.exe") is not None
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        encoded = base64.b64encode(_WINDOWS_SCRIPT.encode("utf-16-le")).decode()
        self._command = ["powershell.exe", "-NoProfile", "-NonInteractive",
                         "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded]

    def start(self) -> None:
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="windows-counters", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(self._command, capture_output=True, timeout=30)
                # Les erreurs de PowerShell sortent dans la page de code de la console.
                stats = parse_windows_counters(result.stdout.decode("utf-8", "replace"))
            except (OSError, subprocess.TimeoutExpired):
                stats = {}
            with self._lock:
                self._latest = stats
            self._stop.wait(self.interval_s)
