"""
bench_load.py — Test de charge : combien de flux tient la machine, et à quelle cadence ?

Pour chaque palier (nombre de flux × cadence par caméra), lance l'application
complète sur N vidéos rejouées en `realtime` et en boucle (CAMERAS,
CAMERA_WORKERS=process, CAMERA_MAX_FPS), écarte la chauffe, puis relève :

  - i/s tenus par caméra (/api/diagnostics), et la part de la cadence visée ;
  - latence de bout en bout p50/p95, décodage, détection et reconnaissance
    p50 (journal d'endurance de l'application) ;
  - CPU et RAM de l'application et de ses processus de caméras (psutil) ;
  - GPU : utilisation (NVML), utilisation et VRAM de WSL vues par Windows.

Les mesures se font sur l'application telle qu'elle tourne : aucune page
ouverte (pas d'encodage JPEG), historique et journal d'événements dans un
dossier jetable. Les vidéos sont réutilisées si les flux sont plus nombreux.
Sous WSL2, la VRAM et l'utilisation GPU de WSL viennent des compteurs Windows
(core/system_probe.py).

Depuis la racine du projet :

    uv run -m tools.bench_load --videos ~/datasets/CHIRLA/videos --streams 2 4 8 16 \\
        --fps 5 10 30 --out /tmp/bench_load
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

import psutil

PORT = 5099
FIELDS = ["streams", "target_fps", "per_worker", "fps_mean", "fps_min", "held",
          "latency_p50_ms", "latency_p95_ms", "decode_p50_ms", "detection_p50_ms",
          "recognition_p50_ms", "frames_dropped", "cpu_pct", "rss_mb", "workers",
          "gpu_util_pct", "wsl_gpu_pct", "wsl_vram_mb"]


def find_videos(root: str) -> list[str]:
    videos = sorted(glob.glob(os.path.join(os.path.expanduser(root), "**", "*.avi"),
                              recursive=True)
                    + glob.glob(os.path.join(os.path.expanduser(root), "**", "*.mp4"),
                                recursive=True))
    if not videos:
        raise SystemExit(f"Aucune vidéo sous {root}")
    return videos


def cameras_spec(videos: list[str], streams: int) -> str:
    return ";".join(f"s{i}={videos[i % len(videos)]}" for i in range(streams))


def diagnostics() -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/diagnostics", timeout=5) as r:
            return json.load(r)
    except OSError:
        return None


def process_tree(pid: int):
    try:
        root = psutil.Process(pid)
        return [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _is_camera_worker(proc) -> bool:
    """Processus de caméras, à l'exclusion des autres enfants (powershell de la
    sonde Windows, core/system_probe.py)."""
    try:
        return "core.camera_worker" in proc.cmdline()
    except psutil.Error:
        return False


def median(rows: list[dict], column: str) -> float:
    values = [float(r[column]) for r in rows if r.get(column) not in (None, "")]
    return statistics.median(values) if values else float("nan")


def run_level(videos, streams, target_fps, per_worker, warmup, duration, out_dir) -> dict:
    """Un palier : lance l'application, mesure, l'arrête ; une ligne de résultats."""
    label = f"n{streams}_f{target_fps:g}"
    endurance = os.path.join(out_dir, f"endurance_{label}.csv")
    env = {**os.environ,
           "CAMERAS": cameras_spec(videos, streams), "CAMERA_WORKERS": "process",
           "CAMERAS_PER_WORKER": str(per_worker), "CAMERA_MAX_FPS": str(target_fps),
           "VIDEO_MODE": "realtime", "VIDEO_LOOP": "true", "FLASK_PORT": str(PORT),
           "ADMIN_PASSWORD_HASH": "", "ADMIN_PASSWORD": "",
           "PRESENCE_DB_PATH": os.path.join(out_dir, f"presence_{label}.db"),
           "ENDURANCE_LOG_PATH": endurance, "ENDURANCE_INTERVAL_S": "15"}
    with open(os.path.join(out_dir, f"app_{label}.log"), "w") as log:
        app = subprocess.Popen([sys.executable, "app.py"], env=env, stdout=log,
                               stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 300
        while diagnostics() is None:
            if app.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{label} : l'application n'a pas démarré")
            time.sleep(2)
        start = time.monotonic()
        time.sleep(warmup)

        fps_samples, cpu, rss = {}, [], []
        # Mêmes objets Process d'un relevé à l'autre : cpu_percent mesure
        # depuis l'appel précédent sur le même objet (0 au premier).
        procs = {p.pid: p for p in process_tree(app.pid)}
        for p in procs.values():
            p.cpu_percent(None)
        while time.monotonic() - start < warmup + duration:
            time.sleep(5)
            d = diagnostics()
            if d:
                for c in d["cameras"]:
                    fps_samples.setdefault(c["id"], []).append(c["fps"])
            for p in process_tree(app.pid):
                procs.setdefault(p.pid, p)
            alive = [p for p in procs.values() if p.is_running()]
            try:
                cpu.append(sum(p.cpu_percent(None) for p in alive))
                rss.append(sum(p.memory_info().rss for p in alive) / 2**20)
            except psutil.NoSuchProcess:
                continue
        workers = sum(1 for p in procs.values() if _is_camera_worker(p))
    finally:
        app.send_signal(signal.SIGTERM)
        try:
            app.wait(60)
        except subprocess.TimeoutExpired:
            app.kill()

    with open(endurance, newline="") as f:
        rows = [r for r in csv.DictReader(f) if float(r["elapsed_s"]) >= warmup]
    per_camera = [statistics.mean(v) for v in fps_samples.values()]
    fps_mean = statistics.mean(per_camera) if per_camera else 0.0
    return {
        "streams": streams, "target_fps": target_fps, "per_worker": per_worker,
        "fps_mean": round(fps_mean, 1), "fps_min": round(min(per_camera, default=0), 1),
        "held": fps_mean >= 0.95 * target_fps,
        "latency_p50_ms": median(rows, "stage_latency_p50_ms"),
        "latency_p95_ms": median(rows, "stage_latency_p95_ms"),
        "decode_p50_ms": median(rows, "stage_decode_p50_ms"),
        "detection_p50_ms": median(rows, "stage_detection_p50_ms"),
        "recognition_p50_ms": median(rows, "stage_recognition_p50_ms"),
        "frames_dropped": (int(rows[-1]["frames_dropped"]) - int(rows[0]["frames_dropped"])
                           if rows else 0),
        "cpu_pct": round(statistics.median(cpu)) if cpu else 0,
        "rss_mb": round(statistics.median(rss)) if rss else 0,
        "workers": workers,
        "gpu_util_pct": median(rows, "ctx_gpu_util_pct"),
        "wsl_gpu_pct": median(rows, "ctx_wsl_gpu_pct"),
        "wsl_vram_mb": median(rows, "wsl_vram_mb"),
    }


def default_per_worker(streams: int) -> int:
    """Caméras par processus : chaque processus pèse ~2,6 Go de RAM, WSL en a 15.

    2 jusqu'à 4 flux, 4 pour 8, 8 au-delà ; un processus tient ~72 i/s au total
    (GIL, docs/performance.md), assez pour 8 caméras à 5 ou 10 i/s.
    """
    return 2 if streams <= 4 else 4 if streams <= 8 else 8


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", required=True, help="dossier des vidéos à rejouer")
    parser.add_argument("--streams", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--fps", type=float, nargs="+", default=[5, 10, 30],
                        help="cadence visée par caméra (CAMERA_MAX_FPS)")
    parser.add_argument("--per-worker", type=int, default=0,
                        help="caméras par processus ; 0 : selon le nombre de flux (RAM)")
    parser.add_argument("--warmup", type=float, default=60)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--out", required=True, help="dossier des journaux et du CSV")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    videos = find_videos(args.videos)
    results_path = os.path.join(args.out, "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for streams in args.streams:
            per_worker = args.per_worker or default_per_worker(streams)
            for target_fps in args.fps:
                row = run_level(videos, streams, target_fps, per_worker, args.warmup,
                                args.duration, args.out)
                writer.writerow(row)
                f.flush()
                print(f"{streams:>3} flux × {target_fps:>4g} i/s : {row['fps_mean']:5.1f} i/s "
                      f"(min {row['fps_min']:.1f}) {'tenu' if row['held'] else 'NON TENU'}, "
                      f"latence p95 {row['latency_p95_ms']:.0f} ms, "
                      f"RAM {row['rss_mb'] / 1024:.1f} Go, WSL GPU {row['wsl_gpu_pct']:.0f} %",
                      flush=True)
    print(f"Résultats : {results_path}")


if __name__ == "__main__":
    main()
