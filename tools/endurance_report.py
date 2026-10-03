"""
endurance_report.py — Dit si un test d'endurance est stable, colonne par colonne.

Lit le CSV écrit par l'application avec ENDURANCE_LOG_PATH (core/endurance.py)
et juge chaque mesure après le warm-up (moteurs TensorRT, caches) :

- mémoire (`*_mb`) : la dérive linéaire sur la durée mesurée ne dépasse pas
  max(50 Mo, 5 % de la moyenne) ;
- FPS : la dérive ne fait pas perdre plus de 5 % de la moyenne ;
- temps par étape et latence (`stage_*_ms`, `latency_*_ms`) : la dérive ne
  dépasse pas max(20 ms, 10 %) ;
- CPU du processus (`proc_cpu_pct`) : la dérive ne dépasse pas max(5 points, 10 %) ;
- structures internes (`size_*`, `visible_tracks`) : le maximum du dernier
  quart de la mesure ne dépasse pas 1,5 × celui du premier quart + 5. Leur
  taille suit le nombre de personnes : bornée, pas constante.

Le contexte (`ctx_*` : GPU entier, Windows, WSL) et les compteurs (images
traitées, jetées, sautées) sont affichés, pas jugés : ils bougent avec ce que
fait le reste de la machine, et servent à expliquer une chute.

Ces seuils sont des garde-fous, pas des tests statistiques : une colonne
« instable » se regarde à la main dans le CSV.

Depuis la racine du projet :

    uv run -m tools.endurance_report data/endurance.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import numpy as np

MIN_ROWS = 10
# Croissent par construction (compteur, historique de présence) : affichés, pas jugés.
COUNTERS = ("frames", "frames_dropped", "source_skipped", "presence_db_kb")
CONTEXT_PREFIX = "ctx_"
HOUR = 3600.0


@dataclass
class Result:
    column: str
    start: float          # moyenne du premier quart après warm-up
    end: float            # moyenne du dernier quart
    drift: float          # pente × durée mesurée (même unité que la colonne)
    stable: bool | None   # None : pas assez de mesures
    rule: str


def _numeric(rows, column):
    """(temps, valeurs) des lignes où la colonne est remplie ; None si elle n'est pas numérique.

    Une case vide est une mesure pas encore disponible (compteurs Windows,
    étape pas encore passée), pas un zéro.
    """
    filled = [r for r in rows if r.get(column) not in (None, "")]
    try:
        return (np.array([float(r["elapsed_s"]) for r in filled]),
                np.array([float(r[column]) for r in filled]))
    except (TypeError, ValueError):
        return None


def analyze(rows, warmup_s: float = 1800) -> list[Result]:
    columns = ([c for c in rows[0] if c not in ("time", "elapsed_s", *COUNTERS)
                and not c.startswith(CONTEXT_PREFIX)] if rows else [])
    measured = [r for r in rows if float(r["elapsed_s"]) >= warmup_s]
    results = []
    for column in columns:
        if len(measured) < MIN_ROWS:
            results.append(Result(column, float("nan"), float("nan"), float("nan"), None,
                                  f"moins de {MIN_ROWS} mesures après le warm-up"))
            continue
        series = _numeric(measured, column)
        if series is None or len(series[1]) < MIN_ROWS:
            continue
        t, values = series
        quarter = max(1, len(values) // 4)
        start, end = values[:quarter].mean(), values[-quarter:].mean()
        slope = np.polyfit(t, values, 1)[0] if np.ptp(t) > 0 else 0.0
        drift = float(slope * (t[-1] - t[0]))
        mean = float(values.mean())
        if column.startswith("size_") or column == "visible_tracks":
            limit = 1.5 * values[:quarter].max() + 5
            stable = bool(values[-quarter:].max() <= limit)
            rule = f"max fin ≤ {limit:.0f}"
        elif column.startswith(("latency_", "stage_")):
            limit = max(20.0, 0.10 * mean)
            stable = drift <= limit
            rule = f"dérive ≤ {limit:.0f}"
        elif column == "proc_cpu_pct":
            limit = max(5.0, 0.10 * mean)
            stable = drift <= limit
            rule = f"dérive ≤ {limit:.0f} points"
        elif column == "fps":
            stable = drift >= -0.05 * mean
            rule = f"dérive ≥ {-0.05 * mean:.2f}"
        else:
            limit = max(50.0, 0.05 * mean)
            stable = drift <= limit
            rule = f"dérive ≤ {limit:.0f}"
        results.append(Result(column, float(start), float(end), drift, stable, rule))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", help="journal écrit avec ENDURANCE_LOG_PATH")
    parser.add_argument("--warmup-min", type=float, default=30,
                        help="minutes ignorées au début (défaut : 30)")
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Journal vide.")
    duration = float(rows[-1]["elapsed_s"]) / HOUR
    gaps = np.diff([float(r["elapsed_s"]) for r in rows])
    print(f"{len(rows)} mesures sur {duration:.2f} h ; warm-up ignoré : {args.warmup_min:.0f} min")
    if "frames" in rows[0] and duration > 0:
        frames = float(rows[-1]["frames"]) - float(rows[0]["frames"])
        print(f"Images traitées : {frames:.0f}, soit {frames / (duration * HOUR):.1f} i/s en moyenne")
    for counter in ("frames_dropped", "source_skipped"):
        if rows[-1].get(counter):
            print(f"{counter} : {rows[-1][counter]} images")
    if "presence_db_kb" in rows[-1] and rows[-1]["presence_db_kb"]:
        first = next(r["presence_db_kb"] for r in rows if r.get("presence_db_kb"))
        print(f"Historique de présence : {first} → {rows[-1]['presence_db_kb']} Ko")
    if len(gaps) and gaps.max() > 3 * np.median(gaps):
        print(f"Attention : trou de {gaps.max():.0f} s dans le journal (pipeline bloqué ?)")

    results = analyze(rows, args.warmup_min * 60)
    print(f"\n{'mesure':<26} │ {'début':>9} {'fin':>9} {'dérive':>9} │ {'verdict':<9} règle")
    for r in results:
        verdict = {True: "stable", False: "INSTABLE", None: "—"}[r.stable]
        print(f"{r.column:<26} │ {r.start:>9.1f} {r.end:>9.1f} {r.drift:>+9.1f} │ "
              f"{verdict:<9} {r.rule}")
    context = [c for c in rows[0] if c.startswith(CONTEXT_PREFIX)]
    if context:
        print(f"\nContexte (non jugé) {'':<9} │ {'moyenne':>9} {'min':>9} {'max':>9}")
        for column in context:
            series = _numeric(rows, column)
            values = series[1] if series is not None and len(series[1]) else None
            if values is None:
                texts = [r[column] for r in rows if r.get(column)]
                top = max(set(texts), key=texts.count) if texts else "—"
                print(f"{column:<30} │ le plus fréquent : {top}")
            else:
                print(f"{column:<30} │ {values.mean():>9.1f} {values.min():>9.1f} {values.max():>9.1f}")

    judged = [r for r in results if r.stable is not None]
    if judged:
        print("\nVerdict : " + ("stable" if all(r.stable for r in judged)
                                else "instable : " + ", ".join(r.column for r in judged
                                                               if not r.stable)))


if __name__ == "__main__":
    main()
