"""
check_sessions.py — Compare l'historique de présence aux sessions recalculées des événements.

L'historique (table `sessions` de presence.db) est calculé au fil de l'eau ;
core/event_log.sessions_from_events() le recalcule à partir des événements
bruts (table `events`). Les deux doivent coïncider, à une image près : c'est
ce qui permet de faire des événements la donnée source (roadmap 2.1).

Seules les sessions arrivées après le premier événement sont comparées :
l'historique plus ancien n'a pas d'événements. Base ouverte en lecture seule.

Depuis la racine du projet :

    uv run -m tools.check_sessions data/presence.db
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime

import config
from core.event_log import sessions_from_events

TOLERANCE_S = 2.0


def compare(stored, rebuilt, tolerance=TOLERANCE_S):
    """Apparie les sessions par nom et heure d'arrivée ; retourne (paires, seules stockées, seules recalculées)."""
    pairs, left = [], list(rebuilt)
    only_stored = []
    for s in stored:
        match = next((r for r in left if r["name"] == s["name"]
                      and abs(r["arrived"] - s["arrived"]) <= tolerance), None)
        if match is None:
            only_stored.append(s)
            continue
        left.remove(match)
        pairs.append((s, match))
    return pairs, only_stored, left


def _fmt(t):
    return datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S") if t else "en cours"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("db", help="presence.db")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        events = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id")]
    except sqlite3.OperationalError:
        raise SystemExit("Pas de table events : l'application n'a pas encore journalisé d'événement.")
    if not events:
        raise SystemExit("Aucun événement.")
    first = min(e["time"] for e in events)
    stored = [dict(r) for r in conn.execute(
        "SELECT name, arrived, departed FROM sessions WHERE arrived >= ? ORDER BY arrived",
        (first - TOLERANCE_S,))]
    conn.close()

    rebuilt = sessions_from_events(events, config.PRESENCE_LOG_GAP_S,
                                   config.PRESENCE_FACE_MAX_AGE_S)
    pairs, only_stored, only_rebuilt = compare(stored, rebuilt)
    print(f"{len(events)} événements depuis {_fmt(first)} ; "
          f"{len(stored)} sessions stockées, {len(rebuilt)} recalculées")
    worst = 0.0
    for s, r in pairs:
        if s["departed"] and r["departed"]:
            delta = r["departed"] - s["departed"]
            worst = max(worst, abs(delta))
            note = f"départ {delta:+.1f} s"
        else:
            note = "en cours" if not s["departed"] and not r["departed"] else "une seule en cours"
        print(f"  {s['name']:<20} {_fmt(s['arrived'])} → {_fmt(s['departed'])}   {note}")
    for s in only_stored:
        print(f"  seulement stockée   : {s['name']} {_fmt(s['arrived'])} → {_fmt(s['departed'])}")
    for r in only_rebuilt:
        print(f"  seulement recalculée : {r['name']} {_fmt(r['arrived'])} → {_fmt(r['departed'])}")
    ok = not only_stored and not only_rebuilt and worst <= TOLERANCE_S
    print(f"\nVerdict : {'identiques' if ok else 'DIFFÉRENTES'} "
          f"(écart de départ max {worst:.1f} s, tolérance {TOLERANCE_S:.0f} s)")


if __name__ == "__main__":
    main()
