"""
history.py — Historique de présence servi par l'API, recalculé des événements.

Roadmap 2.1 : le journal d'événements (core/event_log.py) est la donnée
source. Les sessions sont recalculées à chaque requête par
sessions_from_events(), avec la règle de l'historique (une piste visible porte
le nom, visage reconnu depuis moins de `face_max_age`, session fermée après
`gap` sans présence), vérifiée identique au calcul au fil de l'eau
(tools/check_sessions.py).

La table `sessions` (core/presence_log.py) reste écrite au fil de l'eau, mais
ne sert plus que de cache : l'API n'en lit que les sessions arrivées avant le
premier événement, l'historique d'avant le journal. Les deux suivent la même
rétention, et renommer ou effacer une personne les touche toutes deux.
"""

from __future__ import annotations

from core.event_log import EventLog, sessions_from_events
from core.presence_log import PresenceLog, _iso


def _api_session(s: dict) -> dict:
    """Session recalculée → forme de /api/history (celle de PresenceLog.sessions)."""
    end = s["departed"] if s["departed"] is not None else s["last_seen"]
    return {
        # Pas de numéro SQLite : identifiant stable tiré du nom et de l'arrivée.
        "id": f"{s['name']}@{s['arrived']:.3f}", "name": s["name"],
        "arrived": _iso(s["arrived"]), "departed": _iso(s["departed"]),
        "last_seen": _iso(s["last_seen"]), "ongoing": s["departed"] is None,
        "duration_s": round(end - s["arrived"]),
    }


def presence_history(presence_log: PresenceLog, event_log: EventLog, gap: float,
                     face_max_age: float, now: float, name: str | None = None,
                     start: float | None = None, end: float | None = None,
                     limit: int = 1000) -> list[dict]:
    """Sessions qui chevauchent [start, end[, les plus récentes d'abord."""
    events = event_log.events(until=end)
    first_event = min((e["time"] for e in events), default=None)

    recalculated = []
    for s in sessions_from_events(events, gap, face_max_age, now=now):
        if name and s["name"] != name:
            continue
        if start is not None and s["last_seen"] < start:
            continue
        if end is not None and s["arrived"] >= end:
            continue
        recalculated.append((s["arrived"], _api_session(s)))

    # Historique d'avant le journal : seule la table le connaît.
    legacy_end = first_event if end is None or first_event is None else min(end, first_event)
    legacy = presence_log.sessions(name=name, start=start, end=legacy_end, limit=limit)

    merged = sorted(recalculated, key=lambda pair: -pair[0])
    result = [s for _, s in merged] + legacy      # les anciennes sont toutes plus vieilles
    return result[:limit]


def history_names(presence_log: PresenceLog, event_log: EventLog) -> list[str]:
    """Noms présents dans l'historique, toutes sources confondues."""
    names = {e["name"] for e in event_log.events() if e["name"]}
    return sorted(names | set(presence_log.names()))
