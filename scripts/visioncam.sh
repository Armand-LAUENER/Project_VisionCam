#!/usr/bin/env bash
# visioncam.sh — Lance ou arrête VisionCam et son pont webcam Windows, depuis WSL.
#
#   scripts/visioncam.sh start    pont webcam (si caméra distante) puis application
#   scripts/visioncam.sh stop     application puis pont webcam
#   scripts/visioncam.sh restart  application seule (le pont reste actif)
#   scripts/visioncam.sh status   état des deux
#   scripts/visioncam.sh logs     suit le journal de l'application
#
# L'application tourne en arrière-plan : journal dans logs/visioncam.log,
# PID dans data/visioncam.pid. Rien ne démarre tout seul avec Windows.
# FLASK_PORT, USE_LOCAL_CAM et REMOTE_SOURCE se lisent dans l'environnement,
# puis dans .env.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/data/visioncam.pid"
LOG_FILE="$ROOT/logs/visioncam.log"
APP_START_TIMEOUT=180   # chargement des modèles et des moteurs TensorRT
BRIDGE_START_TIMEOUT=30

env_value() {  # env_value CLE DEFAUT — environnement, puis .env, puis défaut
    local key="$1" default="$2" value
    if [[ -n "${!key:-}" ]]; then
        printf '%s' "${!key}"
        return
    fi
    value="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ROOT/.env" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    printf '%s' "${value:-$default}"
}

PORT="$(env_value FLASK_PORT 5000)"
USE_LOCAL_CAM="$(env_value USE_LOCAL_CAM true)"
REMOTE_SOURCE="$(env_value REMOTE_SOURCE "")"
BRIDGE_BASE="${REMOTE_SOURCE%/video*}"

http_code() {
    curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$1" || true
}

app_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(cat "$PID_FILE")"
    kill -0 "$pid" 2>/dev/null && printf '%s' "$pid"
}

port_in_use() {
    ss -ltn "sport = :$PORT" | grep -q LISTEN
}

uses_bridge() {
    [[ "${USE_LOCAL_CAM,,}" != "true" && -n "$REMOTE_SOURCE" ]]
}

bridge_up() {
    [[ "$(http_code "$BRIDGE_BASE/snapshot")" == 200 ]]
}

run_powershell() {
    powershell.exe -NoProfile -NonInteractive -Command "$1" | tr -d '\r'
}

start_bridge() {
    if ! uses_bridge; then
        echo "Caméra locale (USE_LOCAL_CAM=true) : pas de pont webcam."
        return
    fi
    if bridge_up; then
        echo "Pont webcam déjà actif : $BRIDGE_BASE"
        return
    fi
    if ! command -v powershell.exe >/dev/null; then
        echo "powershell.exe introuvable : lancer le pont côté Windows (py -3.11 tools\\webcam_bridge.py)." >&2
        return 1
    fi
    # Chemin //wsl.localhost/... : Python sous Windows le lit directement.
    local script="//wsl.localhost/${WSL_DISTRO_NAME:-Ubuntu}$ROOT/tools/webcam_bridge.py"
    echo "Démarrage du pont webcam Windows…"
    run_powershell "\$py = (Get-Command py -ErrorAction SilentlyContinue).Source;
        if (-not \$py) { \$py = \"\$env:LOCALAPPDATA\\Programs\\Python\\Launcher\\py.exe\" };
        Start-Process -FilePath \$py -ArgumentList '-3.11','$script' -WindowStyle Minimized"
    for ((i = 0; i < BRIDGE_START_TIMEOUT; i++)); do
        if bridge_up; then
            echo "Pont webcam actif : $BRIDGE_BASE"
            return
        fi
        sleep 1
    done
    echo "Le pont webcam ne répond pas sur $BRIDGE_BASE (webcam occupée ? pare-feu ?)." >&2
    return 1
}

stop_bridge() {
    uses_bridge || return 0
    command -v powershell.exe >/dev/null || return 0
    local stopped
    stopped="$(run_powershell "Get-CimInstance Win32_Process |
        Where-Object { \$_.CommandLine -like '*webcam_bridge.py*' -and \$_.Name -ne 'powershell.exe' } |
        ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; \$_.ProcessId }")"
    if [[ -n "$stopped" ]]; then
        echo "Pont webcam arrêté."
    else
        echo "Pont webcam : déjà arrêté."
    fi
}

start_app() {
    local pid
    if pid="$(app_pid)"; then
        echo "VisionCam tourne déjà (PID $pid) : http://localhost:$PORT"
        return
    fi
    if port_in_use; then
        echo "Le port $PORT est déjà pris par un autre processus (VisionCam lancé à la main ?)." >&2
        echo "L'arrêter, ou choisir un autre port : FLASK_PORT=5001 $0 start" >&2
        return 1
    fi
    mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
    echo "Démarrage de VisionCam (journal : ${LOG_FILE#"$ROOT"/})…"
    cd "$ROOT"
    # Le python du venv directement, pas `uv run` : le PID enregistré est
    # celui qui reçoit SIGTERM à l'arrêt.
    FLASK_PORT="$PORT" nohup "$ROOT/.venv/bin/python" app.py >>"$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" >"$PID_FILE"
    for ((i = 0; i < APP_START_TIMEOUT; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo "VisionCam s'est arrêté au démarrage. Fin du journal :" >&2
            tail -n 20 "$LOG_FILE" >&2
            return 1
        fi
        if [[ "$(http_code "http://127.0.0.1:$PORT/login")" =~ ^(200|302)$ ]]; then
            echo "VisionCam prêt : http://localhost:$PORT (PID $pid)"
            return
        fi
        sleep 1
    done
    echo "VisionCam ne répond pas après ${APP_START_TIMEOUT} s ; voir $LOG_FILE" >&2
    return 1
}

stop_app() {
    local pid
    if ! pid="$(app_pid)"; then
        rm -f "$PID_FILE"
        echo "VisionCam : déjà arrêté."
        return
    fi
    kill -TERM "$pid"
    for ((i = 0; i < 40; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo "VisionCam arrêté."
            return
        fi
        sleep 0.5
    done
    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "VisionCam ne s'arrêtait pas : processus tué."
}

status() {
    local pid
    if pid="$(app_pid)"; then
        echo "VisionCam   : actif (PID $pid) — http://localhost:$PORT"
    elif port_in_use; then
        echo "VisionCam   : port $PORT occupé par un processus lancé hors de ce script"
    else
        echo "VisionCam   : arrêté"
    fi
    if ! uses_bridge; then
        echo "Pont webcam : inutile (caméra locale)"
    elif bridge_up; then
        echo "Pont webcam : actif — $BRIDGE_BASE"
    else
        echo "Pont webcam : arrêté"
    fi
}

case "${1:-}" in
    start) start_bridge && start_app ;;
    stop) stop_app; stop_bridge ;;
    restart) stop_app; start_bridge && start_app ;;
    status) status ;;
    logs) tail -n 50 -f "$LOG_FILE" ;;
    *)
        echo "Usage : $0 {start|stop|restart|status|logs}" >&2
        exit 2
        ;;
esac
