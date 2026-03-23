"""
app.py — Serveur Flask pour le système de reconnaissance faciale.

Architecture :
  - Route '/'          → Page principale (dashboard HTML)
  - Route '/video_feed' → Flux MJPEG temps réel (frames annotées par l'IA)
  - Route '/api/stats'  → API JSON (statistiques en direct)

Le serveur :
  1. Instancie le SmartTracker (moteur d'IA)
  2. Ouvre la caméra dans un thread dédié pour ne pas bloquer Flask
  3. Génère un flux MJPEG consommé par la balise <img> du frontend
  4. Expose une API REST pour les statistiques (polling toutes les 2s)
  5. Libère proprement les ressources à l'arrêt (atexit + signal)
"""

import os
import sys
import time
import atexit
import signal
import threading
from typing import Generator, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

# Ajout du répertoire racine au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.tracker import SmartTracker


# =============================================================================
# APPLICATION FLASK
# =============================================================================

app = Flask(
    __name__,
    template_folder=os.path.join(config.BASE_DIR, "templates"),
    static_folder=os.path.join(config.BASE_DIR, "static")
)


# =============================================================================
# GESTIONNAIRE DE CAMÉRA (Thread-safe)
# =============================================================================

class CameraManager:
    """
    Gère la capture vidéo dans un thread séparé pour garantir
    un flux fluide indépendamment du traitement IA.

    Thread-safety assurée par un verrou (threading.Lock).

    Attributs:
        cap (cv2.VideoCapture): Objet de capture OpenCV
        tracker (SmartTracker): Moteur d'IA (détection + tracking)
        _lock (threading.Lock): Verrou pour l'accès concurrent aux frames
        _current_frame (np.ndarray): Dernière frame brute capturée
        _annotated_frame (np.ndarray): Dernière frame annotée par l'IA
        _running (bool): Flag de contrôle du thread de capture
        _thread (threading.Thread): Thread de capture vidéo
    """

    def __init__(self):
        """Initialise la caméra, le tracker et le thread de capture."""

        print("\n[SERVEUR] ══════════════════════════════════════")
        print("[SERVEUR]   Démarrage du CameraManager...")
        print("[SERVEUR] ══════════════════════════════════════\n")

        # --- Verrous pour thread-safety ---
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()

        # --- Frames partagées ---
        self._current_frame: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None

        # --- Flag de contrôle ---
        self._running = False

        # --- Initialisation du moteur d'IA ---
        print("[SERVEUR] Initialisation du SmartTracker...")
        self.tracker = SmartTracker()

        # --- Ouverture de la caméra ---
        print(f"[SERVEUR] Ouverture de la caméra (source: {config.CAMERA_SOURCE})...")
        self.cap = cv2.VideoCapture(config.CAMERA_SOURCE)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[ERREUR FATALE] Impossible d'ouvrir la caméra "
                f"(source: {config.CAMERA_SOURCE}). "
                f"Vérifiez la connexion ou l'index."
            )

        # Configuration de la résolution de capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)

        # Lecture de la résolution réelle obtenue
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[SERVEUR] Caméra ouverte : {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")

        # --- Variables pour le calcul du FPS de traitement ---
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()

        # --- Lancement du thread de capture ---
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraThread",
            daemon=True  # Se ferme automatiquement quand le programme principal s'arrête
        )
        self._thread.start()

        print("[SERVEUR] Thread de capture vidéo démarré.\n")

    # =========================================================================
    # BOUCLE DE CAPTURE (Thread dédié)
    # =========================================================================

    def _capture_loop(self) -> None:
        """
        Boucle principale exécutée dans un thread dédié.
        Capture les frames, les traite via le SmartTracker,
        et stocke le résultat pour le flux MJPEG.
        """
        print("[THREAD] Boucle de capture démarrée.")

        while self._running:
            try:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    # Tentative de reconnexion si la frame est invalide
                    print("[THREAD] Frame invalide, tentative de reconnexion...")
                    time.sleep(0.5)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(config.CAMERA_SOURCE)
                    continue

                # Sauvegarde de la frame brute
                with self._frame_lock:
                    self._current_frame = frame.copy()

                # --- Traitement par le moteur d'IA ---
                annotated, current_names = self.tracker.process_frame(frame)

                # --- Ajout du FPS sur la frame annotée ---
                self._update_fps()
                cv2.putText(
                    annotated,
                    f"FPS: {self._fps:.1f}",
                    (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Sauvegarde de la frame annotée (thread-safe)
                with self._lock:
                    self._annotated_frame = annotated

            except Exception as e:
                print(f"[THREAD] Erreur dans la boucle de capture : {e}")
                time.sleep(0.1)

        print("[THREAD] Boucle de capture arrêtée.")

    # =========================================================================
    # CALCUL DU FPS
    # =========================================================================

    def _update_fps(self) -> None:
        """Calcule le FPS de traitement (mis à jour chaque seconde)."""
        self._fps_counter += 1
        elapsed = time.time() - self._fps_timer

        if elapsed >= 1.0:
            self._fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = time.time()

    # =========================================================================
    # GÉNÉRATEUR DE FRAMES MJPEG
    # =========================================================================

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """
        Générateur Python qui produit un flux MJPEG (multipart/x-mixed-replace).
        Chaque frame est encodée en JPEG et envoyée au navigateur.

        Yields:
            Bytes au format MJPEG (boundary + image JPEG)
        """
        print("[MJPEG] Client connecté au flux vidéo.")

        # Paramètres d'encodage JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]

        # Frame noire affichée en attendant la première vraie frame
        placeholder = np.zeros(
            (config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH, 3),
            dtype=np.uint8
        )
        cv2.putText(
            placeholder,
            "Initialisation de la camera...",
            (50, config.DISPLAY_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 0),
            2,
            cv2.LINE_AA
        )

        while self._running:
            # Récupération thread-safe de la dernière frame annotée
            with self._lock:
                frame = self._annotated_frame

            if frame is None:
                frame = placeholder

            # Encodage en JPEG
            success, buffer = cv2.imencode(".jpg", frame, encode_params)

            if not success:
                continue

            # Format MJPEG : boundary + Content-Type + données
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

            # Limitation du débit pour ne pas surcharger le réseau
            time.sleep(1.0 / config.MJPEG_FPS_LIMIT)

        print("[MJPEG] Flux vidéo terminé.")

    # =========================================================================
    # API STATISTIQUES
    # =========================================================================

    def get_stats(self) -> dict:
        """
        Retourne les statistiques actuelles pour l'API JSON.

        Returns:
            Dictionnaire avec :
              - total_unique_today (int): Personnes uniques vues
              - currently_present (list): Noms des personnes à l'écran
              - fps (float): FPS du traitement IA
              - database_size (int): Nombre de personnes dans la base
              - all_seen_names (list): Historique des noms vus
        """
        return {
            "total_unique_today": self.tracker.get_total_unique_today(),
            "currently_present": self.tracker.get_currently_present(),
            "fps": round(self._fps, 1),
            "database_size": len(self.tracker.face_db),
            "all_seen_names": self.tracker.get_all_seen_names(),
            "uptime_seconds": int(time.time() - self._fps_timer)
        }

    # =========================================================================
    # NETTOYAGE DES RESSOURCES
    # =========================================================================

    def release(self) -> None:
        """
        Libère proprement toutes les ressources :
          - Arrête le thread de capture
          - Libère la caméra OpenCV
          - Libère le tracker (Mediapipe)
        """
        print("\n[SERVEUR] ══════════════════════════════════════")
        print("[SERVEUR]   Arrêt du système...")
        print("[SERVEUR] ══════════════════════════════════════")

        # Arrêt du thread
        self._running = False

        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            print("[SERVEUR] Thread de capture arrêté.")

        # Libération de la caméra
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            print("[SERVEUR] Caméra libérée.")

        # Libération du tracker
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.release()
            print("[SERVEUR] Tracker libéré.")

        print("[SERVEUR] Toutes les ressources sont libérées.")
        print("[SERVEUR] Au revoir.\n")


# =============================================================================
# INSTANCE GLOBALE DU CAMERA MANAGER
# =============================================================================

# Instance unique (initialisée au démarrage du serveur)
camera_manager: Optional[CameraManager] = None


def get_camera_manager() -> CameraManager:
    """
    Singleton paresseux du CameraManager.
    Créé une seule fois au premier appel.

    Returns:
        Instance unique de CameraManager
    """
    global camera_manager
    if camera_manager is None:
        camera_manager = CameraManager()
    return camera_manager


# =============================================================================
# NETTOYAGE À L'ARRÊT DU SERVEUR
# =============================================================================

def cleanup():
    """Fonction de nettoyage appelée automatiquement à l'arrêt."""
    global camera_manager
    if camera_manager is not None:
        camera_manager.release()
        camera_manager = None


# Enregistrement du nettoyage via atexit (arrêt normal)
atexit.register(cleanup)


def signal_handler(signum, frame):
    """Gestion des signaux SIGINT (Ctrl+C) et SIGTERM."""
    print(f"\n[SIGNAL] Signal {signum} reçu. Arrêt en cours...")
    cleanup()
    sys.exit(0)


# Enregistrement des gestionnaires de signaux
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# ROUTES FLASK
# =============================================================================

@app.route("/")
def index():
    """
    Route principale — Sert le dashboard HTML.

    Returns:
        Page HTML rendue (templates/index.html)
    """
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    Route du flux vidéo MJPEG.

    Le navigateur consomme ce flux via une balise <img src="/video_feed">.
    Le format 'multipart/x-mixed-replace' permet un streaming continu
    où chaque frame JPEG remplace la précédente.

    Returns:
        Response Flask avec le générateur MJPEG
    """
    manager = get_camera_manager()

    return Response(
        manager.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stats")
def api_stats():
    """
    API REST retournant les statistiques en temps réel au format JSON.

    Appelée toutes les 2 secondes par le frontend via fetch().

    Exemple de réponse :
    {
        "total_unique_today": 3,
        "currently_present": ["Jean_Dupont", "Marie_Martin"],
        "fps": 24.5,
        "database_size": 5,
        "all_seen_names": ["Jean_Dupont", "Marie_Martin", "Paul_Durand"],
        "uptime_seconds": 3600
    }

    Returns:
        JSON avec les statistiques
    """
    manager = get_camera_manager()
    stats = manager.get_stats()

    response = jsonify(stats)
    # Pas de cache pour avoir des données fraîches
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


@app.route("/api/health")
def api_health():
    """
    Endpoint de santé pour vérifier que le serveur fonctionne.

    Returns:
        JSON avec le statut du serveur
    """
    global camera_manager
    is_camera_ok = (
        camera_manager is not None
        and camera_manager._running
        and camera_manager.cap.isOpened()
    )

    return jsonify({
        "status": "ok" if is_camera_ok else "degraded",
        "camera_active": is_camera_ok,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/database")
def api_database():
    """
    Endpoint retournant la liste des personnes enregistrées dans la base.

    Returns:
        JSON avec la liste des noms
    """
    manager = get_camera_manager()
    names = manager.tracker.face_db.get_names()

    return jsonify({
        "database_size": len(names),
        "registered_persons": names
    })


# =============================================================================
# GESTION DES ERREURS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """Gestion des erreurs 404."""
    return jsonify({
        "error": "Route non trouvée",
        "message": "Les routes disponibles sont : /, /video_feed, /api/stats, /api/health, /api/database"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs 500."""
    return jsonify({
        "error": "Erreur interne du serveur",
        "message": str(error)
    }), 500


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    """
    Lancement du serveur Flask.

    Usage :
        python app.py
        python app.py --port 8080
        python app.py --host 0.0.0.0

    Le serveur démarre sur http://localhost:5000 par défaut.
    """
    import argparse

    # --- Parsing des arguments CLI ---
    parser = argparse.ArgumentParser(
        description="Serveur de reconnaissance faciale avec dashboard web"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Adresse d'écoute (défaut: 127.0.0.1, utiliser 0.0.0.0 pour réseau local)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port d'écoute (défaut: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Mode debug Flask (NE PAS utiliser en production)"
    )
    args = parser.parse_args()

    # --- Affichage de la bannière de démarrage ---
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║   🎯  SYSTÈME DE RECONNAISSANCE FACIALE                 ║")
    print("║   📊  Dashboard Web en temps réel                       ║")
    print("║                                                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║   🌐  Adresse : http://{args.host}:{args.port}               ║")
    print(f"║   📹  Caméra  : {str(config.CAMERA_SOURCE):<40} ║")
    print(f"║   🧠  Modèle  : {config.INSIGHTFACE_MODEL:<40} ║")
    print(f"║   🎯  Seuil   : {config.RECOGNITION_THRESHOLD:<40} ║")
    print("║                                                          ║")
    print("║   Ctrl+C pour arrêter le serveur                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\n")

    # --- Pré-initialisation du CameraManager ---
    # On le crée AVANT le démarrage de Flask pour afficher les erreurs tôt
    try:
        print("[SERVEUR] Pré-initialisation du système d'IA...\n")
        manager = get_camera_manager()
        print("\n[SERVEUR] Système d'IA initialisé avec succès.")
        print(f"[SERVEUR] Base de données : {len(manager.tracker.face_db)} personne(s)")
        print(f"[SERVEUR] Dashboard accessible sur : http://{args.host}:{args.port}\n")

    except Exception as e:
        print(f"\n[ERREUR FATALE] Échec de l'initialisation : {e}")
        print("[ERREUR FATALE] Vérifiez la caméra et les dépendances.")
        sys.exit(1)

    # --- Démarrage du serveur Flask ---
    # threaded=True permet de servir plusieurs clients simultanément
    # (flux vidéo + API stats + page HTML)
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False  # Important : évite la double initialisation du tracker
        )
    except Exception as e:
        print(f"[ERREUR] Le serveur a crashé : {e}")
    finally:
        cleanup()
