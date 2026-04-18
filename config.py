"""
config.py — Configuration centralisée du projet.

Source de vérité unique pour TOUS les paramètres. Aucune constante
ne doit être hardcodée ailleurs (ni dans app.py, ni dans core/).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Charge .env sans écraser les variables d'environnement système

# =============================================================================
# CHEMINS DU PROJET
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
CAPTURED_FACES_DIR = os.path.join(DATA_DIR, "captured_faces")
EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "embeddings.pkl")


# =============================================================================
# SOURCE VIDÉO
# =============================================================================

# True = webcam locale, False = caméra IP
USE_LOCAL_CAM = os.getenv("USE_LOCAL_CAM", "true").lower() == "true"

LOCAL_SOURCE = 0
REMOTE_SOURCE = os.getenv("REMOTE_SOURCE", "http://192.168.27.65:5000/video")
CAMERA_SOURCE = LOCAL_SOURCE if USE_LOCAL_CAM else REMOTE_SOURCE


# =============================================================================
# SERVEUR FLASK
# =============================================================================

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
DEBUG_MODE = False


# =============================================================================
# INSIGHTFACE (reconnaissance faciale)
# =============================================================================

INSIGHTFACE_MODEL = "buffalo_l"
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
INSIGHTFACE_DET_SIZE = (1280,1280)

# Seuil cosinus minimum pour valider un match (plus bas = plus strict).
# Typique : 0.45 (strict) — 0.65 (permissif).
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.65"))


# =============================================================================
# TRACKER & PERFORMANCE
# =============================================================================

# Traiter 1 frame sur N pour la pipeline IA (reco + pose).
# Les frames intermédiaires réutilisent le cache.
FRAME_SKIP = 2

# DeepSORT — tracker de corps pour le pipeline Body-First.
DEEPSORT_MAX_AGE = 70           # Frames avant suppression d'un track perdu.
DEEPSORT_N_INIT = 3             # Confirmations minimales avant qu'un track soit actif.
DEEPSORT_EMBEDDER = "mobilenet" # Embedder d'apparence pour la ReID (vêtements).
DEEPSORT_EMBEDDER_GPU = True    # Activer si GPU disponible (RTX 4060 ✅).

# =============================================================================
# PIPELINE BODY-FIRST (YOLOv8 + DeepSORT + InsightFace)
# =============================================================================

# Modèle YOLO à utiliser. "yolov8n.pt" = Nano (~3ms/frame GPU), auto-téléchargé.
YOLO_MODEL = "yolov8m.pt"

# Seuil de confiance minimum pour qu'un corps YOLO soit passé au tracker.
YOLO_CONF_THRESHOLD = 0.5

# Lancer InsightFace toutes les N frames seulement.
# 5 = bon compromis (15ms × 1/5 = 3ms amortis/frame) ; baisser si réseau lent.
FACE_RECOGNITION_SKIP = 5

# Nombre d'échecs consécutifs de cap.read() avant de tenter une reconnexion.
# 10 frames ≈ 0.5s à 20 FPS — distingue un drop réseau bref d'une vraie déconnexion.
CAM_MAX_FAILURES = 10

# Timeouts OpenCV pour la caméra distante (ms).
# Sans timeout explicite, VideoCapture sur HTTP bloque ~30s sans aucun log.
CAM_OPEN_TIMEOUT_MS  = 5_000   # Abandon de la connexion si pas de réponse en 5s.
CAM_READ_TIMEOUT_MS  = 3_000   # Abandon d'une lecture de frame si pas de données en 3s.

# Nombre de frames au-delà duquel on considère qu'on "ne voit plus le visage".
# Passé ce délai, le nom est affiché avec l'indicateur "body-tracking" (orange).
# Exemple : 30 frames à 30 FPS = 1 seconde avant le basculement visuel.
FACE_FRESHNESS_FRAMES = 30


# =============================================================================
# STREAMING & AFFICHAGE
# =============================================================================

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# Qualité JPEG du flux MJPEG (0-100). Plus haut = plus net, plus de bande passante.
MJPEG_QUALITY = 75

# FPS max du flux envoyé au navigateur.
MJPEG_FPS_LIMIT = 30


# =============================================================================
# PRÉSENCE
# =============================================================================

# Délai (s) au-delà duquel une personne non détectée est retirée de "currently_present".
PRESENCE_TIMEOUT = 5.0


# =============================================================================
# CRÉATION AUTOMATIQUE DES DOSSIERS
# =============================================================================

for _dir in (DATA_DIR, KNOWN_FACES_DIR, CAPTURED_FACES_DIR):
    os.makedirs(_dir, exist_ok=True)