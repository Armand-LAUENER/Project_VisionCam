"""
config.py — Configuration centralisée du projet de reconnaissance faciale.
Toutes les variables globales sont définies ici pour faciliter la maintenance.
"""

import os

# =============================================================================
# CHEMINS DU PROJET
# =============================================================================

# Répertoire racine du projet (là où se trouve ce fichier)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossier principal des données
DATA_DIR = os.path.join(BASE_DIR, "data")

# Dossier contenant les sous-dossiers de visages capturés (1 sous-dossier = 1 personne)
# Structure attendue : data/captured_faces/Prenom_Nom/01.jpg, 02.jpg, ...
CAPTURED_FACES_DIR = os.path.join(DATA_DIR, "captured_faces")

# Chemin du fichier cache des embeddings (dictionnaire sérialisé avec pickle)
EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "embeddings.pkl")


# =============================================================================
# SOURCE VIDÉO
# =============================================================================

# Index entier (0, 1, ...) pour une webcam locale, ou URL string pour un flux RTSP/HTTP
CAMERA_SOURCE = "http://192.168.27.55:5000/video"
FLASK_PORT = 8080



# =============================================================================
# PARAMÈTRES INSIGHTFACE
# =============================================================================

# Modèle de reconnaissance faciale InsightFace
INSIGHTFACE_MODEL = "buffalo_l"

# Fournisseur d'exécution ONNX — GPU Nvidia en priorité, CPU en fallback
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Taille de détection (plus grand = plus précis mais plus lent)
INSIGHTFACE_DET_SIZE = (640, 640)


# =============================================================================
# SEUILS DE RECONNAISSANCE
# =============================================================================

# Seuil de distance cosinus pour valider une correspondance faciale.
# Plus la valeur est BASSE, plus le match doit être similaire.
# Valeurs typiques : 0.5 (strict) — 0.7 (permissif)
RECOGNITION_THRESHOLD = 0.65


# =============================================================================
# PARAMÈTRES DE SUIVI (DeepSort)
# =============================================================================

# Durée max (en secondes) avant qu'un track sans détection soit supprimé
DEEPSORT_MAX_AGE = 30

# Nombre de détections consécutives avant de confirmer un track
DEEPSORT_N_INIT = 3

# Embedder utilisé par DeepSort pour la ré-identification corporelle
DEEPSORT_EMBEDDER = "mobilenet"


# =============================================================================
# PARAMÈTRES DU DASHBOARD / SERVEUR FLASK
# =============================================================================

# Port du serveur web local
FLASK_PORT = 5000

# Intervalle (en secondes) au-delà duquel une personne est considérée absente
PRESENCE_TIMEOUT = 5.0

# Résolution d'affichage du flux vidéo envoyé au navigateur
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# =============================================================================
# CRÉATION AUTOMATIQUE DES DOSSIERS NÉCESSAIRES
# =============================================================================

for _dir in [DATA_DIR, CAPTURED_FACES_DIR]:
    os.makedirs(_dir, exist_ok=True)
# =============================================================================
# PARAMÈTRES DU SERVEUR FLASK
# =============================================================================

# Qualité d'encodage JPEG pour le flux MJPEG (0-100)
# Plus élevé = meilleure qualité mais plus de bande passante
MJPEG_QUALITY = 75

# Limite de FPS pour le flux MJPEG envoyé au navigateur
# Évite de surcharger le réseau local
MJPEG_FPS_LIMIT = 30

# =============================================================================
# PARAMÈTRES D'AFFICHAGE
# =============================================================================

# Résolution demandée à la caméra
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# =============================================================================
# PARAMÈTRES DE PRÉSENCE
# =============================================================================

# Durée (secondes) avant qu'une personne soit considérée comme "partie"
# Si non vue depuis ce délai, elle disparaît de "currently_present"
PRESENCE_TIMEOUT = 5.0

# =============================================================================
# PARAMÈTRES DEEPSORT
# =============================================================================

# Nombre de frames sans détection avant suppression d'un track
DEEPSORT_MAX_AGE = 70

# Nombre de détections consécutives pour confirmer un track
DEEPSORT_N_INIT = 3

# Modèle d'embedding pour DeepSort (mobilenet léger)
DEEPSORT_EMBEDDER = "mobilenet"
