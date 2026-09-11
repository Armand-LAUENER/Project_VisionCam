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

INSIGHTFACE_MODEL = "antelopev2"
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Résolution d'entrée du détecteur de visages.
# (640, 640) : bon compromis perf/précision pour flux 1080p, personnes à < 5m.
# (320, 320) : +rapide, moins précis sur visages lointains.
# (1280, 1280): détection longue distance, mais ~4x plus lent — éviter en temps réel.
INSIGHTFACE_DET_SIZE = (640, 640)

# Seuil cosinus minimum pour valider un match (plus bas = plus strict).
# Typique : 0.45 (strict) — 0.65 (permissif).
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.45"))


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

# Implémentation de l'association de tracks (cf. core/tracker_backends.py).
# "python" : deep_sort_realtime 1.3.2, embedder MobileNetV2 intégré.
# "rust"   : crate deepsort-rs via PyO3. Même algorithme et même embedder —
#            seule l'association (Kalman + cascade + IoU) passe en Rust.
# Bascule à chaud : TRACKER_BACKEND=rust dans .env, sans toucher au code.
TRACKER_BACKEND = os.getenv("TRACKER_BACKEND", "python").strip().lower()

# =============================================================================
# PIPELINE BODY-FIRST (YOLOv8 + DeepSORT + InsightFace)
# =============================================================================

# Modèle YOLO-Pose : détecte les corps ET les keypoints du squelette COCO (17 pts).
# "yolov8s-pose.pt" = Small-Pose — auto-téléchargé au premier lancement.
YOLO_MODEL = "yolov8s-pose.pt"

# Seuil de confiance minimum pour qu'un corps YOLO soit passé au tracker.
YOLO_CONF_THRESHOLD = 0.5

# Confiance minimale d'un keypoint facial (COCO 0-4) pour le considérer visible.
POSE_NOSE_CONF_THRESHOLD = 0.5

# --- Source de l'estimation d'orientation (face/profil/dos) -------------------
# "mediapipe" : modèle Mediapipe Pose sur un crop, une inférence CPU par
#               personne et par frame.
# "yolo"      : relecture des keypoints COCO déjà produits par YOLOv8-Pose sur
#               GPU — aucune inférence supplémentaire.
POSE_SOURCE = os.getenv("POSE_SOURCE", "mediapipe").strip().lower()

# Seuils de la classification depuis les keypoints YOLO (POSE_SOURCE="yolo").
# Repris de la logique Mediapipe ; la confiance de keypoint YOLO ne suit pas la
# même distribution que la `visibility` Mediapipe, ils sont donc réglables.
POSE_KP_VISIBLE_CONF = 0.3          # en dessous : keypoint considéré non visible
POSE_KP_NOSE_CONF = 0.4             # nez moins sûr que ça → personne de dos
POSE_EAR_DIFF_THRESHOLD = 0.4       # écart de confiance entre oreilles → profil
POSE_OFFSET_RATIO_THRESHOLD = 0.15  # décalage nez/centre des épaules → face
# Écart d'épaules minimal (px) pour que l'orientation soit jugée fiable.
# Mesuré par tools/pose_threshold_study.py sur 7 séquences MOT17 + un plan
# webcam rapproché (657 observations) : l'accord avec Mediapipe reste erratique
# entre 51 % et 80 % en dessous de 100 px, et atteint 100 % au-delà. Sous ce
# seuil, l'estimateur renvoie None — mieux vaut ne rien dire que se tromper.
POSE_MIN_SHOULDER_DIST_PX = 100

# Nombre minimal de keypoints faciaux visibles pour utiliser le centroïde.
# En dessous → fallback sur l'estimation tête depuis la bbox corps.
POSE_FACE_KP_MIN_VISIBLE = 1

# Demi-taille du crop carré centré sur le visage, en pixels.
# Plancher utilisé pour les personnes lointaines, dont le visage est petit :
# l'agrandir diluerait les petits visages au redimensionnement d'InsightFace.
POSE_CROP_HALF_SIZE = 125

# Le crop grandit avec la personne : half = max(POSE_CROP_HALF_SIZE,
# body_height × POSE_CROP_BODY_RATIO). SCRFD, le détecteur d'InsightFace, a
# besoin de marge autour du visage — un visage qui remplit le crop n'est pas
# détecté du tout. Mesuré sur un plan rapproché (corps 499 px, visage
# 184×250 px) : à 0.25 le crop tombait sur le plancher de 125, soit 250×250,
# et InsightFace ne trouvait aucun visage ; à 0.40 il fait 398×398 et le
# détecte à 0.754. Le score de similarité, lui, ne dépend pas de la taille du
# crop (0.57-0.60 dans tous les cas où le visage est trouvé).
POSE_CROP_BODY_RATIO = 0.40

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