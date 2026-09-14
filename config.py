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
EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "embeddings.npz")


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

# Threads du serveur waitress. Chaque onglet ouvert sur /video en occupe un
# tant qu'il regarde le flux : avec les 4 threads par défaut de waitress,
# quatre onglets suffisaient à bloquer /status et l'enrôlement.
SERVER_THREADS = int(os.getenv("SERVER_THREADS", "16"))


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

# Similarité cosinus minimale pour valider un match (0.0–1.0).
# Plus HAUT = plus strict : _identify teste `score >= seuil`.
# Mesuré avec tools/recognition_threshold_study.py sur CHIRLA (enrôlement en
# juin-juillet, reconnaissance en décembre) : sur les visages d'au moins 40 px,
# aucun mauvais nom entre 0,45 et 0,75, et 96 % de bons noms à 0,45 au-delà de
# 56 px. Les erreurs viennent des petits visages, d'où le seuil suivant.
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.45"))

# Taille minimale (côté le plus court, px) d'un visage pour décider d'un nom.
# Même étude, seuil 0,45 : 1,7 % de mauvais noms sur tous les visages (35 px
# en médiane), 0,0 % dès 40 px. Un visage plus petit ne change pas le nom de
# la piste, qui reste celui qu'elle avait.
RECOGNITION_MIN_FACE_PX = int(os.getenv("RECOGNITION_MIN_FACE_PX", "40"))


# =============================================================================
# TRACKER & PERFORMANCE
# =============================================================================

# Traiter 1 frame sur N pour la pipeline IA (reco + pose).
# Les frames intermédiaires réutilisent le cache.
FRAME_SKIP = 2

# DeepSORT — tracker de corps pour le pipeline Body-First.
DEEPSORT_MAX_AGE = 70           # Frames avant suppression d'un track perdu.
# Confirmations minimales avant qu'un track soit actif. 5 plutôt que 3 (défaut
# du papier) : mesuré sur trois jeux de données en validation (MOT17,
# DanceTrack, CHIRLA ; cf. README, section Réglage du tracking), les
# changements d'identité baissent de 10 à 17 % pour un IDF1 inchangé à ±1,6
# point. Coût : une nouvelle personne apparaît 2 images plus tard.
DEEPSORT_N_INIT = 5
DEEPSORT_EMBEDDER = "mobilenet" # Embedder d'apparence pour la ReID (vêtements).
DEEPSORT_EMBEDDER_GPU = True    # Activer si GPU disponible (RTX 4060 ✅).

# Moteur TensorRT FP16 de l'embedder (cf. core/appearance_embedder.py), vide =
# MobileNetV2 en PyTorch. Mesuré sur MOT17 (7 séquences FRCNN, RTX 4060) :
# étape tracker de 19,1 à 13,7 ms en médiane, MOTA 21,7 → 22,1 %, IDF1 48,1 %
# inchangé. Propre au GPU qui l'a construit : non versionné, cf. README.
DEEPSORT_EMBEDDER_ENGINE = os.getenv("DEEPSORT_EMBEDDER_ENGINE", "")

# Nombre de vecteurs d'apparence conservés par piste (les plus récents).
# None = illimité, le défaut de deep_sort_realtime : la banque grossit d'une
# feature par frame et par piste, donc le coût de l'association croît avec la
# durée de la session. Mesuré à 6 personnes/frame sur 200 frames : 5,01 ms
# d'association avec None contre 2,24 ms avec 100, et l'écart s'aggrave.
# 100 est la valeur du papier Wojke et al. 2017 et le défaut du crate.
DEEPSORT_NN_BUDGET = 100

# Seuils d'association : les défauts de deep_sort_realtime, repris du papier.
# Balayés avec tools/sweep_deepsort.py sur MOT17, DanceTrack et CHIRLA, en
# validation : le meilleur seuil cosinus dépend de la scène (0,15 sur MOT17,
# 0,25-0,3 sur DanceTrack et CHIRLA) et 0,15 perd 2 à 10 points d'IDF1 hors de
# MOT17 ; 0,2 reste le compromis. max_age=150 gagne sur DanceTrack et CHIRLA
# mais perd 1 point d'IDF1 sur MOT17 : 70 est conservé.
#   distance cosinus : au-delà, l'apparence est jugée trop différente
#   distance IoU     : au-delà, le recouvrement est jugé insuffisant
DEEPSORT_MAX_COSINE_DISTANCE = 0.2
DEEPSORT_MAX_IOU_DISTANCE = 0.7

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
#
# "yolov8s-pose.engine" : le même modèle compilé en TensorRT FP16. Mesuré par
# tools/bench_yolo.py sur MOT17-04 et MOT17-09 (1920x1080, RTX 4060) : étape
# YOLO 1,8 à 2,2x plus rapide en médiane, p95 de ~33 ms à ~7-8 ms, keypoints
# à 0,16-0,44 px près au p95. Un moteur ne vaut que pour le GPU et la version
# de TensorRT qui l'ont construit : il n'est pas versionné, cf. README.
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s-pose.pt")

# Seuil de confiance minimum pour qu'un corps YOLO soit passé au tracker.
YOLO_CONF_THRESHOLD = 0.5

# Confiance minimale d'un keypoint facial (COCO 0-4) pour le considérer visible.
POSE_NOSE_CONF_THRESHOLD = 0.5

# --- Source de l'estimation d'orientation (face/profil/dos) -------------------
# "mediapipe" : modèle Mediapipe Pose sur un crop, une inférence CPU par
#               personne et par frame.
# "yolo"      : relecture des keypoints COCO déjà produits par YOLOv8-Pose sur
#               GPU — aucune inférence supplémentaire.
# Défaut "yolo" : gratuit, et 100 % d'accord avec Mediapipe au-delà de
# POSE_MIN_SHOULDER_DIST_PX ; en dessous il s'abstient au lieu de deviner.
POSE_SOURCE = os.getenv("POSE_SOURCE", "yolo").strip().lower()

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

# Le crop doit grandir avec le VISAGE : SCRFD, le détecteur d'InsightFace, ne
# trouve pas un visage qui remplit son entrée. L'écart maximal entre deux
# keypoints faciaux visibles (COCO 0-4) mesure directement cette taille, et
# c'est la seule grandeur du genre disponible avant d'appeler le détecteur.
#
# Calé sur 70 mesures live (visages de 51 à 246 px, 7 échelles) : le rapport
# demi-taille minimale / écart vaut 1.16 en médiane, 1.40 au p95 et 1.48 au
# pire. Le facteur retenu couvre ce pire cas avec de la marge — dépasser est
# sans risque pour la détection (vérifié jusqu'à une demi-taille de 500), le
# seul coût est d'attraper le visage d'un voisin dans une scène dense.
POSE_CROP_KP_SPAN_RATIO = 1.6

# Repli quand moins de deux keypoints faciaux sont visibles (personne de profil
# marqué ou de dos) : la hauteur du corps, qui est un proxy nettement plus
# dispersé sur les mêmes mesures (rapport de 0.28 à 0.51). De près, la boîte
# YOLO n'est plus qu'une tête-épaules et sous-estime la taille du visage —
# c'est exactement ce qui faisait échouer la reco en plan rapproché.
POSE_CROP_BODY_RATIO = 0.55

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

for _dir in (DATA_DIR, KNOWN_FACES_DIR):
    os.makedirs(_dir, exist_ok=True)
