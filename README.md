# VisionCam

> Pipeline de **reconnaissance faciale en temps réel** sur flux vidéo,
> résistant aux lunettes, aux occlusions partielles et aux variations de distance.
> Exposé via une interface web MJPEG + API REST.

---

## Fonctionnalités

- **Détection & tracking multi-personnes** — YOLOv8-Pose + DeepSORT (IDs persistants)
- **Reconnaissance faciale robuste** — InsightFace antelopev2 (CUDA)
  - Centroïde des keypoints faciaux visibles (nez, yeux, oreilles)
  - Résistant aux lunettes et aux occlusions partielles
  - Fallback automatique sur la bbox corps si aucun keypoint visible
- **Orientation** — face / profil / dos, depuis les keypoints YOLO déjà calculés (défaut) ou MediaPipe
- **Tracking par corps** — identité maintenue même quand le visage disparaît
- **Enrôlement à chaud** — ajout de personnes sans redémarrer l'application
- **Interface web** — pages Live (boîtes cliquables pour enrôler), Personnes, Historique et Diagnostic ; temps réel par Server-Sent Events, alertes du navigateur, utilisable sur téléphone et hors ligne
- **Reconnexion automatique** — backoff exponentiel 1 s → 30 s pour les caméras IP

---

## Architecture

```
┌───────────────────┐     ┌────────────────────────┐
│  Webcam locale    │  OU │  Caméra IP (MJPEG LAN) │
└────────┬──────────┘     └──────────┬─────────────┘
         └──────────┬────────────────┘
                    ▼
       ┌────────────────────────────────────────┐
       │ Thread A — camera_loop()               │
       │  cap.read() → _frame_queue             │
       └────────────┬───────────────────────────┘
                    ▼
       ┌────────────────────────────────────────────────────┐
       │ Thread B — processing_loop()                       │
       │                                                    │
       │ FaceBodyTracker.update(frame, count)               │
       │   1. YOLOv8-Pose  → corps + keypoints squelette    │
       │   2. DeepSORT      → track_id persistant / ReID    │
       │   3. InsightFace   → crop centré sur tête (1/5f)   │
       │      ├─ centroïde keypoints visibles (lunettes ✓)  │
       │      └─ fallback body-top si dos tourné            │
       │   4. Vote buffer   → anti-flip identité            │
       │   5. Orientation   → keypoints YOLO ou MediaPipe   │
       └────────────┬───────────────────────────────────────┘
                    ▼
       ┌────────────────────┐     ┌──────────────────────────┐
       │  AppState (lock)   │────►│  Flask (thread principal)│
       │  current_frame     │     │  GET  /                  │
       │  currently_present │     │  GET  /video  (MJPEG)    │
       └────────────────────┘     │  GET  /status (JSON)     │
                                  │  POST /api/capture       │
                                  │  POST /api/people/…      │
                                  │  POST /rebuild           │
                                  └──────────────────────────┘
```

### Feedback visuel

Boîtes dessinées par la page Live par-dessus la vidéo :

| Couleur | Signification |
|:--------|:--------------|
| **Vert** | Personne reconnue (nom · orientation) |
| **Bleu** | Personne inconnue |
| **Blanc** | Personne sélectionnée pour l'enrôlement |

Avec `MJPEG_ANNOTATE=true`, le flux `/video` porte aussi ses propres boîtes : vert (visage reconnu récemment), orange `[BODY]` (identité connue, tracking corps seul), bleu (inconnu).

---

## Stack technique

| Domaine | Outil | Version |
|:--------|:------|:--------|
| Langage | Python | 3.11+ |
| Web | Flask + waitress (serveur WSGI) | 3.1 / 3.0 |
| Reconnaissance faciale | InsightFace (antelopev2) + ONNX Runtime GPU | 0.7.3 |
| Détection corps | YOLOv8-Pose (ultralytics) | 8.4 |
| Tracking | DeepSORT + ReID MobileNet GPU | 1.3.2 |
| Orientation (option) | MediaPipe | 0.10 |
| Traitement image | OpenCV | 4.13 |
| GPU | PyTorch CUDA 12.1 | 2.5.1 |

---

## Structure du projet

```
VisionCam/
├── app.py                  # Entrée : Flask + 2 threads (caméra + pipeline AI)
├── config.py               # Source unique de toutes les constantes
├── pyproject.toml          # Dépendances (uv) + réglages ruff / mypy
├── uv.lock                 # Versions figées de toute la chaîne
├── .env.example            # Template configuration locale
│
├── core/
│   ├── face_recognition.py   # FaceRecognizer — InsightFace + enrôlement
│   ├── face_body_tracker.py  # FaceBodyTracker — orchestrateur YOLO+DeepSORT
│   ├── tracker_backends.py   # Association : deep_sort_realtime ou deepsort-rs
│   ├── appearance_embedder.py # Embeddings d'apparence MobileNetV2 (PyTorch / TensorRT)
│   ├── pose_from_keypoints.py # KeypointPoseEstimator — orientation depuis YOLO
│   └── pose_estimation.py    # PoseEstimator — MediaPipe
│
├── scripts/visioncam.sh    # Lance / arrête l'application et le pont webcam
│
├── tools/
│   ├── bench_tracker.py      # Compare les deux backends (parité + vitesse)
│   ├── bench_pose.py         # Compare les deux sources d'orientation
│   ├── bench_yolo.py         # Compare des variantes du modèle YOLO (.pt, TensorRT)
│   ├── bench_embedder.py     # Compare les embedders d'apparence (vitesse, fidélité)
│   ├── bench_face.py         # Compare des variantes de la reconnaissance faciale (vitesse, noms)
│   ├── export_embedder_engine.py # Construit le moteur TensorRT de l'embedder
│   ├── eval_mot.py           # MOTA / IDF1 sur MOT17, balayage des seuils
│   ├── sweep_deepsort.py     # Réglage des seuils DeepSORT avec validation
│   ├── record_sequence.py    # Enregistre une séquence webcam au format MOT17
│   ├── annotate_sequence.py  # Vérité terrain : pré-remplie, corrigée à la main
│   ├── convert_chirla.py     # Vidéos CHIRLA → séquences MOT17
│   ├── pose_threshold_study.py # Choix de POSE_MIN_SHOULDER_DIST_PX
│   ├── set_password.py       # Mot de passe d'accès → ADMIN_PASSWORD_HASH dans .env
│   └── webcam_bridge.py      # Webcam Windows → flux MJPEG pour WSL2
│
├── docs/
│   ├── improvements-backlog.md  # Optimisations et améliorations restantes
│   └── roadmap-rust-tracker.md  # Roadmap (terminée) du backend Rust
│
├── templates/               # Pages : base, live, people, history, diagnostics, login
├── static/
│   ├── css/app.css          # Thème unique, sans CDN (fonctionne hors ligne)
│   └── js/                  # Un module par page + common.js (API, temps réel, alertes)
│
├── tests/                   # Aucun test ne demande de GPU ni de caméra
│   ├── test_face_body_association.py      # Association visage ↔ corps
│   ├── test_tracker_backends.py           # Bascule de backend + adaptateur Rust
│   ├── test_appearance_embedder.py        # Embedder identique à deep_sort_realtime
│   ├── test_face_recognition_cache.py     # Robustesse du cache d'embeddings
│   ├── test_identify.py                   # Recherche du meilleur match
│   ├── test_pose_from_keypoints.py        # Orientation depuis les keypoints
│   ├── test_pose_keypoint_propagation.py  # Keypoints YOLO → TrackedPerson
│   ├── test_eval_mot.py                   # Outil MOTA / IDF1
│   ├── test_web_routes.py                 # Page, routes Flask, flux MJPEG
│   ├── test_annotate_sequence.py          # Logique d'annotation, formats MOT
│   ├── ui/test_browser.py                 # Pages dans Chromium headless (Playwright)
│   └── regression/                        # Un fichier par bug corrigé
│
├── known_faces/             # Non inclus (RGPD) — voir section Enrôlement
└── data/                    # Non inclus — généré au démarrage
    ├── embeddings.npz       # Cache des embeddings (reconstructible)
    └── presence.db          # Historique des présences (SQLite)
```

---

## Installation

### Prérequis

- [uv](https://docs.astral.sh/uv/) — il installe aussi Python 3.11 si besoin
- [Rust](https://rustup.rs) (`cargo`) — pour construire `deepsort-rs`, le backend de tracking optionnel
- GPU NVIDIA avec CUDA 12.1 **fortement recommandé** (CPU possible mais ~5× plus lent)
- Webcam OU caméra IP accessible sur le LAN

### Étapes

```bash
# 1. Cloner
git clone https://github.com/Armand-LAUENER/Project_VisionCam.git
cd Project_VisionCam

# 2. Environnement + dépendances, versions de uv.lock (.venv/ dans le projet)
uv sync                                   # GPU : torch CUDA 12.1
# uv sync --no-group cu121 --group cpu    # sans GPU : torch CPU

# 3. Configuration
cp -n .env.example .env                   # -n : ne remplace jamais un .env existant
# Éditer .env si nécessaire (source vidéo, seuils, port)
uv run python tools/set_password.py       # recommandé : mot de passe d'accès

# 4. Préparer le dataset (voir section Enrôlement)
mkdir -p known_faces/MonNom
# Y copier 3-5 photos du visage

# 5. Lancer
uv run app.py
```

L'interface est disponible sur `http://localhost:5000`.

### Lancement en arrière-plan (WSL2 + webcam Windows)

```bash
scripts/visioncam.sh start    # pont webcam Windows (si USE_LOCAL_CAM=false), puis l'application
scripts/visioncam.sh status
scripts/visioncam.sh logs     # suit logs/visioncam.log
scripts/visioncam.sh stop     # application (sessions de présence fermées), puis pont webcam
```

Le pont est lancé côté Windows par `powershell.exe` (Python 3.11 via le lanceur `py`, fenêtre réduite) et n'est pas relancé s'il répond déjà. Rien ne démarre avec Windows : la caméra ne filme que sur `start`.

---

## Configuration

Tous les paramètres sont dans `config.py` (surchargeable via `.env`) :

| Constante | Défaut | Rôle |
|:----------|:-------|:-----|
| `USE_LOCAL_CAM` | `True` | Webcam locale si `True`, caméra IP sinon |
| `REMOTE_SOURCE` | URL MJPEG | URL du flux caméra IP |
| `FLASK_PORT` | `5000` | Port du serveur HTTP |
| `PRESENCE_LOG_GAP_S` | `60` | Absence (s) après laquelle une session de présence se ferme |
| `PRESENCE_RETENTION_DAYS` | `30` | Conservation de l'historique (jours) ; `0` = sans limite |
| `ADMIN_PASSWORD_HASH` / `ADMIN_PASSWORD` | vide | Mot de passe exigé pour tout l'accès ; vide = accès ouvert au réseau local |
| `SESSION_DAYS` | `7` | Durée d'une session de connexion |
| `SERVER_THREADS` | `32` | Threads waitress ; chaque onglet ouvert en garde deux (flux vidéo + événements) |
| `UNKNOWN_ALERT_S` | `3` | Délai avant de signaler une personne visible restée inconnue |
| `RECOGNITION_THRESHOLD` | `0.45` | Similarité cosinus min pour identifier (0–1, plus haut = plus strict) |
| `RECOGNITION_MIN_FACE_PX` | `40` | Taille min d'un visage pour décider d'un nom ; en dessous, la piste garde le sien |
| `FACE_RECOGNITION_SKIP` | `5` | InsightFace toutes les N frames |
| `MJPEG_ANNOTATE` | `false` | Boîtes dessinées dans le flux MJPEG lui-même (l'interface dessine les siennes) |
| `FACE_FRESHNESS_FRAMES` | `30` | Frames avant passage en mode orange [BODY] |
| `POSE_NOSE_CONF_THRESHOLD` | `0.5` | Confiance minimale d'un keypoint facial |
| `POSE_FACE_KP_MIN_VISIBLE` | `1` | Keypoints visibles min pour utiliser le centroïde |
| `YOLO_MODEL` | `yolov8s-pose.pt` | Modèle YOLO (auto-téléchargé), ou `yolov8s-pose.engine` (TensorRT) |
| `DEEPSORT_EMBEDDER_ENGINE` | vide | Moteur TensorRT de l'embedder ; vide = PyTorch |
| `INSIGHTFACE_TENSORRT` | `false` | Reconnaissance faciale en TensorRT FP16 (moteurs dans `data/trt_cache/`) |
| `DEEPSORT_MAX_AGE` | `70` | Frames avant suppression d'un track perdu |
| `FRAME_SKIP` | `2` | Cadence de l'estimation d'orientation |
| `POSE_SOURCE` | `yolo` | Orientation : `yolo` (keypoints, gratuit) ou `mediapipe` |
| `TRACKER_BACKEND` | `python` | Association de tracks : `python` ou `rust` |

---

## Accélération TensorRT (optionnel)

Deux réseaux peuvent tourner en moteur TensorRT FP16 : le détecteur
YOLOv8-Pose et l'embedder d'apparence MobileNetV2 du tracker. Un moteur dépend
du GPU et de la version de TensorRT qui l'ont construit : il n'est pas
versionné (`*.engine` est ignoré), chaque machine construit les siens, et les
reconstruit après une mise à jour de TensorRT ou du pilote. Sans moteur, le
pipeline tourne en PyTorch.

### YOLOv8-Pose (~4 min de construction)

```bash
# TensorRT et onnxslim font partie du groupe cu121, installé par uv sync
uv run yolo export model=yolov8s-pose.pt format=engine half=True device=0 imgsz=384,640
```

Puis `YOLO_MODEL=yolov8s-pose.engine` dans `.env`. Pour revenir en arrière,
retirer la ligne.

`imgsz=384,640` est la taille letterbox d'une image 16:9 ramenée à 640 px de
large, celle qu'utilise PyTorch sur le flux 1920x1080. Un moteur 640x640
ajoute un padding qui modifie les détections, sans rapport avec le FP16.

#### Résultats mesurés

`uv run -m tools.bench_yolo --images <vidéo ou motif> <modèle de référence> <modèles…>`
mesure l'étape YOLO telle que le pipeline la paie (`predict` + lecture des
résultats sur le CPU) et compare les sorties à la référence.

| Séquence | Modèle | Médiane | p95 | Détections non appariées | Écart keypoints (méd. / p95) |
|:---------|:-------|--------:|----:|-------------------------:|-----------------------------:|
| MOT17-04 (600 images) | `.pt` FP32 | 9,8 ms | 33,4 ms | — | — |
| | `.engine` FP16 640x640 | 6,5 ms | 16,9 ms | 558 | 0,10 / 1,50 px |
| | `.engine` FP16 384x640 | **5,5 ms** | **7,9 ms** | 24 | 0,04 / 0,16 px |
| MOT17-09 (400 images) | `.pt` FP32 | 11,5 ms | 33,6 ms | — | — |
| | `.engine` FP16 384x640 | **5,3 ms** | **6,8 ms** | 7 | 0,09 / 0,44 px |

RTX 4060, 1920x1080, 30 images de warm-up. `half=True` sur le `.pt`, sans
TensorRT, est plus lent que FP32 (0,91x) : le gain vient de TensorRT, pas du
FP16 seul. Ces temps portent sur l'étape YOLO uniquement.

### Embedder d'apparence MobileNetV2 (~1 min de construction)

Le tracker calcule un vecteur d'apparence par personne à chaque image
(`core/appearance_embedder.py`). Même réseau et mêmes poids que
`deep_sort_realtime`, mais le lot de crops est normalisé sur le GPU en un seul
transfert, au lieu de crop par crop sur le CPU.

```bash
uv run -m tools.export_embedder_engine
```

Puis `DEEPSORT_EMBEDDER_ENGINE=mobilenetv2-embedder.engine` dans `.env`. Sans
cette ligne, l'embedder tourne en PyTorch FP16 `channels_last`, déjà plus
rapide que l'embedder d'origine, pour des pistes identiques.

#### Résultats mesurés

Vitesse et fidélité des vecteurs, sur les crops YOLO de MOT17-04 (300 images,
7,8 crops/image) : `uv run -m tools.bench_embedder --images <…> --engine <moteur>`.
L'écart compare les distances cosinus entre crops d'une même image à celles de
la référence ; le gating d'apparence coupe à 0,2.

| Embedder | Médiane | p95 | Gain | Écart de distance (méd. / max) |
|:---------|--------:|----:|-----:|-------------------------------:|
| `deep_sort_realtime` (d'origine) | 13,3 ms | 33,6 ms | — | — |
| PyTorch, prétraitement GPU + `channels_last` | 9,2 ms | 20,5 ms | ×1,45 | 7e-5 / 4e-4 |
| TensorRT FP16 | **5,0 ms** | **7,0 ms** | **×2,67** | 3e-3 / 2e-2 |

TensorRT déplace un peu les distances : son effet a donc été mesuré sur le
tracking lui-même, sur les 7 séquences MOT17 FRCNN d'entraînement (5 316
images, détections publiques, backend `python`, protocole de
`tools/eval_mot.py`). Temps = étape tracker complète, embedder + association.

| Embedder | MOTA | IDF1 | Changements d'ID | Étape tracker (méd. / p95) |
|:---------|-----:|-----:|-----------------:|---------------------------:|
| `deep_sort_realtime` (d'origine) | 21,7 % | 48,1 % | 862 | 22,4 / 54,9 ms |
| PyTorch (défaut) | 21,7 % | 48,1 % | 866 | 19,1 / 39,7 ms |
| TensorRT FP16 | **22,1 %** | **48,1 %** | **838** | **13,7 / 34,8 ms** |

Pas de dégradation d'ensemble avec TensorRT. D'une séquence à l'autre, les
écarts vont dans les deux sens (IDF1 de −2,4 à +0,5 point) : quelques
associations à la limite du seuil basculent, sans tendance. Le calcul d'une
image sur deux, l'autre piste envisagée, n'a pas été retenu : il change
l'algorithme, là où TensorRT garde le même réseau.

### Reconnaissance faciale InsightFace (~2 min au premier lancement)

La reconnaissance était l'étape la plus lourde : SCRFD en 640×640 puis
glintr100 (ResNet-100) en CUDA FP32, par crop de tête. Deux changements :

- **SCRFD en 320×320**, par défaut. Les crops de tête font 250 px en médiane :
  en 640 ils étaient agrandis, et les visages devenus trop gros étaient ratés.
  La base d'embeddings est reconstruite au démarrage quand cette taille change.
- **TensorRT FP16** pour les deux modèles, via le `TensorrtExecutionProvider`
  d'onnxruntime : `INSIGHTFACE_TENSORRT=true` dans `.env`. Les moteurs se
  construisent au premier lancement et sont gardés dans `data/trt_cache/`.

#### Résultats mesurés

`uv run -m tools.bench_face --gallery <…> --probe <…>` : crops de tête de
l'application (YOLOv8-Pose + `head_crop_box`) sur CHIRLA, enrôlement sur deux
séquences, reconnaissance sur deux autres prises des mois plus tard ; 620
visages de test ≥ 40 px, seuil 0,45. Temps par crop, médiane / p95.

| Variante | Détection | Embedding | Bons noms | Mauvais noms | « Inconnu » |
|:---------|----------:|----------:|----------:|-------------:|------------:|
| CUDA, SCRFD 640 (avant) | 10,5 / 20,5 ms | 9,0 / 33,4 ms | 410 | 16 | 180 |
| TensorRT, SCRFD 640 | 5,3 / 9,3 ms | 3,7 / 18,9 ms | 411 | 18 | 177 |
| TensorRT, SCRFD 384 | 2,5 / 6,1 ms | 3,3 / 16,7 ms | 413 | 15 | 184 |
| **TensorRT, SCRFD 320** | **2,2 / 5,8 ms** | **3,3 / 16,3 ms** | **489** | 18 | **100** |
| TensorRT, SCRFD 256 | 1,9 / 5,5 ms | 3,3 / 17,7 ms | 464 | 16 | 126 |

Dans `FaceRecognizer`, sur 336 crops : 5,7 ms par crop en médiane avec
TensorRT, 12,6 ms en CUDA (SCRFD 320 dans les deux cas). TensorRT et CUDA
donnent les mêmes embeddings à 0,998 près (cosinus).

---

## Réglage du tracking

### Pistes en roue libre

Quand une personne est occultée ou sort du champ, DeepSORT garde sa piste
jusqu'à `DEEPSORT_MAX_AGE` images, avec une boîte prédite par le filtre de
Kalman. `FaceBodyTracker` garde l'identité de ces pistes en mémoire, mais ne
les affiche pas et ne les soumet pas à la reconnaissance faciale : seules les
pistes appariées à une détection YOLO de l'image courante le sont
(appariement un-pour-un par IoU). Avant, les boîtes fantômes restaient à
l'écran et leur crop visage, qui montrait l'obstacle, partait à InsightFace.

### Seuils DeepSORT

`tools/sweep_deepsort.py` balaie `max_cosine_distance`, `max_iou_distance`,
`max_age` et `n_init` sur des séquences de réglage, puis rejoue la meilleure
combinaison sur des séquences de validation qu'elle n'a jamais vues. Les
séquences MOT17 FRCNN les plus proches de VisionCam (05, 09, 11 : peu de
monde, filmé de près) servent au réglage, les quatre autres à la validation.

Résultats sur la validation (02, 04, 10, 13 ; détections publiques) :

| Pistes en roue libre | Seuils (cos / iou / age / init) | MOTA | IDF1 | Changements d'ID | FP |
|:---------------------|:--------------------------------|-----:|-----:|-----------------:|---:|
| affichées (avant) | 0.2 / 0.7 / 70 / 3 (actuels) | 21,5 % | 48,3 % | 715 | 31 865 |
| affichées | 0.15 / 0.3 / 30 / 5 (meilleurs) | 39,2 % | 50,5 % | 310 | 12 407 |
| **masquées** | **0.2 / 0.7 / 70 / 3 (actuels)** | **44,2 %** | **54,4 %** | 550 | 6 471 |
| masquées | 0.15 / 0.5 / 70 / 5 (meilleurs) | 43,9 % | 52,3 % | 348 | 5 568 |

Le gain vient de masquer les pistes en roue libre, pas des seuils : une fois
les fantômes masqués, les seuils réglés sur MOT17 gagnent sur les séquences de
réglage (IDF1 62,5 % contre 54,9 %) mais perdent sur la validation.

### Validation sur trois jeux de données

MOT17 filme des foules de loin. Deux jeux plus proches de VisionCam ont servi
à confirmer ou écarter les candidats, toujours en réglage puis validation sur
des vidéos distinctes, pistes en roue libre masquées :

- [CHIRLA](https://huggingface.co/datasets/bdager/CHIRLA) (CC-BY 4.0) : bureau
  filmé à 30 i/s, 2 à 3 personnes vues de près, visages visibles. Réglage sur
  5 vidéos de juin-juillet, validation sur 5 de décembre (autres jours, autres
  tenues). Conversion : `tools/convert_chirla.py`.
- [DanceTrack](https://github.com/DanceTrack/DanceTrack) (recherche non
  commerciale) : 4 à 7 danseurs vus de près, beaucoup de croisements, mais
  costumes identiques, ce qui rend l'apparence peu fiable. Réglage sur 5 vidéos
  de `train1`, validation sur 5 de `val`. Détections :
  `tools/record_sequence.py --detect-only`.

Effet de chaque candidat seul, depuis `cos=0.2 iou=0.7 age=70 n_init=3`, sur
les séquences de validation (écarts en points, changements d'identité en %) :

| Candidat | MOT17 : MOTA / IDF1 / IDs | DanceTrack | CHIRLA |
|:---------|:--------------------------|:-----------|:-------|
| **`n_init=5`** | −0,1 / −0,4 / **−17 %** | −0,4 / +1,6 / **−10 %** | −0,6 / +0,5 / **−12 %** |
| `cos=0.15` | −0,1 / −2,1 / +3 % | −0,8 / −9,9 / +24 % | non retenu |
| `max_age=150` (avec `n_init=5`) | 0,0 / −1,0 / +5 % | +0,2 / +1,2 / 0 % | +0,0 / +0,5 / 0 % |

**`n_init=5` est retenu** : c'est le seul changement qui réduit les changements
d'identité sur les trois jeux, pour un IDF1 à ±1,6 point et un MOTA à −0,6 au
pire. Le seuil cosinus idéal dépend de la scène (0,15 sur MOT17, 0,25 à 0,3
sur DanceTrack et CHIRLA) : 0,2 reste le compromis. `max_age=150` perd sur
MOT17 : 70 est conservé.

Sur CHIRLA, l'IDF1 absolu reste bas (23 à 26 %) : ses annotations gardent la
même identité à une personne pendant toute la vidéo, même après une longue
sortie du champ, alors que DeepSORT ouvre une nouvelle piste passé `max_age`.
Dans VisionCam, c'est la reconnaissance faciale qui redonne le nom au retour.

Pour valider sur ta propre caméra :

```bash
uv run -m tools.record_sequence --seconds 90 --output ~/datasets/visioncam/scene-01
uv run -m tools.annotate_sequence ~/datasets/visioncam/scene-01   # corriger les identités
uv run -m tools.sweep_deepsort --only-updated --tune ~/datasets/visioncam/scene-01 \
    --grid init=3,5 --validate ~/datasets/visioncam/scene-02
```

---

## Tracker Rust (optionnel)

L'association de tracks — filtre de Kalman, matching cascade, passage IoU —
existe en deux implémentations interchangeables, choisies par
`TRACKER_BACKEND` :

| Valeur | Implémentation | Installation |
|:-------|:---------------|:-------------|
| `python` (défaut) | `deep_sort_realtime` 1.3.2 | installé par `uv sync` |
| `rust` | crate [`deepsort-rs`](https://github.com/Armand-LAUENER/deepsort-rs) via PyO3 | installé par `uv sync` (groupe `rust`), cf. ci-dessous |

Seule l'association passe en Rust. La détection (YOLOv8-Pose), la
reconnaissance faciale (InsightFace) et l'embedder d'apparence
(MobileNetV2) restent identiques : les deux backends reçoivent les vecteurs du
même embedder (`core/appearance_embedder.py`), calculés sur les mêmes crops,
ce qui les rend comparables à l'identique.

### Installation

`uv sync` construit le crate depuis git, au commit figé dans `pyproject.toml`
(il faut `cargo`). Le build passe par le backend PEP 517 de maturin, en
`--release` par défaut : un build debug fausserait complètement la
comparaison. Pour ne pas l'installer : `uv sync --no-group rust`.

Pour travailler sur le crate en parallèle, remplacer temporairement la version
figée par le dépôt local, puis lancer sans resynchroniser :

```bash
uv pip install -e ../deepsort-rs
uv run --no-sync app.py
```

Puis `TRACKER_BACKEND=rust` dans `.env`.

### Revenir en arrière

Remettre `TRACKER_BACKEND=python` (ou retirer la ligne du `.env`) et
redémarrer. Aucune désinstallation nécessaire, aucun autre fichier à toucher :
c'est le seul point de bascule.

### Résultats mesurés

`uv run -m tools.bench_tracker --video <fichier> --frames 400` fait tourner les
deux backends sur les mêmes détections YOLO, frame par frame, et compare pistes
et temps. Les séquences MOT17 se passent directement en motif d'images, par
exemple `--video ~/datasets/MOT17/train/MOT17-04-FRCNN/img1/%06d.jpg`.

| Séquence | Personnes/frame | `python` (méd. / p95) | `rust` (méd. / p95) | Parité |
|:---------|:----------------|:----------------------|:--------------------|:-------|
| MOT17-04 | 7,5 (max 12) | 8,0 / 11,2 ms | 10,3 / 14,8 ms (×0,78) | 350/350 frames identiques |
| MOT17-09 | 7,4 (max 12) | 9,1 / 13,2 ms | 10,7 / 16,7 ms (×0,84) | 350/350 frames identiques |

Temps par frame, 50 frames de warm-up, RTX 4060, YOLO et embedder en
TensorRT. Les deux colonnes incluent le même embedder sur les mêmes crops. Sur
l'association seule, le crate mesure ×5,8 à ×20,6 selon la densité (cf. son
README), mais ce gain ne se retrouve pas sur l'étape complète : côté `rust`,
elle inclut aussi la conversion des embeddings en float32 et le passage des
tableaux vers le crate. **Le backend `python` est le plus rapide**, et reste le
défaut. L'embedder accéléré a creusé l'écart (×0,90 avec l'embedder
d'origine) : il pesait sur les deux colonnes à la fois.

Les chiffres publiés auparavant (×1,42 et ×1,20 en faveur de `rust`) ont été
mesurés avant que les outils fixent `OPENBLAS_NUM_THREADS=1`. Les threads
OpenBLAS se disputaient alors le CPU avec torch et ralentissaient surtout le
backend `python`, qui fait son association en numpy : sa médiane passait à
21,9 / 29,5 ms et son p95 à 83 / 98 ms sur ces mêmes séquences.

« Parité » = mêmes `track_id` et mêmes boîtes à 1e-3 px, pistes tentatives
comprises.

---

## Enrôlement

### Option 1 — Dossier `known_faces/`

Créer un sous-dossier par personne et y placer 3 à 5 photos :

```
known_faces/
├── Alice/
│   ├── alice_1.jpg
│   └── alice_2.jpg
└── Bob/
    └── bob.jpg
```

Supprimer `data/embeddings.npz` pour forcer la reconstruction, puis relancer.

### Option 2 — Interface web (à chaud, sans redémarrer)

- **Live** : cliquer sur une personne, saisir son nom. Le mode guidé enregistre face, profil gauche et profil droit.
- **Personnes** : ajouter des photos, renommer, supprimer, reconstruire la base.

### Option 3 — API REST

**Depuis des fichiers** (crée la personne si besoin) :
```bash
curl -X POST http://localhost:5000/api/people/Alice/photos \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"
```

**Depuis le flux caméra** (`track_id` : identifiant de piste visible dans `GET /status`) :
```bash
curl -X POST http://localhost:5000/api/capture \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "track_id": 3}'
```

**Reconstruire la base depuis `known_faces/` :**
```bash
curl -X POST http://localhost:5000/rebuild
```

Avec un mot de passe configuré, ajouter un cookie de session : `curl -c jar -d password=… http://localhost:5000/login`, puis `-b jar` sur chaque appel.

Noms et labels acceptés : lettres (accents compris), chiffres, `_`, `-`, espace et point, sans point initial, 64 caractères max.

Réenrôler une personne ajoute les nouvelles photos aux anciennes : l'embedding est la moyenne de toutes ses photos, comme après une reconstruction.

Le mode guidé de la page Live garde un vecteur par angle (face, profils) : plus précis sur les grands changements de pose qu'un embedding moyen.

---

## Accès protégé

Définir un mot de passe dans `.env` pour exiger une connexion sur toutes les pages et toute l'API :

```bash
uv run python tools/set_password.py            # saisie masquée, écrit ADMIN_PASSWORD_HASH dans .env
uv run python tools/set_password.py --remove   # retour à l'accès ouvert
```

Seule la ligne `ADMIN_PASSWORD_HASH` est modifiée ; le mot de passe n'apparaît ni à l'écran ni dans l'historique du shell. Redémarrer l'application ensuite.

Sans session, une page redirige vers `/login` et l'API répond 401. Après 5 échecs en 5 minutes depuis une même adresse, les tentatives sont bloquées. Le cookie de session est `HttpOnly` et `SameSite=Lax` ; sa clé de signature est `SECRET_KEY` ou, à défaut, générée au premier lancement dans `data/secret_key`.

---

## Endpoints API

| Méthode | Chemin | Description |
|:--------|:-------|:------------|
| `GET` | `/` | Interface web (stream + liste de présence) |
| `GET` `POST` | `/login` | Connexion, si un mot de passe est configuré |
| `POST` | `/logout` | Déconnexion |
| `GET` | `/video` | Flux MJPEG `multipart/x-mixed-replace` |
| `GET` | `/status` | `{ currently_present[], fps, total_known, tracks[], frame_size }` — `tracks` : boîtes des personnes visibles |
| `POST` | `/api/capture` | Enrôle la personne cliquée (`{name, track_id, label?}`), label `Face`/`ProfilG`/`ProfilD` en mode guidé |
| `POST` | `/rebuild` | Reconstruction base embeddings depuis `known_faces/` (409 si déjà en cours) |
| `POST` | `/bench/pose` | Compare les deux sources d'orientation sur le flux |
| `GET` | `/api/people` | Personnes connues : photos, labels multitemplate, miniature |
| `GET` | `/api/people/<nom>/thumbnail` | Miniature de la première photo |
| `POST` | `/api/people/<nom>/rename` | Renommer (`{"new_name": …}`) : dossiers et base ensemble |
| `DELETE` | `/api/people/<nom>` | Supprimer photos et entrées (droit à l'effacement) |
| `POST` | `/api/people/<nom>/photos` | Ajouter des photos ; l'embedding est recalculé sur toutes |
| `GET` | `/api/history` | Sessions de présence (`name`, `from`, `to` en AAAA-MM-JJ, `limit`) |
| `GET` | `/api/history.csv` | Mêmes filtres, export CSV |
| `GET` | `/api/events` | Server-Sent Events : état toutes les 0,5 s, arrivées, départs, inconnus, enrôlements |
| `GET` | `/api/diagnostics` | FPS, temps par étape (médiane/p95), GPU, modèles et réglages actifs |

---

## Tests

```bash
uv run playwright install --only-shell chromium   # une fois, pour tests/ui
uv run pytest tests/
# 284 tests — 0 GPU requis, ~15 s
uv run ruff check .
```

`tests/ui` ouvre les pages dans Chromium headless : erreurs JavaScript, capture par clic, renommage, historique, affichage sur téléphone. Sans navigateur installé, ces tests sont ignorés en local ; en CI, ils échouent.

---

## Limitations connues

- La reconnaissance est optimale avec 3-5 photos minimum par personne, prises sous angles variés
- DeepSORT peut inverser des IDs lors de croisements serrés (corrigé au frame suivant par la reconnaissance)
- Sans `ADMIN_PASSWORD_HASH` ni `ADMIN_PASSWORD`, l'accès est ouvert à tout le réseau local (un avertissement s'affiche au démarrage). Le serveur parle HTTP : sur un réseau non maîtrisé, le placer derrière un proxy HTTPS
- CPU fallback disponible mais déconseillé en temps réel (InsightFace seul : ~200 ms/face)

---

## Licence & confidentialité

Code source disponible à des fins d'apprentissage et de démonstration.

**Aucune donnée biométrique** (photos, embeddings) n'est incluse dans ce dépôt.
Les dossiers `known_faces/` et `data/` sont exclus par `.gitignore` (RGPD).

Le modèle InsightFace est soumis aux conditions d'usage d'Insightface — à vérifier avant tout déploiement commercial.

---

## Auteur

**Armand Lauener** — [armand.lauener@outlook.fr](mailto:armand.lauener.26@gmail.com)
