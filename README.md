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
- **Estimation de pose** — MediaPipe (debout / assis / allongé / autre)
- **Tracking par corps** — identité maintenue même quand le visage disparaît
- **Enrôlement à chaud** — ajout de personnes sans redémarrer l'application
- **Streaming MJPEG** — flux annoté en direct dans le navigateur
- **Reconnexion automatique** — backoff exponentiel 1 s → 30 s pour les caméras IP

---

## Architecture

```
┌───────────────────┐     ┌───────────────────────┐
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
       │   1. YOLOv8-Pose  → corps + keypoints squelette   │
       │   2. DeepSORT      → track_id persistant / ReID   │
       │   3. InsightFace   → crop centré sur tête (1/5f)  │
       │      ├─ centroïde keypoints visibles (lunettes ✓) │
       │      └─ fallback body-top si dos tourné           │
       │   4. Vote buffer   → anti-flip identité           │
       │   5. PoseEstimator → 4 classes MediaPipe          │
       └────────────┬───────────────────────────────────────┘
                    ▼
       ┌────────────────────┐     ┌──────────────────────────┐
       │  AppState (lock)   │────►│  Flask (thread principal) │
       │  current_frame     │     │  GET  /                  │
       │  currently_present │     │  GET  /video  (MJPEG)    │
       └────────────────────┘     │  GET  /status (JSON)     │
                                  │  POST /enroll            │
                                  │  POST /capture           │
                                  │  POST /rebuild           │
                                  └──────────────────────────┘
```

### Feedback visuel

| Couleur | Signification |
|:--------|:--------------|
| **Vert** | Visage reconnu récemment (< `FACE_FRESHNESS_FRAMES`) |
| **Orange** `[BODY]` | Identité connue, tracking corps seul |
| **Bleu** | Personne inconnue |

---

## Stack technique

| Domaine | Outil | Version |
|:--------|:------|:--------|
| Langage | Python | 3.11+ |
| Web | Flask | 3.1 |
| Reconnaissance faciale | InsightFace (antelopev2) + ONNX Runtime GPU | 0.7.3 |
| Détection corps | YOLOv8-Pose (ultralytics) | 8.4 |
| Tracking | DeepSORT + ReID MobileNet GPU | 1.3.2 |
| Pose estimation | MediaPipe | 0.10 |
| Traitement image | OpenCV | 4.13 |
| GPU | PyTorch CUDA 12.1 | 2.5.1 |

---

## Structure du projet

```
VisionCam/
├── app.py                  # Entrée : Flask + 2 threads (caméra + pipeline AI)
├── config.py               # Source unique de toutes les constantes
├── requirements.txt        # Dépendances Python
├── .env.example            # Template configuration locale
│
├── core/
│   ├── face_recognition.py   # FaceRecognizer — InsightFace + enrôlement
│   ├── face_body_tracker.py  # FaceBodyTracker — orchestrateur YOLO+DeepSORT
│   └── pose_estimation.py    # PoseEstimator — MediaPipe 4 classes
│
├── templates/
│   └── index.html           # UI : stream + liste de présence
│
├── tests/
│   ├── test_face_body_association.py      # 18 tests géométriques (0 GPU)
│   └── regression/
│       └── test_pose_exposed_in_status.py # Tests régression pose
│
├── known_faces/             # Non inclus (RGPD) — voir section Enrôlement
└── data/                    # Non inclus — généré au démarrage
    └── embeddings.pkl       # Cache des embeddings (reconstructible)
```

---

## Installation

### Prérequis

- Python 3.11+
- GPU NVIDIA avec CUDA 12.1 **fortement recommandé** (CPU possible mais ~5× plus lent)
- Webcam OU caméra IP accessible sur le LAN

### Étapes

```bash
# 1. Cloner
git clone https://github.com/Armand-LAUENER/Project_VisionCam.git
cd Project_VisionCam

# 2. Environnement virtuel
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Dépendances (PyTorch CUDA 12.1 inclus)
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# Éditer .env si nécessaire (source vidéo, seuils, port)

# 5. Préparer le dataset (voir section Enrôlement)
mkdir known_faces\MonNom
# Y copier 3-5 photos du visage

# 6. Lancer
python app.py
```

L'interface est disponible sur `http://localhost:5000`.

---

## Configuration

Tous les paramètres sont dans `config.py` (surchargeable via `.env`) :

| Constante | Défaut | Rôle |
|:----------|:-------|:-----|
| `USE_LOCAL_CAM` | `True` | Webcam locale si `True`, caméra IP sinon |
| `REMOTE_SOURCE` | URL MJPEG | URL du flux caméra IP |
| `FLASK_PORT` | `5000` | Port du serveur Flask |
| `RECOGNITION_THRESHOLD` | `0.65` | Seuil cosinus min pour identifier (0–1) |
| `FACE_RECOGNITION_SKIP` | `5` | InsightFace toutes les N frames |
| `FACE_FRESHNESS_FRAMES` | `30` | Frames avant passage en mode orange [BODY] |
| `POSE_NOSE_CONF_THRESHOLD` | `0.5` | Confiance minimale d'un keypoint facial |
| `POSE_FACE_KP_MIN_VISIBLE` | `1` | Keypoints visibles min pour utiliser le centroïde |
| `YOLO_MODEL` | `yolov8s-pose.pt` | Modèle YOLO (auto-téléchargé) |
| `DEEPSORT_MAX_AGE` | `70` | Frames avant suppression d'un track perdu |
| `FRAME_SKIP` | `2` | Cadence pose estimation |

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

Supprimer `data/embeddings.pkl` pour forcer la reconstruction, puis relancer.

### Option 2 — API REST (à chaud, sans redémarrer)

**Depuis des fichiers :**
```bash
curl -X POST http://localhost:5000/enroll \
  -F "name=Alice" \
  -F "method=average" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"
```

**Depuis la frame courante du flux caméra :**
```bash
curl -X POST http://localhost:5000/capture \
  -F "name=Alice"
```

**Reconstruire la base depuis `known_faces/` :**
```bash
curl -X POST http://localhost:5000/rebuild
```

Méthodes d'enrôlement : `average` (embedding moyen — recommandé) ou `multitemplate` (un vecteur par angle — plus précis sur les grands changements de pose).

---

## Endpoints API

| Méthode | Chemin | Description |
|:--------|:-------|:------------|
| `GET` | `/` | Interface web (stream + liste de présence) |
| `GET` | `/video` | Flux MJPEG `multipart/x-mixed-replace` |
| `GET` | `/status` | `{ currently_present[], fps, total_known }` |
| `POST` | `/enroll` | Enrôlement depuis fichiers image |
| `POST` | `/capture` | Enrôlement depuis la frame courante |
| `POST` | `/rebuild` | Reconstruction base embeddings depuis `known_faces/` |

---

## Tests

```bash
pytest tests/
# 23 tests — 0 GPU requis, ~0.2 s
```

---

## Limitations connues

- La reconnaissance est optimale avec 3-5 photos minimum par personne, prises sous angles variés
- DeepSORT peut inverser des IDs lors de croisements serrés (corrigé au frame suivant par la reconnaissance)
- Pas d'authentification sur les endpoints — conçu pour usage en LAN uniquement
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
