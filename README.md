# VisionCam

> Pipeline de **reconnaissance faciale en temps réel** sur flux vidéo,
> résistant aux lunettes, aux occlusions partielles et aux variations de distance.
> Exposé via une interface web MJPEG + API REST.

---

## Résultats clés

Mesuré sur RTX 4060 (WSL2), sur des jeux publics : aucune séquence filmée
par VisionCam n'est encore annotée, ces chiffres viennent de proxys.

| Mesure | Résultat | Détail |
|:-------|:---------|:-------|
| Reconnaissance, seuil 0,45 (CHIRLA, crops de l'application, visages ≥ 40 px) | 80,4 % de bons noms, **3,0 % de mauvais** [IC 95 % : 1,9-4,6], 16,6 % « Inconnu » | [performance](docs/performance.md#seuil-de-reconnaissance) |
| Nom affiché de bout en bout (CHIRLA, application complète, personnes enrôlées) | **35,5 %** du temps le bon nom, **1,1 %** un mauvais, le reste « Inconnu » ou sans piste ; 2,6 % / 0,0 % sur une scène filmée de loin | [performance](docs/performance.md#identité-de-bout-en-bout) |
| Masquer les pistes en roue libre (MOT17, validation) | MOTA 21,5 → **44,2 %**, IDF1 48,3 → **54,4 %** | [tracking](docs/tracking.md#seuils-deepsort) |
| `n_init=5` (MOT17, DanceTrack, CHIRLA, validation) | **−10 à −17 %** de changements d'identité, IDF1 à ±1,6 point | [tracking](docs/tracking.md#validation-sur-trois-jeux-de-données) |
| YOLOv8-Pose en TensorRT FP16 | 9,8 → **5,5 ms** (médiane), p95 33,4 → 7,9 ms | [performance](docs/performance.md#yolov8-pose-4-min-de-construction) |
| Embedder d'apparence en TensorRT FP16 | étape tracker 22,4 → **13,7 ms**, mêmes MOTA / IDF1 | [performance](docs/performance.md#embedder-dapparence-mobilenetv2-1-min-de-construction) |
| Reconnaissance (SCRFD 320 + TensorRT) | 12,6 → **5,7 ms** par visage | [performance](docs/performance.md#reconnaissance-faciale-insightface-2-min-au-premier-lancement) |
| Endurance : 8 h en continu, vidéo CHIRLA en boucle à 30 i/s | 30,0 i/s tenus, RSS 2 376 Mo (+0,4 Mo), VRAM et structures internes constantes | [performance](docs/performance.md#endurance) |
| Tracker Rust (`deepsort-rs`) | pistes identiques, mais **×0,78-0,84** sur l'étape complète : `python` reste le défaut | [tracker Rust](docs/rust-tracker.md#résultats-mesurés) |

Chaque visage est jugé seul dans la ligne reconnaissance : l'application vote
sur 3 reconnaissances par piste avant de nommer.

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
│   ├── performance.md           # TensorRT et seuil de reconnaissance, mesurés
│   ├── tracking.md              # Réglage et validation du tracking
│   ├── rust-tracker.md          # Backend d'association en Rust
│   ├── api.md                   # Routes HTTP
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
| `CAMERAS` | vide | Plusieurs caméras : `id=source;id=source` (webcam, URL, vidéo ou séquence) ; vide = une seule, `CAMERA_ID` |
| `CAMERA_WORKERS` | `thread` | `process` : caméras traitées dans des processus à part, `CAMERAS_PER_WORKER` (2) chacun ; au-delà de 2 caméras, seul `process` tient 30 i/s |
| `CAMERA_ID` | `cam0` | Identifiant de la caméra dans le journal d'événements (table `events` de `presence.db`) |
| `YOLO_NMS_IOU` | `0.6` | Recouvrement au-delà duquel YOLO fusionne deux boîtes ; 0,6 évite les pistes dédoublées d'une personne assise |
| `NESTED_BOX_MIN_CONTAINED` | `0.9` | Une boîte contenue à ce point dans une autre plus sûre (et IoU ≥ `NESTED_BOX_MIN_IOU`, 0,5) est écartée ; `0` désactive |
| `IDENTITY_PROTECT_FRAMES` | `90` | Une piste visible reconnue sous son nom depuis moins de ce nombre d'images le garde quand une autre l'obtient (sauf confiance plus haute de `IDENTITY_STEAL_MARGIN`, 0,05) |
| `VIDEO_FILE` | vide | Vidéo, motif d'images ou séquence MOT17 à la place de la caméra |
| `VIDEO_MODE` | `realtime` | `realtime` : cadence d'origine, images sautées si le pipeline est lent ; `every_frame` : toutes les images, déterministe |
| `VIDEO_LOOP` | `false` | Relire la vidéo en boucle (test d'endurance) |
| `TRACKS_LOG_PATH` | vide | CSV des pistes affichées, une ligne par piste et par image |
| `ENDURANCE_LOG_PATH` | vide | Test d'endurance : CSV de mémoire, VRAM, FPS et taille des structures internes ; analyse avec `uv run -m tools.endurance_report` |
| `ENDURANCE_INTERVAL_S` | `60` | Intervalle (s) entre deux lignes du journal d'endurance |
| `FLASK_PORT` | `5000` | Port du serveur HTTP |
| `PRESENCE_LOG_GAP_S` | `60` | Absence (s) après laquelle une session de présence se ferme |
| `PRESENCE_RETENTION_DAYS` | `30` | Conservation de l'historique (jours) ; `0` = sans limite |
| `PRESENCE_FACE_MAX_AGE_S` | `300` | Une piste n'inscrit son nom dans l'historique que si son visage a été reconnu depuis moins de ce délai (s) |
| `ADMIN_PASSWORD_HASH` / `ADMIN_PASSWORD` | vide | Mot de passe exigé pour tout l'accès ; vide = accès ouvert au réseau local |
| `SESSION_DAYS` | `7` | Durée d'une session de connexion |
| `SERVER_THREADS` | `32` | Threads waitress ; chaque onglet ouvert en garde deux (flux vidéo + événements) |
| `UNKNOWN_ALERT_S` | `3` | Délai avant de signaler une personne visible restée inconnue |
| `RECOGNITION_THRESHOLD` | `0.45` | Similarité cosinus min pour identifier (0–1, plus haut = plus strict) |
| `RECOGNITION_MIN_FACE_PX` | `40` | Taille min d'un visage pour décider d'un nom ; en dessous, la piste garde le sien |
| `RECOGNITION_UNKNOWN_STREAK` | `3` | Visages nets reconnus « Inconnu » d'affilée avant qu'une piste perde son nom |
| `FRONTAL_NOSE_MARGIN` | `0.25` | Seuls les visages de face comptent dans cette série : une personne enrôlée qui se retourne garde son nom |
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
| `TRACKER_BACKEND` | `python` | Association de tracks : `python`, `rust` (DeepSORT), ou `bytetrack`, `botsort`, `botsort-reid` (ultralytics, comparés dans `docs/tracking.md`) |

---

## Documentation

- [docs/performance.md](docs/performance.md) — accélération TensorRT (construction des moteurs) et seuil de reconnaissance
- [docs/tracking.md](docs/tracking.md) — réglage et validation du tracking, validation sur ta propre caméra
- [docs/rust-tracker.md](docs/rust-tracker.md) — backend d'association en Rust (optionnel)
- [docs/api.md](docs/api.md) — routes HTTP
- [docs/improvements-backlog.md](docs/improvements-backlog.md) — optimisations faites et restantes

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

## Tests

```bash
uv run playwright install --only-shell chromium   # une fois, pour tests/ui
uv run pytest tests/
# 304 tests (dont tests/ui) — 0 GPU requis
uv run ruff check .
```

`tests/ui` ouvre les pages dans Chromium headless : erreurs JavaScript, capture par clic, renommage, historique, affichage sur téléphone. Sans navigateur installé, ces tests sont ignorés en local ; en CI, ils échouent.

---

## Limitations connues

- La reconnaissance est optimale avec 3-5 photos minimum par personne, prises sous angles variés
- DeepSORT peut inverser des IDs lors de croisements serrés. Entre deux personnes enrôlées, la reconnaissance rétablit les noms à la prochaine passe. Vers une personne non enrôlée, le nom ne tombe qu'après `RECOGNITION_UNKNOWN_STREAK` visages nets et de face reconnus « Inconnu » d'affilée : jusqu'à ~1,5 s à 30 i/s si le visage est visible, jamais tant qu'il ne l'est pas (personne de dos)
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
