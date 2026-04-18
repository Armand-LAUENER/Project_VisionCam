# Project_VisionCam

> Application de vision par ordinateur temps réel combinant **reconnaissance faciale**,
> **tracking multi-personnes** et **estimation de pose** sur flux vidéo, exposée via
> une interface web Flask.

---

## Aperçu

Un pipeline d'IA vidéo qui :

1. Lit un flux vidéo — **webcam locale** ou **caméra IP** (MJPEG sur LAN).
2. Détecte et identifie les visages connus via **InsightFace** (modèle `buffalo_l`).
3. Suit chaque personne détectée avec un **tracker IoU** maison (ID persistant).
4. Diffuse le rendu annoté en **streaming MJPEG** sur une UI web minimaliste.
5. Expose un endpoint JSON `/status` listant les personnes actuellement visibles.

Le tout tourne dans un seul processus Python en ~300 lignes de code,
avec un thread daemon dédié au traitement et Flask pour le serveur web.

---

## Démo

```
┌────────────────────┐         ┌──────────────────┐
│ Webcam / Caméra IP │ ──────► │ processing_loop  │
└────────────────────┘         │  ├─ InsightFace  │
                               │  ├─ IoU Tracker  │
                               │  └─ JPEG encode  │
                               └────────┬─────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Flask — MJPEG stream + JSON │
                         │  GET /video  (stream)       │
                         │  GET /status (présence)     │
                         └─────────────────────────────┘
```

---

## Stack technique

| Domaine | Outil |
|:--|:--|
| Langage | Python 3.11+ |
| Web | Flask (streaming MJPEG + routes JSON) |
| Reconnaissance faciale | InsightFace (buffalo_l, ONNX Runtime GPU) |
| Tracking | Algorithme IoU greedy (implémentation maison) |
| Pose estimation | MediaPipe |
| Traitement image | OpenCV |
| Persistance | Pickle (cache embeddings) + arborescence fichier |

---

## Structure du projet

```
Project_VisionCam/
├── app.py                 # Entrée Flask + boucle de traitement
├── config.py              # Constantes (caméra, seuils, port)
├── requirements.txt
├── core/
│   ├── face_recognition.py   # FaceRecognizer (InsightFace + embeddings)
│   ├── tracker.py            # PersonTracker (association IoU)
│   └── pose_estimation.py    # PoseEstimator (MediaPipe)
├── templates/
│   └── index.html         # UI web (stream + liste présence)
└── known_faces/           # Dataset d'entraînement (non inclus — RGPD)
    └── <Personne>/*.jpg
```

---

## Installation

### Prérequis

- Python 3.11 ou supérieur
- Webcam OU caméra IP accessible sur le LAN
- GPU CUDA **recommandé** (fallback CPU possible mais lent)

### Étapes

```bash
# 1. Cloner
git clone https://github.com/Armand-LAUENER/Project_VisionCam.git
cd Project_VisionCam

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
.venv\Scripts\activate              # Windows

# 3. Dépendances
pip install -r requirements.txt

# 4. Préparer le dataset
mkdir -p known_faces/MonNom
# Y copier 3-5 photos claires du visage de la personne à reconnaître

# 5. Variables d'environnement (optionnel)
cp .env.example .env
# Éditer .env si besoin

# 6. Lancer
python app.py
```

L'application est ensuite disponible sur `http://localhost:8080`.

---

## Configuration

Tous les paramètres sont dans `config.py` :

| Constante | Défaut | Rôle |
|:--|:--|:--|
| `USE_LOCAL_CAM` | `True` | Webcam locale si `True`, caméra IP sinon |
| `LOCAL_SOURCE` | `0` | Index OpenCV de la webcam |
| `REMOTE_SOURCE` | URL MJPEG | Flux caméra IP |
| `RECOGNITION_THRESHOLD` | `0.45` | Seuil cosinus min pour identifier |
| `FRAME_SKIP` | `2` | Traiter 1 frame sur N (perf) |
| `FLASK_PORT` | `8080` | Port du serveur Flask |

Pour **ajouter une personne** : créer `known_faces/<Nom>/` et y placer quelques
photos. Supprimer `data/embeddings.pkl` pour forcer la reconstruction de la base.

---

## Endpoints

| Méthode | Chemin | Réponse |
|:--|:--|:--|
| GET | `/` | Interface web (stream + liste) |
| GET | `/video` | Flux MJPEG `multipart/x-mixed-replace` |
| GET | `/status` | `{ currently_present[], fps, total_known }` |

---

## Points techniques notables

- **Pipeline multi-thread** : le traitement IA tourne dans un thread daemon
  dédié, Flask dans le thread principal. Communication via un `AppState`
  protégé par `threading.Lock()`.
- **Frame skip** : la reconnaissance ne tourne qu'une frame sur N. Les frames
  intermédiaires réutilisent le cache de détections, ce qui double le FPS
  effectif sans perte visuelle notable.
- **Embedding moyen par personne** : pour chaque personne, plusieurs photos
  sont agrégées en un unique embedding moyen normalisé, plus robuste aux
  variations d'angle et d'éclairage.
- **Tracker IoU greedy** : association frame-à-frame par matrice IoU +
  matching glouton, avec TTL de 30 frames avant suppression d'un track perdu.

---

## Limitations connues

- La reconnaissance faciale n'est robuste qu'avec 3-5 photos minimum par personne
- Le tracker IoU peut mélanger les IDs lors de croisements serrés
  (corrigé au frame suivant par la reconnaissance)
- Pas d'authentification sur les endpoints — prévu pour usage en LAN uniquement
- `PoseEstimator` est instancié mais pas encore intégré dans `processing_loop`

---

## Licence & usage

Code disponible à des fins de démonstration et d'apprentissage.
Le modèle `buffalo_l` de InsightFace a ses propres conditions d'usage —
à auditer avant tout déploiement commercial.

**Aucune donnée personnelle** (photos, embeddings) n'est incluse dans ce repo.
Le dossier `known_faces/` et le fichier `data/embeddings.pkl` sont strictement locaux.

---

## Auteur

**Armand Lauener** — [armand.lauener@outlook.fr](mailto:armand.lauener.26@gmail.com)