# Backlog — Optimisations et améliorations VisionCam

Relevé fait le 2026-09-14 en relisant `app.py`, `core/face_body_tracker.py`,
`core/face_recognition.py` et `config.py` (état du commit `11c8fa1`).

Aucun chiffre de cette liste n'a été mesuré : les gains de performance sont des
estimations, à confirmer au bench avant de retenir quoi que ce soit.

Déjà explorés à la session précédente, donc absents d'ici : `det_size` 320 pour
InsightFace (la config est restée à 640) et le traitement groupé des visages pour la reconnaissance.

---

## 1. Bugs

- [x] **Personnes jamais reconnues** — `core/face_body_tracker.py:137`
  `tracks_to_recognize[:2]` prend toujours les deux premiers tracks. S'ils
  restent « Inconnu » (deux inconnus, deux personnes de dos), les tracks
  suivants ne sont jamais soumis à InsightFace.
  → Choisir ceux qui attendent depuis le plus longtemps (dernier essai le plus ancien), ou
  tourner à tour de rôle. Test de régression : 3 tracks, les 2 premiers
  toujours inconnus, le 3e doit finir par être reconnu.

- [x] **`/capture` enrôle à partir d'une image déjà annotée** — `app.py:448-458`
  La route décode `state.current_frame`, c'est-à-dire le JPEG qualité 75 avec
  les boîtes et les textes dessinés dessus. Il faudrait prendre
  `state.bench_frame`, l'image brute. Autre problème : s'il y a plusieurs
  personnes à l'écran, c'est le plus grand visage qui est enregistré.

- [x] **Écriture de fichiers hors du dossier prévu** — `core/face_recognition.py:233`
  `name` vient du formulaire sans contrôle : `name=../../x` écrit des JPEG en
  dehors de `known_faces/`. Même problème pour `label` (nom de fichier) en
  mode multitemplate.
  → N'accepter qu'une liste blanche de caractères, et vérifier que le chemin
  résolu reste bien dans `known_faces/`.

- [x] **Personnes brièvement « Inconnu » pendant `/rebuild`** — `core/face_recognition.py:114`
  `_build_database` vide la base en mémoire sans verrou pendant que la
  reconnaissance continue de la lire. Rien n'empêche non plus deux
  reconstructions en même temps.
  → Construire la nouvelle base à part puis la remplacer d'un coup sous
  `_lock`, et refuser une reconstruction si une autre est déjà en cours.

- [x] **Commentaire contraire au code** — `core/face_body_tracker.py:8-9`
  Il annonce un « skip automatique si nez absent », mais le code retombe sur
  un crop en haut du corps et lance quand même la détection de visage.
  → Soit sauter vraiment ce cas (économise des appels GPU inutiles sur les
  personnes de dos), soit corriger le commentaire.

## 2. Performances

- [x] **Encodage JPEG hors du verrou** — `app.py:335`
  `cv2.imencode` en 1080p s'exécute alors que `state.lock` est tenu, à chaque frame :
  `/status` et le flux vidéo attendent la fin de l'encodage. Encoder avant, ne
  prendre le verrou que pour l'affectation.

- [x] **Flux vidéo : n'envoyer que les nouvelles frames** — `app.py:352`
  `generate_frames` renvoie la même frame à 30 FPS, et le pipeline encode
  même sans client connecté. Utiliser un `threading.Condition` ou un compteur de frames ;
  n'encoder que si au moins un client écoute.

- [ ] **YOLO en TensorRT FP16**
  `predict` tourne en PyTorch FP32, sans `half`.
  `yolo export model=yolov8s-pose.pt format=engine half=True`, puis
  `YOLO_MODEL = "yolov8s-pose.engine"`. Souvent ×2 à ×3 sur l'inférence
  YOLO sur ce type de GPU — à mesurer.

- [ ] **Embedder MobileNetV2 du tracker**
  D'après le README, c'est lui qui domine le temps du tracker (déjà en `half=True`).
  Pistes : TensorRT, ou le calculer une frame sur deux quand l'association par
  IoU n'est pas ambiguë (vérifier MOTA/IDF1 avec `tools/eval_mot.py`).

- [ ] **`POSE_SOURCE="yolo"` par défaut** — `config.py:121`
  MediaPipe fait une inférence CPU par personne toutes les `FRAME_SKIP`
  frames. `tools/pose_threshold_study.py` donne 100 % d'accord avec MediaPipe
  au-delà de 100 px d'écart d'épaules, et la source YOLO ne coûte rien.

- [x] **Recherche vectorisée dans `_identify`** — `core/face_recognition.py:261`
  Boucle Python et copie des listes à chaque appel. Garder une matrice
  pré-empilée (reconstruite à l'enrôlement) et faire un seul `argmax`. Gain
  faible, sauf avec beaucoup de personnes enrôlées.

## 3. Qualité et réglages

- [ ] **Seuils DeepSORT jamais réglés pour VisionCam** — `config.py:86-93`
  `tools/eval_mot.py` sait balayer `DEEPSORT_MAX_COSINE_DISTANCE`, mais MOT17
  filme de loin. Annoter quelques minutes de vidéo webcam pour régler les
  seuils sur le cas réel.

- [ ] **`RECOGNITION_THRESHOLD` incohérent**
  Le README annonce 0.65, `config.py` vaut 0.45, et le commentaire « plus bas =
  plus strict » est faux en similarité cosinus (c'est l'inverse).

## 4. Hygiène du projet

- [ ] **README à jour**
  Il manque `core/pose_from_keypoints.py`, `tools/eval_mot.py`,
  `tools/webcam_bridge.py`, `tools/pose_threshold_study.py` et les nouveaux
  tests dans l'arborescence ; le nombre de tests est à revérifier ;
  l'installation est encore décrite avec `pip`/`requirements.txt`.

- [ ] **Migration vers uv**
  `pyproject.toml` n'a pas de section `[project]` ; pas de `uv.lock`.

- [ ] **CI**
  Pas de `.github/`. Un workflow `pytest` + `ruff` suffit : les tests ne
  demandent pas de GPU.

- [ ] **Déploiement**
  Serveur de dev Flask (`app.run`) → waitress ou gunicorn. Cache d'embeddings
  en `pickle` → `.npz` (le chargement d'un `pickle` altéré peut exécuter du code).

---

## Ordre suggéré

1. Bugs de la section 1, un commit par bug, chacun avec son test de régression.
2. Encodage JPEG hors du verrou et flux vidéo (petits changements, effet direct sur la latence).
3. TensorRT pour YOLO puis l'embedder — probablement le plus gros gain de FPS
   restant, à confirmer par mesure avant/après.
