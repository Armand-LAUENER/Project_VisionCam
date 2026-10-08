# Backlog — Optimisations et améliorations VisionCam

Relevé fait le 2026-09-14 en relisant `app.py`, `core/face_body_tracker.py`,
`core/face_recognition.py` et `config.py` (état du commit `11c8fa1`).

Aucun chiffre de cette liste n'a été mesuré : les gains de performance sont des
estimations, à confirmer au bench avant de retenir quoi que ce soit.

Déjà explorés à la session précédente, donc absents d'ici : `det_size` 320 pour
InsightFace (la config est restée à 640) et le traitement groupé des visages pour la reconnaissance.
`det_size` a depuis été remesuré sur les crops de tête de l'application : cf. section 5.

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

- [x] **Le nom tombe quand une personne reconnue se retourne** —
  `core/face_body_tracker.py:590` (depuis `78f01aa`). La piste est conservée,
  mais SCRFD trouve encore un visage de profil ou de trois-quarts ≥ 40 px,
  ArcFace le juge « Inconnu », et au bout de `RECOGNITION_UNKNOWN_STREAK` le
  nom est retiré : le maintien du nom de dos est perdu.
  → Ne compter un « Inconnu » que pour un visage de face (nez entre les yeux
  d'après les 5 points SCRFD). Vérifier `test_swap_to_unenrolled` et
  mesurer `tools/eval_identity.py` avant/après sur CHIRLA.
  Fait : `FRONTAL_NOSE_MARGIN=0.25` ; sur `seq_026_camera_3`, 10 → 2 noms
  retirés, bons noms 37,4 → 39,6 %, mauvais noms inchangés
  (docs/performance.md).

## 2. Performances

- [x] **Encodage JPEG hors du verrou** — `app.py:335`
  `cv2.imencode` en 1080p s'exécute alors que `state.lock` est tenu, à chaque frame :
  `/status` et le flux vidéo attendent la fin de l'encodage. Encoder avant, ne
  prendre le verrou que pour l'affectation.

- [x] **Flux vidéo : n'envoyer que les nouvelles frames** — `app.py:352`
  `generate_frames` renvoie la même frame à 30 FPS, et le pipeline encode
  même sans client connecté. Utiliser un `threading.Condition` ou un compteur de frames ;
  n'encoder que si au moins un client écoute.

- [x] **YOLO en TensorRT FP16**
  `predict` tourne en PyTorch FP32, sans `half`.
  `yolo export model=yolov8s-pose.pt format=engine half=True`, puis
  `YOLO_MODEL = "yolov8s-pose.engine"`. Souvent ×2 à ×3 sur l'inférence
  YOLO sur ce type de GPU — à mesurer.
  → Mesuré : ×1,8 à ×2,2 en médiane et p95 de ~33 à ~7-8 ms avec un moteur
  384x640 ; un moteur 640x640 change les détections. Cf. README.

- [x] **Embedder MobileNetV2 du tracker**
  D'après le README, c'est lui qui domine le temps du tracker (déjà en `half=True`).
  Pistes : TensorRT, ou le calculer une frame sur deux quand l'association par
  IoU n'est pas ambiguë (vérifier MOTA/IDF1 avec `tools/eval_mot.py`).
  → Fait : prétraitement GPU + `channels_last` par défaut (×1,45), moteur
  TensorRT FP16 en option (×2,67 sur l'embedder, étape tracker 19,1 → 13,7 ms,
  MOTA/IDF1 non dégradés sur MOT17). Un calcul sur deux images non retenu.

- [x] **`POSE_SOURCE="yolo"` par défaut** — `config.py:121`
  MediaPipe fait une inférence CPU par personne toutes les `FRAME_SKIP`
  frames. `tools/pose_threshold_study.py` donne 100 % d'accord avec MediaPipe
  au-delà de 100 px d'écart d'épaules, et la source YOLO ne coûte rien.

- [x] **Recherche vectorisée dans `_identify`** — `core/face_recognition.py:397`
  Boucle Python et copie des listes à chaque appel. Garder une matrice
  pré-empilée (reconstruite à l'enrôlement) et faire un seul `argmax`. Gain
  faible, sauf avec beaucoup de personnes enrôlées.
  Matrice réellement mise en cache le 2026-10-01 (avant : `np.stack` à chaque
  appel) : 11 → 4,5 µs par appel à 10 personnes, 3,7 → 0,3 ms à 1000.

## 3. Qualité et réglages

- [ ] **Seuils DeepSORT jamais réglés pour VisionCam** — `config.py:86-93`
  `tools/eval_mot.py` sait balayer `DEEPSORT_MAX_COSINE_DISTANCE`, mais MOT17
  filme de loin. Annoter quelques minutes de vidéo webcam pour régler les
  seuils sur le cas réel.
  - [x] Balayage MOT17 avec validation (`tools/sweep_deepsort.py`) : pas de
    meilleurs seuils, mais les pistes en roue libre sont maintenant masquées
    (MOTA 21,5 → 44,2 %, IDF1 48,3 → 54,4 % en validation). Cf. README.
  - [x] Validation sur CHIRLA (bureau, 30 i/s) et DanceTrack : `n_init=5`
    retenu (−10 à −17 % de changements d'identité sur les trois jeux),
    `cos=0.15` et `max_age=150` écartés. Cf. README.
  - [ ] Séquences webcam annotées (`tools/record_sequence.py`,
    `tools/annotate_sequence.py`) : dernière vérification sur le cas réel.

- [x] **`RECOGNITION_THRESHOLD` incohérent**
  Le README annonce 0.65, `config.py` vaut 0.45, et le commentaire « plus bas =
  plus strict » est faux en similarité cosinus (c'est l'inverse).

## 4. Hygiène du projet

- [x] **README à jour**
  Arborescence, nombre de tests, source d'orientation, routes et installation
  (uv) mis à jour.

- [x] **Migration vers uv**
  `pyproject.toml` n'a pas de section `[project]` ; pas de `uv.lock`.
  → Fait : groupes `cu121` / `cpu` exclusifs pour torch, `deepsort-rs` depuis
  git (groupe `rust`), `requirements.txt` supprimé.

- [x] **CI**
  Pas de `.github/`. Un workflow `pytest` + `ruff` suffit : les tests ne
  demandent pas de GPU.

- [x] **Déploiement**
  Serveur de dev Flask (`app.run`) → waitress ou gunicorn (nouvelle dépendance,
  à valider).
  → waitress (fonctionne aussi sous Windows, contrairement à gunicorn),
  `SERVER_THREADS=16` : chaque flux MJPEG ouvert garde un thread.
  - [x] Cache d'embeddings en `pickle` → `.npz` (le chargement d'un `pickle`
    altéré peut exécuter du code).

## 5. Deuxième passe (2026-09-15)

- [x] **Mot de passe sur la machine** — `tools/set_password.py` : saisie
  masquée, seule la ligne `ADMIN_PASSWORD_HASH` de `.env` est écrite.
  Reste à le lancer sur la machine.

- [x] **Durée de conservation de l'historique** — `PRESENCE_RETENTION_DAYS=30` :
  les sessions terminées depuis plus longtemps sont supprimées au démarrage
  puis toutes les heures ; une session en cours n'est jamais supprimée.

- [x] **Anciennes routes `/enroll` et `/capture` retirées** — remplacées par
  `/api/people/<nom>/photos` et `/api/capture`, qui enrôlent la personne
  désignée et non le plus grand visage. `/rebuild` reste (page Personnes).

- [x] **Reconnaissance faciale plus légère** — `tools/bench_face.py` sur CHIRLA :
  SCRFD 640 → 320 (les crops de tête, 250 px en médiane, étaient agrandis et
  les visages trop gros ratés) et TensorRT FP16 (`INSIGHTFACE_TENSORRT`).
  Par crop 19,5 → 5,5 ms ; bons noms 410 → 489, mauvais 16 → 18, « Inconnu »
  180 → 100. Réutiliser les keypoints YOLO à la place de SCRFD n'a pas été
  tenté : YOLO n'a pas les coins de la bouche qu'attend l'alignement ArcFace.

- [x] **Lancement** — `scripts/visioncam.sh start|stop|status|logs` : pont
  webcam Windows puis application, arrêt propre (SIGTERM, sessions fermées).
  Pas de démarrage automatique avec Windows, par choix.

- [x] **Tests de l'interface en CI** — `tests/ui` : pages dans Chromium
  headless (Playwright), erreurs JavaScript, capture par clic, renommage,
  historique, largeur téléphone.

- [ ] **Séquences webcam annotées** — cf. section 3, à enregistrer sur la machine.

## 6. Observations à creuser (2026-10-07)

Relevées pendant les mesures de la 2.2, 2.3 et 2.7 ; rien n'est décidé ici.
Chiffres et protocoles dans docs/performance.md et docs/tracking.md.

### Charge et ressources

- **Le GIL plafonne chaque processus de caméras** vers 72-76 images traitées
  par seconde. À 8 caméras par processus, c'est pire : ~50-58 i/s par
  processus, la détection passe de ~10 à ~66 ms par image (attente du GIL).
  → Mesurer le nombre de caméras par processus qui maximise le débit ;
  regarder quelles parties Python tiennent le GIL (py-spy, profil par étape).
- **Un processus de caméras pèse ~2,6 Go de RAM**, l'application ~2 Go
  (torch, ultralytics, onnxruntime chargés partout). C'est ce qui borne le
  nombre de processus, donc de flux (16 flux : impossible de passer à 4
  processus dans les 15 Go de WSL). → Piste : un worker sans torch (moteurs
  TensorRT appelés directement, prétraitement et NMS en numpy) ; en mode
  process, l'application n'a pas non plus besoin de charger
  core.face_body_tracker (ultralytics, torch). Mesurer le gain de RAM avant.
- **Le GPU baisse sa fréquence à faible charge** (360 MHz à 2 × 5 i/s contre
  2 790 à pleine charge) : la latence est plus haute à 5 i/s qu'à 30 (p95
  100 ms contre 38). Le débit n'en souffre pas. → Voir si le verrouillage des
  fréquences (`nvidia-smi -lgc`) est possible sous WSL2, et si ça vaut la
  consommation.
- **Le « GPU de WSL » vu par Windows n'est pas fiable** pour juger une
  saturation (97 % à 2 × 30, 82 % à 4 × 30, 31 % puis 88 % à charges
  voisines). NVML et les i/s tenus le sont plus. → Le documenter comme
  contexte seulement.
- **`wsl_vram_mb` varie d'un run à l'autre** à configuration égale (742 puis
  835 Mo à 2 caméras). → Plusieurs runs avant de comparer des VRAM.
- **Démarrage d'un processus de caméras : 10 à 45 s** avant la pleine
  cadence (moteurs, première inférence). → Mesurer précisément, et voir si
  une inférence de chauffe au lancement le réduit.
- **Décodage mesuré sur des AVI 1280×720 : 0,4-1,4 ms.** Le coût d'un vrai
  flux RTSP H.264 n'est pas mesuré (étape `read` : attente comprise). → À
  mesurer avec une caméra IP ; NVDEC à considérer au-delà de quelques flux.
- **Pas de contre-pression worker → application.** Si l'application traite
  les résultats moins vite qu'ils n'arrivent, le worker bloque sur l'envoi :
  ces images-là ne sont comptées nulle part. → Mesurer à 16 flux, compter les
  envois bloqués.
- **Un palier à 8 × 30 i/s avec `yolov8n-pose` n'a tenu que 0,7 i/s**
  (GPU à 100 %, pleine fréquence, aucune application Windows, 2026-10-07
  23:52) ; refait 40 min plus tard, il tient 21,2 i/s. Cause non trouvée. →
  Si ça se reproduit : `nvidia-smi` et un profil (py-spy) pendant le palier.
- **`/bench/pose` en mode process** demande une image au worker à chaque
  échantillon (délai 2 s chacun) : coût et durée non mesurés.

### Suivi

- **deepsort-rs 8484623** : 15 fois plus rapide à 40 personnes, mais aucun
  gain sur CHIRLA (3 à 6 personnes) : le « tracking » y est surtout
  l'embedder d'apparence (6,4 ms sur GPU), pas l'association.
  → Si le tracking compte, c'est l'embedder qu'il faut alléger ou espacer.
- **ByteTrack / BoT-SORT** suivent mieux (MOTA +4, −15 % de changements d'ID,
  10 points de couverture en plus) mais montent les mauvais noms (0,7 →
  1,0-1,2 %) à cause du second passage sur détections faibles.
  → Pistes non testées : réglage des seuils d'ultralytics (non réglés,
  contrairement à DeepSORT sur DanceTrack) ; second passage seulement pour
  prolonger une piste sans nom ; ne pas reconnaître de visage sur une boîte
  issue d'une détection faible.
- **La ré-identification de BoT-SORT ne joue qu'entre boîtes qui se
  recouvrent** (IoU ≥ 0,5) : elle ne rattache pas une personne sortie du champ.
- **DanceTrack a été enregistré avec YOLO à 0,5** : le second passage de
  ByteTrack n'y est pas testable. → Réenregistrer à 0,1 si on reprend la
  comparaison.
- **Le rejeu multi-caméras en `every_frame` reproduit exactement** les
  chiffres d'une caméra seule : le mode process est déterministe sur ce plan.

### Identité

- **Une petite perturbation des détections fait nommer une personne non
  enrôlée.** Avec le moteur YOLO dynamique (mêmes poids, FP16, construit à
  part), `seq_026_camera_3` passe de 0 à 21 % d'images où un non enrôlé
  porte un nom, et `seq_025_camera_5` de 0,1 à 2,1 % de mauvais noms
  (docs/performance.md, « YOLO par lots »). Ce n'est pas le moteur qui
  reconnaît mal : c'est la chaîne piste → nom qui amplifie un petit écart
  (échange de pistes que `RECOGNITION_UNKNOWN_STREAK` / `FRONTAL_NOSE_MARGIN`
  ne rattrapent pas). → Retrouver la piste en cause (`tracks.csv` du rejeu,
  `scratchpad`), voir pourquoi le nom n'est pas retombé ; c'est le cas réel
  qui manquait pour tester la protection de la 1.2.

- **Le facteur limitant des noms est le nombre de visages exploitables**, pas
  le suivi ni le partage entre caméras : sur `seq_025`, 90 % des crops de
  tête n'ont pas de visage ≥ 40 px ; passer un nom d'une caméra à l'autre
  rapporterait au mieux 3,3 % des suivis en « Inconnu ». → Pistes à mesurer :
  `RECOGNITION_MIN_FACE_PX`, enrôlement multi-angles, résolution des crops de
  tête, InsightFace à plus grande taille d'entrée sur les visages lointains.
- **Les champs de vision des caméras CHIRLA se recouvrent** : un anti-clonage
  entre caméras serait faux. Si un jour il en faut un, il dépendra de la
  topologie (2.4 : caméras sans recouvrement seulement).
- **`FRONTAL_NOSE_MARGIN`** : CHIRLA ne contient aucun échange de pistes vers
  une personne non enrôlée vue de face ; cette protection n'est vérifiée que
  par le test de régression. → Trouver ou enregistrer une séquence qui la
  met en scène.
- **Renommer une personne ne renomme pas les pistes en cours** (les trackers
  gardent l'ancien nom jusqu'à la prochaine reconnaissance), en mode thread
  comme en mode process.
- **Rechargement de la base dans les processus de caméras** : testé en unitaire
  seulement. → Essai à la main : enrôler depuis la page Live en
  `CAMERA_WORKERS=process`, vérifier la reconnaissance.

### Outillage

- **Une chaîne `pytest … | grep` a laissé passer un commit avec 6 tests en
  échec** (code de retour de grep, pas de pytest). → Toujours tester le code
  de retour de pytest lui-même (`pytest …; echo $?`).
- **Un test de régression installe un faux `ultralytics` dans `sys.modules`**
  (`setdefault`) : tout test collecté après qui importe le vrai paquet en
  hérite. Contourné dans test_tracker_backends.py. → Le remplacer par un
  monkeypatch limité au test, comme pour `build_body_tracker`.

---

## Ordre suggéré

1. Bugs de la section 1, un commit par bug, chacun avec son test de régression.
2. Encodage JPEG hors du verrou et flux vidéo (petits changements, effet direct sur la latence).
3. TensorRT pour YOLO puis l'embedder — probablement le plus gros gain de FPS
   restant, à confirmer par mesure avant/après.
