# Performances mesurées

Accélération TensorRT de chaque réseau et choix du seuil de reconnaissance,
avec les mesures qui les justifient. Résumé : [README](../README.md#résultats-clés).

Deux réseaux peuvent tourner en moteur TensorRT FP16 : le détecteur
YOLOv8-Pose et l'embedder d'apparence MobileNetV2 du tracker. Un moteur dépend
du GPU et de la version de TensorRT qui l'ont construit : il n'est pas
versionné (`*.engine` est ignoré), chaque machine construit les siens, et les
reconstruit après une mise à jour de TensorRT ou du pilote. Sans moteur, le
pipeline tourne en PyTorch.

## YOLOv8-Pose (~4 min de construction)

```bash
# TensorRT et onnxslim font partie du groupe cu121, installé par uv sync
uv run yolo export model=yolov8s-pose.pt format=engine half=True device=0 imgsz=384,640
```

Puis `YOLO_MODEL=yolov8s-pose.engine` dans `.env`. Pour revenir en arrière,
retirer la ligne.

`imgsz=384,640` est la taille letterbox d'une image 16:9 ramenée à 640 px de
large, celle qu'utilise PyTorch sur le flux 1920x1080. Un moteur 640x640
ajoute un padding qui modifie les détections, sans rapport avec le FP16.

### Résultats mesurés

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

## Embedder d'apparence MobileNetV2 (~1 min de construction)

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

### Résultats mesurés

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

## Reconnaissance faciale InsightFace (~2 min au premier lancement)

La reconnaissance était l'étape la plus lourde : SCRFD en 640×640 puis
glintr100 (ResNet-100) en CUDA FP32, par crop de tête. Deux changements :

- **SCRFD en 320×320**, par défaut. Les crops de tête font 250 px en médiane :
  en 640 ils étaient agrandis, et les visages devenus trop gros étaient ratés.
  La base d'embeddings est reconstruite au démarrage quand cette taille change.
- **TensorRT FP16** pour les deux modèles, via le `TensorrtExecutionProvider`
  d'onnxruntime : `INSIGHTFACE_TENSORRT=true` dans `.env`. Les moteurs se
  construisent au premier lancement et sont gardés dans `data/trt_cache/`.

### Résultats mesurés

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

## Seuil de reconnaissance

`uv run -m tools.bench_face --gallery <…> --probe <…> --variants trt:320/trt
--thresholds 0.35 0.40 0.45 0.50 0.55 0.60 0.65` : mêmes crops et mêmes
séquences que ci-dessus, configuration déployée. 607 visages de test ≥ 40 px
de personnes enrôlées ; intervalles de confiance à 95 % (Wilson).

| Seuil | Bons noms | Mauvais noms [IC 95 %] | « Inconnu » |
|------:|----------:|-----------------------:|------------:|
| 0,40 | 83,9 % | 4,1 % [2,8-6,0 %] | 12,0 % |
| **0,45** (défaut) | **80,4 %** | **3,0 % [1,9-4,6 %]** | **16,6 %** |
| 0,50 | 64,9 % | 1,2 % [0,6-2,4 %] | 33,9 % |
| 0,55 | 27,2 % | 0,3 % [0,1-1,2 %] | 72,5 % |
| 0,60 | 1,0 % | 0,0 % [0,0-0,6 %] | 99,0 % |

Chaque visage est jugé seul : dans l'application, une piste ne prend un nom
qu'à la majorité de 3 reconnaissances (`VOTE_WINDOW`), ce que ce tableau ne
compte pas. Les personnes absentes de la base ne fournissent que 15 visages
de test (1 reconnu à tort à 0,45) : trop peu pour un taux.

L'étude précédente (`tools.recognition_threshold_study`, InsightFace sur
l'image entière) trouvait 0 % de mauvais noms à 0,45 au-delà de 40 px. Sur
les crops de tête de l'application, les scores de la bonne personne sont plus
bas : au-delà de 0,55, elle est presque toujours rejetée.

## Identité de bout en bout

Le seuil ci-dessus juge chaque visage seul. Ce que voit l'utilisateur dépend
aussi du tracking, du vote par piste, de la taille des visages et du temps
passé face caméra. `tools/eval_identity.py` le mesure : l'application rejoue
une séquence annotée en `every_frame` (`VIDEO_FILE`, `TRACKS_LOG_PATH`), puis
chaque personne annotée, à chaque image, reçoit une issue selon la piste
affichée qui la recouvre (IoU ≥ 0,5) : bon nom, mauvais nom, « Inconnu »,
ou pas de piste.

Protocole CHIRLA, configuration déployée : identités enrôlées avec
`tools/enroll_from_sequence.py` sur `seq_004_camera_2` et `seq_020_camera_4`
(10 crops de tête par personne, visages ≥ 40 px seulement : 13 identités),
rejeu de `seq_026_camera_3` et `seq_025_camera_2`, prises des mois plus tard.
Les personnes 9 et 14, sans visage exploitable à l'enrôlement, sont les
non-enrôlées.

| Séquence | Personnes enrôlées : bon nom / mauvais nom / « Inconnu » / sans piste | Non enrôlées : « Inconnu » / nommées à tort |
|:---------|:----------------------------------------------------------------------|:---------------------------------------------|
| `seq_026_camera_3` (7 754 images) | **35,5 %** / **1,1 %** / 38,1 % / 25,3 % | 85,9 % / 1,9 % |
| `seq_025_camera_2` (8 089 images) | 2,6 % / 0,0 % / 83,6 % / 13,8 % | 89,5 % / 0,0 % |

En % des images-personnes annotées. Sur `seq_026_camera_3`, la personne
assise face caméra pendant toute la séquence (id 6) a le bon nom 87 % du
temps ; celles qui ne font que passer ne sont presque jamais nommées (44
apparitions sur 49). Sur `seq_025_camera_2`, filmée de plus loin, 90 % des
crops de tête n'ont pas de visage d'au moins 40 px (`RECOGNITION_MIN_FACE_PX`) :
l'application ne décide presque jamais d'un nom, et ne se trompe pas.

Ce que ça dit : les mauvais noms restent rares (1 à 2 %) ; la limite est
le nombre de visages exploitables, pas le seuil. Le « 80 % de bons noms par
visage » du tableau précédent ne vaut que pour les visages assez grands et de
face, qui sont l'exception dans une scène de bureau vue de loin.

### Nom gardé quand la personne se retourne (`FRONTAL_NOSE_MARGIN`)

Depuis le correctif de l'échange de pistes (3 visages « Inconnu » d'affilée
retirent le nom), une personne enrôlée qui se retournait perdait son nom : de
profil ou de trois-quarts, SCRFD trouve encore un visage, qu'ArcFace juge
« Inconnu ». Seuls les visages de face comptent désormais dans cette série (nez
dans la partie centrale de l'écart entre les yeux, points SCRFD). Même
protocole, le 2026-10-07 :

| `seq_026_camera_3` | Noms retirés | Enrôlés : bon nom | dont id 2 | Mauvais nom | Non enrôlés nommés à tort |
|:--|---:|---:|---:|---:|---:|
| Avant (tout « Inconnu » compte) | 10 | 37,4 % | 17,4 % | 0,7 % | 0,0 % |
| Marge 0,10 | 4 | 38,1 % | 18,7 % | 0,7 % | 0,0 % |
| **Marge 0,25** (défaut) | **2** | **39,6 %** | **28,6 %** | 0,7 % | 0,0 % |
| Marge 0,40 | 2 | 39,6 % | 28,6 % | 0,7 % | 0,0 % |

Sur `seq_025_camera_2`, rien ne change (1 nom retiré avant comme après) :
les visages y sont presque tous trop petits pour être reconnus. 0,25 est la
plus petite marge qui atteint le palier : une marge plus grande écarterait
davantage de visages de face et affaiblirait la protection contre l'échange
de pistes, que CHIRLA ne met pas en scène (couverte par
`tests/regression/test_swap_to_unenrolled.py`).

Limites : deux séquences d'un même bureau ; les images d'une même personne ne
sont pas indépendantes (les intervalles de confiance de l'outil sont
indicatifs) ; « sans piste » mêle les personnes non détectées et les boîtes
dont l'IoU avec la vérité terrain reste sous 0,5 (personne assise en partie
masquée, id 5 : 72 %).

## Endurance

L'application complète tourne 8 h d'affilée sur une vidéo en boucle, avec une
mesure par minute (`ENDURANCE_LOG_PATH`), puis
`uv run -m tools.endurance_report <journal>` juge chaque colonne après 30 min
de warm-up. Source : CHIRLA `seq_025_camera_2` (8 089 images, 30 i/s) en
boucle et en `realtime`, configuration déployée (YOLO, embedder et
InsightFace en TensorRT), RTX 4060 sous WSL2, le 2026-10-02.

| Mesure | Après warm-up | Fin | Dérive sur 7 h 45 | Verdict |
|:-------|--------------:|----:|------------------:|:--------|
| FPS | 30,0 | 30,0 | +0,0 | stable |
| RSS | 2 375,7 Mo | 2 375,9 Mo | +0,4 Mo | stable |
| VRAM du GPU | 1 520 Mo | 1 520 Mo | 0 | stable |
| Structures internes (12) | — | — | maxima constants | bornées |

892 465 images traitées, soit 30,0 i/s en moyenne : le pipeline suit la
cadence de la source sans en sauter. Aucun trou dans le journal (pas de
blocage), aucune erreur dans les logs.

Dans ce premier run, aucune personne de CHIRLA n'était enrôlée : les
structures liées aux noms (votes, identités, sessions de présence) sont
restées vides, et aucune page web n'était ouverte. Le temps écoulé est mesuré
sur l'horloge monotone : 8,27 h pour 8 h d'horloge murale, que WSL2 recale en
arrière toutes les ~30 s.

### Mesures du journal

Depuis le 2026-10-03, chaque ligne du journal (`core/endurance.py`,
`core/system_probe.py`) contient, en plus du FPS et de la mémoire :

- **chaque étape** (détection, tracking, reconnaissance, orientation,
  encodage, image complète, latence de bout en bout) : médiane, p95 et
  maximum sur toute la minute écoulée, pas seulement la fenêtre glissante ;
- **les images perdues** : jetées faute de place dans la file, sautées par
  une source fichier en `realtime` ;
- **le processus** : CPU (% d'un cœur), threads, descripteurs ouverts ;
- **la VRAM de VisionCam** (`wsl_vram_mb`), lue dans les compteurs Windows :
  tout WSL y apparaît sous le processus `vmwp`, et VisionCam est le seul
  programme GPU de WSL pendant un run. Sous WSL2, NVML ne donne pas la
  mémoire par processus ;
- **le contexte**, affiché sans être jugé (`ctx_*`) : utilisation, mémoire,
  température, fréquence et bridage du GPU entier (NVML) ; utilisation GPU
  de WSL et des applications Windows, et la plus gourmande ; CPU et RAM de
  la machine (Windows) et de WSL.

Les compteurs Windows sont lus par `powershell.exe` dans un thread à part,
une fois par intervalle (un appel prend ~7 s). Les utilisations GPU de NVML
et de Windows ne se comparent pas entre elles : fenêtres et définitions
différentes. Hors de WSL, les colonnes Windows sont absentes.

### Second run : personnes enrôlées, page ouverte

Même protocole sur `seq_026_camera_3` (7 754 images), avec les 13 identités
CHIRLA enrôlées (cf. « Identité de bout en bout ») et la page Live ouverte
dans Chromium headless pendant tout le run, le 2026-10-02.

| Mesure | Après warm-up | Fin | Dérive sur 7 h 40 | Verdict |
|:-------|--------------:|----:|------------------:|:--------|
| FPS | 27,9 | 29,9 | +2,7 | stable |
| RSS | 2 390,2 Mo | 2 391,0 Mo | +1,0 Mo | stable |
| VRAM du GPU | 1 520 Mo | 1 520 Mo | 0 | stable |
| Latence p50 / p95 (réception → publication) | 42 / 80 ms | 22 / 53 ms | en baisse | stable |
| Structures internes (13), dont identités, votes, sessions ouvertes | — | — | maxima constants | bornées |

Les structures liées aux noms travaillent cette fois (jusqu'à 4 identités, 18
noms confirmés, 3 sessions ouvertes à la fois) et restent bornées.
L'historique de présence passe de 20 à 36 Ko en 8 h. La page reste ouverte
7 h 59 sans erreur JavaScript.

850 719 images traitées, 28,9 i/s en moyenne : les 57 minutes sous 27 i/s
tombent presque toutes entre 13 h et 17 h 30, quand d'autres calculs
(suites de tests avec Chromium) tournaient sur la machine ; il en reste 4
dans les 3 h 30 suivantes, sans autre charge.

Le run a révélé deux défauts, sans effet sur la stabilité : un message de
journal mal formaté (corrigé depuis) et un nom qui saute entre deux pistes
d'une même personne (8 875 corrections d'identité en 8 h, cf. roadmap 1.8).

### Troisième run : 1 h après la roadmap 1.8

Même protocole que le second run (seq_026 en boucle, personnes enrôlées, page
ouverte), pendant 1 h, avec le seuil NMS à 0,6, la suppression des boîtes
emboîtées et l'hystérésis de l'anti-clonage, le 2026-10-04.

| Mesure | Second run (avant 1.8) | Troisième run (après 1.8) |
|:-------|---:|---:|
| Corrections d'identité par passe de la séquence | ~79 | **~11** |
| Reconnaissance, p95 | — (32-38 ms sur un essai de 5 min) | **22-26 ms** |
| Image complète, p95 | — (40-45 ms sur un essai de 5 min) | **34-36 ms** |

Verdict stable sur toutes les mesures jugées, VRAM de VisionCam constante à
521 Mo, page ouverte 1 h sans erreur JavaScript.

Le FPS de ce run (27,3 i/s en moyenne, 9 507 images jetées) a d'abord été
attribué à un jeu qui tournait sous Windows. C'était faux : un run sur machine
libre a montré les mêmes blocages (cf. ci-dessous).

### Blocages des écritures SQLite (trouvés et corrigés le 2026-10-04)

Sur les runs d'une heure, 35 à 42 minutes sur 55 contenaient une image de
plus de 300 ms, jusqu'à 14 s, alors que toutes les étapes chronométrées
restaient sous 130 ms. Une fois le reste de la boucle chronométré, les
blocages sont apparus dans l'historique de présence (jusqu'à 4,2 s) et le
journal d'événements (jusqu'à 5,3 s) : une connexion SQLite et un fsync par
écriture, dans la boucle vidéo. Isolées, 1 586 écritures ne dépassent pas
38 ms : le fsync n'attend que quand d'autres programmes écrivent sur le même
système de fichiers (ici, Chromium avec la page ouverte), ce qui sous ext4
l'oblige à attendre leurs écritures. Même sans blocage, le journal
d'événements coûtait 24 ms au p95 par image concernée.

Correctif (`core/db_writer.py`) : un thread unique écrit `presence.db`, avec
une connexion gardée ouverte, en WAL et `synchronous=NORMAL` ; la boucle
dépose ses écritures et repart. 30 min, page ouverte, même séquence :

| Mesure | Avant | Après |
|:-------|---:|---:|
| FPS moyen | 26,4 | **30,0** |
| Images jetées | 5 044 | **18** |
| Minutes avec une image > 300 ms | 16 / 27 | **0** |
| Pire image | 8 818 ms | **109 ms** |
| Historique / journal, pire temps dans la boucle | 4 182 / 5 324 ms | **0 / 0 ms** |

Les runs d'endurance faits entre l'ajout du journal d'événements (roadmap 2.1)
et ce correctif ont un FPS tiré vers le bas par ces blocages ; leurs mesures
de mémoire et de structures internes restent valables.

## Plusieurs caméras (roadmap 2.2)

N vidéos CHIRLA rejouées en même temps (`CAMERAS`, `realtime`, en boucle,
30 i/s chacune) : `seq_025` caméras 2, 3 et 5 (filmées en même temps), plus
`seq_026_camera_3` pour la quatrième. 5 min par palier, la première minute
écartée ; i/s par caméra lus sur `/api/diagnostics` toutes les 10 s, le reste
dans le journal d'endurance (une ligne toutes les 30 s, médiane des lignes).
Configuration déployée (YOLO, embedder et InsightFace en TensorRT), aucune
page ouverte, RTX 4060 sous WSL2, le 2026-10-07.

| Caméras | i/s par caméra (min–max) | i/s au total | Image p50 / p95 | Latence p50 / p95 | Détection p50 | Reconnaissance p50 | Images jetées | VRAM (`wsl_vram_mb`) | CPU du processus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,0 (30,0–30,0) | 30,0 | 13,2 / 31,4 ms | 13,4 / 38,8 ms | 9,6 ms | 10,4 ms | 7 | 522 Mo | 50 % |
| 2 | 29,4 (29,3–29,4) | 58,7 | 15,1 / 41,4 ms | 16,4 / 59,7 ms | 9,7 ms | 9,6 ms | 266 | 835 Mo | 87 % |
| 3 | 24,3 (23,6–25,0) | 72,9 | 33,6 / 74,5 ms | 74,1 / 124,4 ms | 21,4 ms | 21,8 ms | 3 871 | 1 071 Mo | 133 % |
| 4 | 17,9 (14,8–19,4) | 71,4 | 50,2 / 111,3 ms | 97,8 / 163,1 ms | 35,0 ms | 38,5 ms | 11 860 | 1 251 Mo | 141 % |

Images jetées : faute de place dans la file d'une caméra, sur les 4 min
mesurées, toutes caméras confondues. CPU du processus : en % d'un cœur.

- **Deux caméras tiennent la cadence** (29,4 i/s chacune). Au-delà, le débit
  total plafonne vers 72 i/s : chaque caméra ajoutée ralentit les autres.
- **Les étapes GPU ralentissent avec le nombre de caméras** : la détection
  passe de 9,6 à 35 ms par image, la reconnaissance de 10 à 38 ms, alors que
  chaque image demande le même travail. Une ressource partagée sature : le
  GPU (l'utilisation de WSL vue par Windows monte à 98 % à 4 caméras), ou le
  GIL entre les threads de traitement. Cette mesure ne les départage pas.
- **Chaque caméra coûte ~240 Mo de VRAM** : chaque `FaceBodyTracker` charge
  son propre moteur YOLO et son embedder d'apparence.

### GPU ou GIL : 4 caméras en 2 processus

Mêmes 4 vidéos, même protocole, mais réparties sur deux instances de
l'application (2 caméras chacune, un GIL chacune) qui partagent le GPU :

| | 1 processus, 4 caméras | 2 processus × 2 caméras |
|:--|---:|---:|
| i/s par caméra | 14,8–19,4 | 29,9–30,0 |
| i/s au total | 71,4 | **119,9** |
| Image p50 / p95 | 50,2 / 111,3 ms | 13,5–16,3 / 35 ms |
| Latence p50 / p95 | 97,8 / 163,1 ms | 13,9–16,9 / 38–39 ms |
| Détection p50 | 35,0 ms | 7,9–10,8 ms |
| Images jetées | 11 860 | 2 |
| Utilisation GPU de WSL (Windows) | 98 % | 83 % |
| VRAM de WSL (`wsl_vram_mb`) | 1 251 Mo | 1 483 Mo (les deux) |

Le GPU suit les 4 caméras à pleine cadence dès qu'elles ne partagent plus un
GIL : **le plafond d'un processus vient du GIL**, pas du GPU. Les étapes
« GPU » ralentissaient parce que leur partie Python (préparation, lecture des
sorties) attendait le GIL. Le second contexte CUDA coûte ~230 Mo de VRAM.

### Tracker Rust à 4 caméras

Même protocole, un seul processus, `TRACKER_BACKEND=rust` (deepsort-rs) au
lieu du tracker Python :

| 4 caméras, 1 processus | Python | Rust |
|:--|---:|---:|
| i/s par caméra | 14,8–19,4 | 22,6–24,8 |
| i/s au total | 71,4 | **96,4** |
| Tracking p50 | 10,7 ms | 6,4 ms |
| Latence p50 / p95 | 97,8 / 163,1 ms | 83,6 / 131,4 ms |
| Images jetées | 11 860 | 5 744 |
| CPU du processus | 141 % | 214 % |

Moins de temps sous le GIL : +35 % de débit, mais toujours sous les 120 i/s
de deux processus. Le reste du pipeline Python garde le plafond.

Ce gain dépend du nombre de personnes par image. Sur MOT17-04 (une foule,
détections publiques, `tools.eval_mot`), les deux backends donnent les mêmes
pistes (MOTA 73,6 %, IDF1 72,3 %, 102 changements d'identité) mais le Rust
prend 138–145 ms par image contre 40–42 ms (deux runs, application
arrêtée). L'association seule, sur des personnes synthétiques (300 images,
budget 100) :

| Personnes par image | Python | Rust |
|---:|---:|---:|
| 3 | 0,69 ms | 0,26 ms |
| 10 | 2,14 ms | 2,65 ms |
| 40 | 9,74 ms | 43,12 ms |

Le Rust gagne sur les scènes de CHIRLA (quelques personnes) et perd dès une
dizaine : son coût croît bien plus vite que celui de deep_sort_realtime,
dont les distances passent par numpy.

### Processus de caméras (`CAMERA_WORKERS=process`)

Même protocole, application complète, caméras réparties en processus de 2
(`CAMERAS_PER_WORKER=2`, core/camera_worker.py), tracker Python, le
2026-10-07 :

| | 2 caméras, threads | 2 caméras, 1 processus | 4 caméras, threads | 4 caméras, 2 processus |
|:--|---:|---:|---:|---:|
| i/s par caméra | 29,4 | 30,1 | 14,8–19,4 | **29,9–30,2** |
| i/s au total | 58,7 | 60,2 | 71,4 | **120,0** |
| Latence p50 / p95 | 16,4 / 59,7 ms | 16,2 / 39,1 ms | 97,8 / 163,1 ms | 19,0 / 46,1 ms |
| Détection p50 | 9,7 ms | 10,8 ms | 35,0 ms | 11,8 ms |
| Images jetées | 266 | 0 | 11 860 | 1 |
| VRAM (`wsl_vram_mb`) | 835 Mo | 857 Mo | 1 251 Mo | 1 598 Mo |

En mode processus, l'image complète et la latence se mesurent dans
l'application, transfert depuis le processus compris. RAM : 2,0 Go pour
l'application et ~2,6 Go par processus de caméras ; à 4 caméras, WSL en
utilise 11,5 Go sur 16 (4,4 Go restés libres au pire). Chaque processus met
10 à 45 s à atteindre sa cadence au démarrage.

### deepsort-rs 8484623 : distances par sgemm, GIL relâché

Après `perf(association)` dans deepsort-rs (échantillons `f32` contigus,
distances par produit matriciel, GIL relâché pendant `update`), le défaut
mesuré plus haut disparaît. Association seule, mêmes personnes synthétiques
(deux runs) :

| Personnes par image | Python | Rust 6f12a5d | Rust 8484623 |
|---:|---:|---:|---:|
| 3 | 0,68–0,73 ms | 0,26 ms | **0,13 ms** |
| 10 | 1,94 ms | 2,65 ms | **0,36 ms** |
| 40 | 9,2–9,4 ms | 43,12 ms | **2,84–2,93 ms** |

MOT17-04 : pistes toujours identiques au Python (MOTA 73,6 %, IDF1 72,3 %,
102 IDs, FP 3 204, FN 9 247) ; 31–32 ms par image, embedder compris, contre
42–45 ms pour le Python et 138–145 ms avant.

À 4 caméras dans un seul processus (CHIRLA), rien ne change : 95,3 i/s au
total contre 96,4, tracking p50 6,4 ms dans les deux cas. Avec 3 à 6
personnes par image, l'association ne prenait déjà que ~0,2 ms : le GIL
qu'elle libère maintenant ne pesait rien, le plafond vient du reste du
pipeline. Le gain vaut pour les scènes chargées.
