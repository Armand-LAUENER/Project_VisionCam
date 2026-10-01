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
