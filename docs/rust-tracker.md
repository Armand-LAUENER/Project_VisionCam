# Tracker Rust

Backend d'association optionnel en Rust : installation, retour en arrière
et comparaison mesurée. Résumé : [README](../README.md#résultats-clés).

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

## Installation

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

## Revenir en arrière

Remettre `TRACKER_BACKEND=python` (ou retirer la ligne du `.env`) et
redémarrer. Aucune désinstallation nécessaire, aucun autre fichier à toucher :
c'est le seul point de bascule.

## Résultats mesurés

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
