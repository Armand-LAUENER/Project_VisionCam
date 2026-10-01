# Réglage du tracking

Pistes en roue libre, seuils DeepSORT et validation sur trois jeux de
données. Résumé : [README](../README.md#résultats-clés).

## Pistes en roue libre

Quand une personne est occultée ou sort du champ, DeepSORT garde sa piste
jusqu'à `DEEPSORT_MAX_AGE` images, avec une boîte prédite par le filtre de
Kalman. `FaceBodyTracker` garde l'identité de ces pistes en mémoire, mais ne
les affiche pas et ne les soumet pas à la reconnaissance faciale : seules les
pistes appariées à une détection YOLO de l'image courante le sont
(appariement un-pour-un par IoU). Avant, les boîtes fantômes restaient à
l'écran et leur crop visage, qui montrait l'obstacle, partait à InsightFace.

## Seuils DeepSORT

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

## Validation sur trois jeux de données

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
