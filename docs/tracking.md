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

### Seuil NMS de YOLO : pistes dédoublées (roadmap 1.8)

Une personne assise derrière un bureau reçoit parfois deux boîtes YOLO qui se
recouvrent à ~0,63 (corps entier, haut du corps). Au seuil NMS d'ultralytics
(0,7), les deux survivent : une seconde piste naît, puis DeepSORT associe la
boîte tantôt à l'une, tantôt à l'autre, et le nom saute à chaque bascule
(8 875 corrections d'identité en 8 h d'endurance).

Mesuré dans l'application complète, rejeu `every_frame`, pistes affichées
comparées à la vérité terrain (`tools/eval_identity.py`) :

| `YOLO_NMS_IOU` | CHIRLA seq_026 : corrections d'identité / changements d'ID | Personne assise : bon nom / « Inconnu » | DanceTrack (5 séq.) : MOTA / IDF1 | MOT17-04, -09, CHIRLA seq_025 |
|---:|:---|:---|:---|:---|
| 0,7 (avant) | 86 / 137 | 87,3 % / 7,3 % | 84,6 % / 61,2 % | — |
| **0,6** | **68 / 124** | **88,9 % / 5,6 %** | **84,7 % / 61,1 %** | inchangés |
| 0,5 | 59 / 106 | 89,8 % / 4,7 % | 84,6 % / 59,4 % | inchangés |

0,6 est retenu : il réduit les corrections de 21 % sans rien coûter sur
DanceTrack, où des danseurs se croisent de près (le cas où un NMS plus
agressif risquerait de supprimer une vraie personne : le MOTA ne bouge pas).
0,5 gagne un peu plus sur CHIRLA mais perd 1,8 point d'IDF1 sur DanceTrack.
L'IDF1 d'une séquence varie de 3 à 7 points d'un réglage à l'autre : les
écarts moyens de 1 à 2 points sont à la limite du bruit.

Le NMS ne règle qu'une partie du problème : il reste 68 corrections sur
seq_026. Une partie vient de boîtes emboîtées (haut du corps contenu à 93-94 %
dans le corps entier, IoU 0,56-0,64 : sous le seuil NMS). Elles sont écartées
quand elles sont contenues à au moins 90 % dans une boîte plus sûre et la
recouvrent à au moins 0,5 (`NESTED_BOX_MIN_CONTAINED`, `NESTED_BOX_MIN_IOU`) ;
l'IoU minimale garde une petite personne plus loin derrière une autre.

| Variante | seq_026 : corrections / changements d'ID / IDF1 | Personne assise : bon nom / « Inconnu » | DanceTrack : MOTA / IDF1 / changements d'ID |
|:---------|:---|:---|:---|
| NMS 0,6 | 68 / 124 / 41,6 % | 88,9 % / 5,6 % | 84,7 % / 61,1 % / 99 |
| **NMS 0,6 + boîtes emboîtées** | **59 / 105** / 34,0 % | **89,8 % / 4,7 %** | **84,7 % / 61,7 % / 97** |

MOT17-04, -09 et seq_025 : inchangés. Sur seq_026, la suppression retire les
mêmes boîtes qu'un NMS à 0,5, sans toucher aux danseurs qui se recouvrent. Son
IDF1 recule par rapport au NMS 0,6 seul (41,6 → 34,0 %, au-dessus des 31,7 %
d'origine) ; sur cette séquence il ne varie pas de façon régulière, quand les
changements d'ID baissent à chaque étape (137, 124, 105).

Il restait 59 corrections : un nom qui passait à la voisine assise à côté
(son visage reconnu par moments sous ce nom), à une boîte partielle dont le
crop attrapait le visage, ou à une piste dédoublée non emboîtée. Point commun :
le nom était pris à une piste visible qui venait d'être reconnue. L'hystérésis
de l'anti-clonage le lui laisse : une piste visible dont le visage a été
reconnu sous ce nom depuis moins de `IDENTITY_PROTECT_FRAMES` (90 images) le
garde, sauf reconnaissance plus sûre d'au moins `IDENTITY_STEAL_MARGIN`
(0,05). En roue libre ou sans visage récent, elle le perd comme avant.

| Variante | seq_026 : corrections | Personne assise : bon nom / « Inconnu » / mauvais | Enrôlés : bon nom / mauvais nom |
|:---------|---:|:---|:---|
| NMS 0,6 + boîtes emboîtées | 59 | 89,8 % / 4,7 % / 1,8 % | 37,9 % / 1,1 % |
| **+ hystérésis** | **16** | **90,8 % / 3,7 %** / 1,8 % | 37,4 % / **0,7 %** |

Pistes, MOTA et IDF1 inchangés sur les 9 séquences : l'hystérésis ne touche
que les noms. Elle n'est mise à l'épreuve que sur seq_026, seule séquence avec
des personnes enrôlées qui se côtoient.

Bilan sur seq_026, depuis l'origine : corrections d'identité 86 → 16 (−81 %),
changements d'ID 137 → 105, « Inconnu » sur la personne assise 7,3 → 3,7 %,
mauvais noms des enrôlés 1,1 → 0,7 %, MOTA inchangé.

Pour rejouer une séquence dans l'application complète, au lieu de la caméra,
et garder les pistes affichées (deux exécutions en `every_frame` donnent les
mêmes : vérifié sur 600 images de CHIRLA, 1 266 pistes-images identiques) :

```bash
VIDEO_FILE=~/datasets/visioncam/scene-01 VIDEO_MODE=every_frame \
    TRACKS_LOG_PATH=data/tracks.csv uv run app.py
```

Pour valider sur ta propre caméra (roadmap 1.4) : trois séquences de 60 à
90 s — entrées et sorties, croisements, lunettes, contre-jour, et une
personne non enrôlée.

```bash
# 1. Enregistrer et annoter (une fois par scène)
uv run -m tools.record_sequence --seconds 90 --output ~/datasets/visioncam/scene-01
uv run -m tools.annotate_sequence ~/datasets/visioncam/scene-01   # corriger les identités

# 2. Tracking, réglages actuels, sans les réajuster (validation, pas réglage)
uv run -m tools.eval_mot --seq ~/datasets/visioncam/scene-01

# 3. Identité de bout en bout : rejouer dans l'application, puis comparer.
#    --names relie chaque identité annotée au nom enrôlé ; la personne non
#    enrôlée n'y figure pas.
VIDEO_FILE=~/datasets/visioncam/scene-01 VIDEO_MODE=every_frame \
    TRACKS_LOG_PATH=data/tracks-scene-01.csv uv run app.py   # Ctrl+C à « Fin de la vidéo »
uv run -m tools.eval_identity --tracks data/tracks-scene-01.csv \
    --sequence ~/datasets/visioncam/scene-01 --names 1=Armand 2=Alice
```

Pour régler DeepSORT sur ses propres scènes plutôt que valider :

```bash
uv run -m tools.sweep_deepsort --only-updated --tune ~/datasets/visioncam/scene-01 \
    --grid init=3,5 --validate ~/datasets/visioncam/scene-02
```

## DeepSORT, ByteTrack ou BoT-SORT

ByteTrack et BoT-SORT sont aujourd'hui plus courants que DeepSORT, et
ultralytics les fournit déjà (`TRACKER_BACKEND=bytetrack|botsort|botsort-reid`,
`UltralyticsBackend` dans core/tracker_backends.py). Pour comparer les
trackers et non leurs conventions : les pistes perdues sont renvoyées comme
les pistes DeepSORT en roue libre (sinon leur nom serait purgé), YOLO descend
à `BYTETRACK_LOW_THRESH` (0,1) pour le second passage de ByteTrack, seuil
principal et création de piste à 0,5 comme DeepSORT, `track_buffer` = 70
images comme `max_age`, pas de compensation de mouvement (caméra fixe).
`botsort-reid` branche le MobileNetV2 de DeepSORT comme ré-identification.
Le 2026-10-07.

**Tracker seul**, détections publiques (`tools/eval_mot.py`) :

| | MOT17-04 : MOTA / IDF1 / IDs / ms par image | DanceTrack val (5 séq.) : MOTA / IDF1 / IDs |
|:--|:--|:--|
| DeepSORT (Python) | 73,6 % / 72,3 % / 102 / 38,8 | 69,5 % / 56,5 % / 108 |
| ByteTrack | 75,5 % / 75,9 % / **60** / **2,6** | 63,9 % / 40,6 % / 120 |
| BoT-SORT | **75,7 %** / 75,6 % / 64 / 2,8 | 64,6 % / 43,9 % / 118 |

Sur MOT17-04 (foule), ByteTrack et BoT-SORT font mieux sur tout, et 15 fois
plus vite (pas d'embedder d'apparence). Sur DanceTrack, ils perdent 5 points de
MOTA et 13 à 16 d'IDF1 : les seuils de DeepSORT ont été réglés sur ce jeu
(`n_init=5`, cf. plus haut), ceux d'ultralytics non.

**Application complète**, CHIRLA, rejeu `every_frame`, 13 identités enrôlées
(`tools/eval_identity.py`, protocole de docs/performance.md) :

| `seq_026_camera_3` | MOTA | IDF1 | IDs | Bons noms | Mauvais noms | Sans piste |
|:--|---:|---:|---:|---:|---:|---:|
| **DeepSORT** | 67,7 % | 34,0 % | 105 | **39,6 %** | **0,7 %** | 25,4 % |
| ByteTrack | 71,3 % | 32,6 % | 88 | 38,7 % | 1,0 % | 15,2 % |
| BoT-SORT | 71,6 % | 32,4 % | 89 | 38,3 % | 1,0 % | 14,8 % |
| BoT-SORT + ré-identification | 71,8 % | 45,7 % | 91 | 38,2 % | 1,2 % | 14,8 % |
| BoT-SORT sans second passage (`BYTETRACK_LOW_THRESH=0.5`) | 68,7 % | 33,0 % | 87 | 38,1 % | 0,8 % | 24,3 % |

Sur `seq_025_camera_2`, les noms ne bougent pas (visages trop petits) ; le
suivi s'améliore (MOTA 84,4 → 86,1–87,9 %, aucun mauvais nom).

**DeepSORT reste le défaut.** Les trackers d'ultralytics suivent mieux
(MOTA +4, 15 % de changements d'identité en moins, 10 points de couverture
en plus), mais ce qu'ils couvrent en plus, ce sont des personnes de dos ou
masquées, qui restent « Inconnu » : les bons noms ne montent pas, et les
mauvais noms montent de 0,7 à 1,0–1,2 %. Une mauvaise identité coûte plus
cher qu'un « Inconnu ». Cause : les détections faibles du second passage
prolongent une piste à travers une occultation, parfois jusque sur la
personne voisine ; sans elles, BoT-SORT revient à 0,8 % mais ne nomme pas
mieux. La ré-identification de BoT-SORT gonfle l'IDF1 sans aider les noms : elle
ne départage que des boîtes qui se recouvrent déjà.

Un run par configuration, deux séquences d'un même bureau : les écarts de
bons noms (±1,5 point) sont dans le bruit, celui des mauvais noms est plus
net mais indicatif. À refaire sur une scène où les visages restent visibles
mais où le suivi casse (foule, croisements), ou avec des identités globales
(roadmap 2.3).

## Topologie des caméras (roadmap 2.4)

CHIRLA ne fournit pas le plan de ses 7 caméras : `tools/infer_topology.py` le
déduit des annotations (identités communes à toutes les caméras d'une
séquence) et écrit `topologies/chirla.yaml`, agrégats seulement :
recouvrements des champs (caméras 1-2-3 : une même pièce, 78 % entre 2 et 3),
liens avec l'histogramme de leurs temps de transit (6 → 5 : 91 passages,
médiane 1,4 s ; 4 → 5 : 14,3 s), zones d'entrée et de sortie par caméra.

`core/topology.py` en tire la plausibilité qu'une sortie de la caméra A soit
l'entrée sur B Δ secondes plus tard, en distribution et non en fenêtre
stricte : P(B | sortie de A) × densité de Δ sur le lien (noyau de 1 s).
Validation sans fuite (`tools/eval_topology.py`) : topologie apprise sur les 7
séquences de juin-juillet, jugée sur les 3 de décembre ; 68 301 paires
candidates (sortie sur A, entrée sur B entre −5 et +60 s), dont 751 vrais
passages. Le 2026-10-08 :

| Score | AUC | Faux écartés en gardant 90 % des vrais |
|:--|---:|---:|
| Délai seul (plus c'est proche, plus c'est plausible) | 0,925 | 87,2 % |
| Fenêtre p10-p90 plate (premier essai) | 0,862 | — |
| **Vraisemblance du délai selon le lien** | **0,961** | 87,6 % |

Une fenêtre plate fait moins bien que le seul délai : c'est la distribution
qui apporte l'information. Le gain vient surtout du classement des cas
ambigus (1 − AUC divisé par deux), ce qui sert à départager plusieurs
candidats ; au seuil de 90 % de rappel, les deux écartent autant de faux. Les
liens vus une seule fois comptent (0,942 si on ne garde que ceux vus 3 fois).
