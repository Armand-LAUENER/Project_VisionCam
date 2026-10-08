# Roadmap — Implémentations à ajouter à VisionCam

Relevé du 2026-10-01, à partir du dépôt au commit `52700d2` et des discussions
des 28-29 septembre. Complète `docs/improvements-backlog.md` (optimisations
déjà faites) : ici, uniquement ce qui reste à construire.

**Règle pour chaque item** : un problème constaté ou une hypothèse, un critère
de fin mesurable, et un résultat documenté dans le README, y compris s'il est
négatif. Une fonctionnalité sans métrique n'entre pas dans le projet.

**Ordre** : la phase 1 fiabilise ce qui existe. Rien de la phase 2 ne démarre
avant que la phase 1 soit terminée. La phase 3 est optionnelle : une seule
fonctionnalité à la fois.

---

## Phase 1 — Fiabiliser le suivi et la reconnaissance

Estimation : 3 semaines à temps partiel, probablement 4.

### 1.1 Source vidéo (simulation de caméra)

- [x] Source « fichier » dans `_open_camera` : `cv2.VideoCapture` sur une vidéo
  ou un motif d'images.
- [x] Deux modes :
  - `realtime` : cadence d'origine, images sautées si le pipeline est trop lent
    (réaliste, non déterministe) ;
  - `every_frame` : chaque image traitée, sans perte (déterministe, **mode des
    tests**).
- [x] Lecture en boucle (option), pour le test d'endurance.
- [x] Fichier d'horodatages à côté de chaque vidéo enregistrée
  (`record_sequence`), pour rejouer plus tard plusieurs caméras synchronisées.

**Terminé quand** : une séquence annotée passe dans l'application complète en
mode `every_frame`, et deux exécutions donnent exactement les mêmes pistes.

### 1.2 Bug : un changement d'ID donne durablement une identité à un inconnu

Constat : `core/face_body_tracker.py:476` ignore tout résultat « Inconnu ».
Après un échange de pistes entre Alice (enrôlée) et une personne non enrôlée,
l'inconnu garde « Alice » indéfiniment. Conséquences dans `app.py:375` :
l'historique enregistre Alice présente, l'alerte « inconnu » ne part pas.

- [x] Dégrader l'identité après K tentatives successives où un visage net
  (≥ `RECOGNITION_MIN_FACE_PX`) est reconnu « Inconnu ».
- [x] N'inscrire dans l'historique de présence que les identités fraîches
  (`last_face_frame` récent).
- [x] Corriger la phrase du README « corrigé au frame suivant par la
  reconnaissance » (vrai seulement si les deux personnes sont enrôlées).

**Terminé quand** : test de régression `tests/regression/` qui simule l'échange
vers un non-enrôlé et vérifie que le nom tombe et que l'alerte part.

### 1.3 Recalibrer le seuil de reconnaissance sur le pipeline réel

Constat : `config.py:94-102` et `.env.example` annoncent 0 % de mauvais noms
dès 40 px au seuil 0,45 (étude sur image entière). Le tableau `bench_face` du
README, sur les crops de tête avec SCRFD 320, donne 18 mauvais noms sur 607
(~3 %) au même seuil.

- [x] Relancer `recognition_threshold_study` sur les crops de `head_crop_box`,
  SCRFD 320.
- [x] Publier les taux avec intervalle de confiance (0 erreur sur n essais ne
  borne le taux qu'à ~3/n à 95 %).
- [x] Corriger les commentaires de `config.py` et `.env.example`.

**Terminé quand** : le taux de mauvais noms publié correspond à la
configuration déployée.

### 1.4 Séquences webcam annotées (validation sur le cas réel)

Tous les chiffres actuels viennent de proxys (MOT17, CHIRLA, DanceTrack).

- [ ] 3 séquences de 60-90 s : entrées/sorties, croisements, lunettes,
  contre-jour, **une personne non enrôlée**.
- [ ] Annotation avec `annotate_sequence`.
- [ ] Rejouer les réglages retenus **sans les réajuster** (validation, pas
  réglage).

**Terminé quand** : tableau MOTA / IDF1 / changements d'ID sur ces séquences,
comparé aux proxys. Limite à écrire : petit échantillon, vérification de
cohérence et non taux d'erreur précis.

### 1.5 Métrique d'identité de bout en bout

Aujourd'hui, le tracking (MOTA/IDF1) et la reconnaissance (par visage) sont
mesurés séparément, jamais le résultat affiché à l'utilisateur.

- [ ] Sur les séquences de 1.4 rejouées via 1.1 : % du temps où chaque piste
  affiche le bon nom, un mauvais nom, « Inconnu ».
- [x] Mesurer aussi la latence bout en bout (image reçue → nom affiché), p50/p95.

**Terminé quand** : un chiffre unique « nom correct X %, mauvais nom Y % » en
tête du README.

### 1.8 Pistes dédoublées : le nom saute entre deux pistes d'une même personne

Constat (endurance, 2026-10-02) : sur une personne assise, DeepSORT garde une
piste en roue libre pendant qu'il en crée une nouvelle, puis réassocie tour à
tour l'une et l'autre (163 ↔ 216 : 39 bascules sur une séquence). À chaque
bascule, l'anti-clonage déplace le nom : 8 875 corrections en 8 h. Sur cette
personne : 569 images « Inconnu » et 145 avec un autre nom, sur 7 542.

- [x] Mesurer avant/après avec `tools/eval_identity.py` (rejeu `every_frame`) :
  MOTA, IDF1 et changements d'ID des pistes affichées.
- [x] Cause 1 : deux boîtes YOLO sur la même personne (IoU ~0,63, sous le seuil
  NMS de 0,7). `YOLO_NMS_IOU=0.6` : corrections 86 → 68 sur seq_026, sans
  perte sur DanceTrack ni MOT17 (cf. docs/tracking.md).
- [x] Cause 2 : boîte emboîtée (haut du corps dans le corps entier, IoU ~0,6).
  Écartée si contenue à 90 % dans une boîte plus sûre : corrections 68 → 59,
  changements d'ID 124 → 105, DanceTrack inchangé.
- [x] Cause 3 : le nom passe à une autre personne assise à côté, dont le
  visage est par moments reconnu sous ce nom ; l'anti-clonage lui donne le nom
  puis le rend. Hystérésis de l'anti-clonage : corrections 59 → 16, mauvais
  noms des enrôlés 1,1 → 0,7 %.

Résultat (2026-10-04) : sur seq_026, corrections d'identité 86 → 16,
changements d'ID 137 → 105, « Inconnu » sur la personne assise 7,3 → 3,7 %,
MOTA inchangé partout, IDF1 31,7 → 34,0 %.
- [ ] Piste : quand le nom passe d'une piste en roue libre à une piste visible
  qui recouvre la même personne, fusionner plutôt que basculer ; ou réduire
  `max_age` pour les pistes sans détection.

**Terminé quand** : bascules et images « Inconnu » sur cette personne en
baisse mesurée, sans perte d'IDF1.

### 1.6 Test d'endurance

- [x] 8 h en continu (vidéo en boucle via 1.1), journal toutes les minutes :
  RSS, VRAM, FPS, nombre de pistes, taille des dictionnaires internes.

**Terminé quand** : mémoire et FPS stables (pas de pente) sur 8 h.

### 1.7 Nettoyage

- [x] `_identify` : matrice d'embeddings pré-empilée, reconstruite à
  l'enrôlement (aujourd'hui `np.stack` à chaque appel, alors que le backlog
  coche cette case).
- [x] README : résumé des résultats clés en tête (un écran), détail déplacé dans
  `docs/`.

---

## Phase 2 — Multi-caméras

Estimation : au moins 3 à 5 semaines. Ne sera pas prête pour les entretiens de
novembre.

### 2.1 Journal d'événements (refonte du stockage)

- [x] Table d'événements bruts en ajout seul dans SQLite : `camera_id`,
  `track_id`, horodatage, type (apparition, entrée/sortie de zone,
  identification, fin de piste), zone.
- [x] Sessions de présence recalculées à partir des événements (vue dérivée,
  plus donnée source). `/api/history` et l'export CSV les recalculent à chaque
  requête (core/history.py) ; la table `sessions` n'est plus qu'un cache,
  lue seulement pour l'historique antérieur au premier événement. Validé :
  sessions identiques au calcul au fil de l'eau (simulations et rejeu CHIRLA).
- [x] Horloge unique monotone à la réception (ForwardClock). La mesure du
  décalage des caméras IP passe en 2.2 : elle demande plusieurs caméras.
- [x] RGPD : rétention courte pour les événements des pistes inconnues,
  effacement d'une personne étendu à ses événements. (Choix : même rétention
  que l'historique, 30 jours.)

SQLite suffit à cette échelle : pas de Kafka, Postgres ni Airflow.

### 2.2 N caméras, un tracker par caméra

- [x] Un thread de capture et un `FaceBodyTracker` par caméra, interface
  multi-flux. `CAMERAS="id=source;…"` (core/cameras.py), un thread de capture
  et un de traitement par caméra, reconnaissance et journaux partagés ;
  `?camera=` sur /video, /status, /api/events, /api/capture, /bench/pose ;
  sélecteur sur la page Live. Deux caméras tiennent la cadence (29,4 i/s
  chacune), le total plafonne vers 72 i/s au-delà (docs/performance.md).
- [ ] Mesurer le décalage d'horloge des caméras IP (reporté de 2.1).
- [x] Partage du GPU. Mesure : le GPU n'est pas la limite, le GIL l'est (4
  caméras en 2 processus : 120 i/s). Pas de lots d'images ni de cadence
  réduite : `CAMERA_WORKERS=process`, un processus par `CAMERAS_PER_WORKER`
  caméras (core/camera_worker.py) ; 4 caméras à 30 i/s chacune
  (docs/performance.md).
- [x] Processus de caméras, suite : image brute demandée au processus pour
  /api/capture et /bench/pose (commande "frame"), base de visages relue par
  les processus après enrôlement, renommage, suppression ou reconstruction
  (commande "reload", `FaceRecognizer.reload_cache`), processus mort relancé
  (délai de 5 s doublé jusqu'à 60 s ; relancé en 6 s après un `kill -9`).

### 2.3 Identités globales

- [x] Personnes enrôlées : nom global via la reconnaissance faciale. Vérifié :
  aucun conflit ni clone entre caméras ; plusieurs caméras nomment plus de
  moments qu'une seule (7,9 % contre 4,4 % sur `seq_025`). Champs de vision
  qui se recouvrent : l'anti-clonage reste propre à chaque caméra.
- [x] Évaluation inter-caméras sur CHIRLA : `tools/eval_global_identity.py`
  (docs/performance.md). Passer un nom d'une caméra à l'autre rapporterait au
  mieux 3,3 % des suivis en « Inconnu » : à garder en tête pour prioriser
  2.4 et 2.5.

### 2.4 Topologie des caméras

- [ ] Fichier YAML : zones d'entrée/sortie par caméra, liens entre zones,
  fenêtres de transit, zones « vers l'extérieur ».
- [ ] Association inter-caméras limitée aux candidats plausibles ; fenêtre de
  temps en distribution (score réduit), pas en contrainte stricte.
- [ ] Si des champs de vision se chevauchent : homographie vers un plan au sol
  commun.
- [ ] Vérifier si CHIRLA fournit la disposition des caméras ; sinon la déduire
  des annotations.

### 2.5 Couleur des vêtements

- [ ] Descripteur couleur haut / bas du corps à partir des keypoints épaules et
  hanches (pas de modèle supplémentaire).
- [ ] Usage : ré-identification des inconnus entre caméras, et après une sortie
  plus longue que `max_age`.

**Terminé quand** : réduction mesurée des changements d'ID / des erreurs
d'association inter-caméras, sur le protocole réglage/validation existant.

### 2.6 Apprentissage des temps de transit

- [ ] Transits observés à partir des personnes enrôlées (étiquettes gratuites) ;
  distribution par lien, conservée en agrégat.
- Référence : travaux de Javed, Shafique et Shah sur le suivi à travers des
  caméras aux champs disjoints ; pistes multi-caméras de l'AI City Challenge.

### 2.7 Banc de ressources et test de charge

- [x] Instrumentation par étape (`psutil` + `pynvml`, warm-up exclu) : journal
  d'endurance (core/endurance.py, core/system_probe.py), étape « decode »
  ajoutée. Pas de `torch.cuda.synchronize` : chaque étape rapatrie ses
  sorties sur le CPU avant la fin du chronomètre.
- [x] Balayage : taille de modèle, FP32/FP16, cadence de reconnaissance,
  nombre de visages (docs/performance.md) : `s` en TensorRT FP16 reste le
  meilleur compromis, la cadence de reconnaissance trace la frontière
  qualité/coût. INT8 et résolution non mesurés (calibrage, moteurs à
  reconstruire).
- [x] 2, 4, 8, 16 flux simulés : caméras tenues à 5 / 10 / 30 i/s, goulot
  (décodage, GPU, CPU), latence p95. `tools/bench_load.py`, `CAMERA_MAX_FPS` ;
  goulot : GIL par processus, borné par la RAM (docs/performance.md).
- [ ] Adaptation en marche au nombre de caméras actives : cadence par caméra
  et cadence de reconnaissance recalculées quand une caméra se connecte ou se
  déconnecte, à partir de la capacité mesurée (images par seconde par
  processus, GPU) ; réglages manuels prioritaires. Pas de « config minimale »
  sur du matériel non testé : des besoins mesurés (VRAM pic, ms GPU par image).
- [x] Préciser que les mesures viennent de WSL2.

---

## Phase 3 — Options (une à la fois, chacune avec sa métrique)

Classées de la plus à la moins cohérente avec VisionCam.

- [ ] **Anti-spoofing / liveness** : le trou fonctionnel le plus visible.
  Métriques APCER/BPCER ; jeux CelebA-Spoof, OULU-NPU.
- [ ] **Qualité du visage** (SER-FIQ, CR-FIQA, MagFace) : ne calculer
  l'embedding que sur les images exploitables. Mesure : taux de mauvais noms
  avant/après.
- [ ] **Objet abandonné** : objet déposé près d'une personne identifiée, qui
  reste après son départ, alerte après X s. Nécessite un deuxième modèle
  (`yolov8s.pt`, 80 classes) : YOLOv8-Pose ne détecte que les personnes.
  Mesure : fausses alertes par heure, latence ajoutée.
- [ ] **Actions à partir du squelette** : classes à définir avant tout (chute,
  immobilité, course…). Keypoints déjà disponibles ; ST-GCN ou classifieur sur
  fenêtres temporelles. Demande des données annotées : le plus gros chantier.
- [ ] **Masquage du fond dans les crops de ré-identification** (expérience) :
  segmentation d'instance (`yolov8s-seg`) ou enveloppe du squelette. Hypothèse :
  moins de changements d'ID sans perte d'IDF1. Résultat négatif documenté s'il
  n'y a pas de gain.
- [ ] **Floutage des personnes non enrôlées** dans le flux : flou de la boîte de
  tête, pas besoin de segmentation.
- [ ] **Lettres et chiffres (OCR)** : à cadrer, voir questions ouvertes.

---

## Document d'architecture — passage à 1000 caméras

Pas une implémentation : un document de conception appuyé sur les mesures de 2.7.

- [ ] Hypothèses explicites et extrapolation chiffrée (nombre et type de GPU).
- [ ] Changements : RTSP/H.264 et décodage matériel (NVDEC), workers
  d'inférence (Triton), Kubernetes, tracker par caméra sans état partagé et
  service central d'association, Kafka + base analytique (ClickHouse), index
  vectoriel (FAISS / Milvus) si des milliers de personnes enrôlées, monitoring
  par caméra.
- [ ] Topologie apprise obligatoire à cette échelle.
- [ ] Cadre légal : AI Act art. 5(1)(h) (identification biométrique à distance
  en temps réel dans les espaces publics), système à haut risque, RGPD art. 9.

---

## Écarté (et pourquoi)

| Idée | Raison |
|:-----|:-------|
| Reconstruction faciale 3D | N'améliore ni la reconnaissance (ArcFace déjà robuste à la pose) ni l'anti-spoofing (une photo donne un visage 3D plausible). Pose 3D déjà disponible via MediaPipe. |
| Fiches d'attributs (sexe, taille, âge) liées à l'identité | Profilage biométrique (RGPD art. 9, AI Act annexe III) ; taille impossible sans calibration ; biais documentés (*Gender Shades*, 2018). Seule variante acceptable : statistiques agrégées anonymes, avec évaluation des biais. |
| Détection d'objets générique (boîtes « chaise », « tasse ») | Démo sans lien avec le suivi ; remplacée par « objet abandonné ». |
| Segmentation sémantique | Ne distingue pas les individus ; remplacée par l'expérience de masquage (phase 3). |
| Reconnaissance des émotions / « comportements suspects » | Reconnaissance des émotions au travail interdite (AI Act art. 5(1)(f)) ; classes non définies. |

---

## Questions ouvertes

- **Actions** : quelles classes exactement ?
- **OCR** : lire quoi (badges, écrans, plaques) et pour quel usage ? Les plaques
  d'immatriculation sont des données personnelles.
- **Cible de déploiement** (même hypothétique) : PC avec GPU, CPU seul, Jetson ?
  Détermine les configurations à tester en 2.7.
