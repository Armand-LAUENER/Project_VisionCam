# Roadmap — VisionCam Tracker en Rust

Objectif : réécrire le module d'association/tracking (DeepSORT) de VisionCam en Rust, l'exposer via PyO3, et démontrer un gain de perf mesuré sur le goulot d'étranglement réel du pipeline.

**Portée explicite** : uniquement le module d'association (Hungarian algorithm + filtre de Kalman + cycle de vie des tracks). L'inférence YOLOv8/InsightFace reste en Python/ONNX — hors scope.

---

## Phase 0 — Setup (0.5 jour)

**Objectif** : environnement Rust prêt, sans complexité inutile.

- Installer Rust (`rustup`), `cargo`
- Installer `maturin` dans ton venv WSL2 existant (`pip install maturin --break-system-packages`)
- Créer le crate : `maturin new visioncam-tracker --bindings pyo3`
- Vérifier le build minimal : fonction Rust "hello world" appelable depuis Python

**Critère de fin** : `import visioncam_tracker` fonctionne dans un script Python, retourne une valeur factice.

**Risque** : aucun à ce stade. Si le build maturin échoue, c'est presque toujours un problème de version Python/toolchain — pas la peine de chercher plus loin.

---

## Phase 1 — Profiling & baseline (1 jour, bloquant)

**Objectif** : confirmer que l'association de tracks est bien le goulot d'étranglement avant d'écrire une ligne de Rust utile.

- Profiler le pipeline VisionCam actuel avec `py-spy` (profiling non-intrusif, préférable à cProfile sur du code avec appels GPU) sur 2-3 vidéos de test représentatives (peu de personnes / beaucoup de personnes, résolutions différentes)
- Isoler le temps passé dans : inférence YOLO, extraction embeddings InsightFace, association DeepSORT
- Noter les chiffres précis (ms/frame par étape) — c'est ta baseline de comparaison finale

**Critère de fin** : un tableau chiffré `étape → % du temps de frame`, sur au moins 2 scénarios différents.

**Point de décision** : si l'association représente <10% du temps de frame, le projet garde sa valeur pédagogique mais perd son argument "optimisation mesurée" en entretien — sois honnête là-dessus si c'est le cas plutôt que d'enjoliver le chiffre après coup.

---

## Phase 2 — Rust ciblé (3-5 jours, en parallèle de la phase 3)

**Objectif** : apprendre uniquement ce dont tu as besoin, pas un cours complet.

Ne pas suivre un cours Rust généraliste de A à Z avant de coder — ça retarde le projet sans bénéfice proportionnel. À la place, cible :

- Ownership/borrowing sur des collections mutables (`Vec<T>`, itération + mutation) — c'est ton vrai point de friction identifié plus tôt
- `struct` + `impl`, traits de base
- Gestion d'erreurs (`Result`, `?`) — pas besoin d'aller plus loin que ça pour ce projet
- Lecture rapide de la doc `nalgebra` (algèbre linéaire) et `pyo3` (guide officiel, section "calling Rust from Python")

**Critère de fin** : tu es capable d'écrire une struct avec état mutable, de la stocker dans un `Vec`, et de faire une passe de suppression conditionnelle (`retain`) sans que le compilateur te bloque plus de 10 minutes sur un problème d'ownership.

**Risque réel** : sous-estimer ce temps. Si tu n'as jamais touché Rust, 3-5 jours est optimiste — applique le principe de Hofstadter, même en le sachant. Ne bloque pas la phase 3 dessus : les deux avancent en parallèle, tu apprends en codant.

---

## Phase 3 — Implémentation core Rust (5-8 jours)

**Objectif** : le moteur de tracking fonctionne en Rust pur, testé indépendamment de Python.

Ordre de développement (chaque étape doit être testable isolément avant de passer à la suivante) :

1. **`kalman.rs`** — filtre de Kalman : prédiction + update de position/vélocité. Teste avec des cas simples à la main (une trajectoire connue, vérifier que la prédiction converge).
2. **`assignment.rs`** — Hungarian algorithm sur une matrice de coût (distance IoU + distance embeddings). Teste avec des matrices de coût connues où tu peux calculer la solution optimale à la main.
3. **`track.rs`** — struct `Track` (id, état Kalman, historique d'embeddings, âge, statut confirmé/tentative).
4. **`manager.rs`** — cycle de vie complet : matching détections↔tracks existants, création de nouveaux tracks, suppression des tracks expirés (`max_age`), confirmation (`min_hits`).

**Critère de fin** : tests unitaires Rust (`cargo test`) qui couvrent au minimum : un cycle complet detection→track confirmé, un track qui expire après `max_age` frames sans détection, un cas d'occlusion simple (track qui disparaît puis réapparaît avant expiration).

**Risque identifié à l'avance** : le cycle de vie des tracks (ajout/suppression pendant itération) est le point exact où le borrow checker va te bloquer. Prévoir du temps buffer ici spécifiquement, pas seulement sur l'apprentissage général de la phase 2.

---

## Phase 4 — Tests de non-régression contre l'implémentation Python (2-3 jours, bloquant avant intégration)

**Objectif** : garantir que le module Rust produit les **mêmes résultats de tracking** que l'implémentation Python actuelle — sans ça, un gain de perf ne veut rien dire si le comportement a changé.

- Faire tourner les deux implémentations (Python DeepSORT actuel + Rust) sur les mêmes vidéos de test de la phase 1
- Comparer : nombre de tracks créés, IDs assignés dans le même ordre (ou équivalents), pas de divergence de trajectoire au-delà d'une tolérance définie
- Documenter les écarts s'il y en a, et justifier s'ils sont acceptables (ex: ordre de traitement différent qui change l'assignation d'ID sans changer le comportement de fond)

**Critère de fin** : un script de comparaison qui donne un verdict pass/fail objectif, pas une inspection visuelle approximative.

**Ne saute pas cette phase.** C'est la partie la moins excitante du projet et donc la plus tentante à bâcler — c'est pourtant elle qui rend le résultat final crédible.

---

## Phase 5 — Bindings PyO3 & packaging (2 jours)

**Objectif** : le module Rust s'utilise depuis Python comme un remplacement direct de l'appel DeepSORT actuel.

- Exposer `Tracker::new()` et `Tracker::update(detections, embeddings) -> Vec<TrackResult>` via PyO3
- Gérer le transfert de données NumPy↔Rust sans copie inutile (crate `numpy`)
- Builder le wheel avec `maturin build --release` (le mode `--release` est non négociable pour le benchmark final — un build debug fausserait complètement la comparaison de perf)

**Critère de fin** : `pip install` du wheel généré localement, appel depuis un script Python isolé qui reproduit l'API de l'ancien tracker DeepSORT.

---

## Phase 6 — Intégration dans le pipeline VisionCam (1-2 jours)

**Objectif** : remplacer l'appel DeepSORT existant par le module Rust dans le pipeline réel, pas dans un script isolé.

- Remplacer l'import et l'appel dans le code VisionCam
- Faire tourner le pipeline complet de bout en bout sur les vidéos de test
- Vérifier que rien d'autre en aval (affichage, logging, export des résultats) ne casse à cause d'un format de sortie légèrement différent

**Critère de fin** : le pipeline VisionCam tourne intégralement avec le tracker Rust, sortie visuelle équivalente à l'ancienne version.

---

## Phase 7 — Benchmark final & documentation (1-2 jours)

**Objectif** : produire le chiffre concret et le raconter correctement.

- Rejouer le profiling de la phase 1 avec le nouveau module, sur les mêmes vidéos
- Comparer : ms/frame avant/après sur l'étape association, et impact sur le temps de frame total
- Écrire un README du crate `visioncam-tracker` : pourquoi ce module existe, comment le builder, résultats de benchmark
- Mettre à jour le README principal de VisionCam pour mentionner le composant Rust

**Critère de fin du projet** : un chiffre défendable ("réduction de X ms à Y ms par frame sur l'étape d'association, soit Z% du temps de frame total"), un repo propre, une non-régression démontrée.

---

## Ce qui est explicitement HORS SCOPE (à ne pas ajouter avant d'avoir fini le MVP ci-dessus)

- Tracking multi-caméra
- Ré-identification cross-caméra
- API REST autour du tracker
- Optimisations SIMD/parallélisation du Hungarian algorithm (pertinent seulement si le nombre de tracks simultanés devient très grand — vérifie que c'est un besoin réel avant d'investir dessus)
- Généralisation du crate en lib réutilisable hors VisionCam

Si l'un de ces points te démange en cours de route, note-le dans un fichier `IDEAS.md` et reviens-y seulement après la phase 7.

---

## Estimation totale

| Phase | Estimation optimiste | Réaliste (Hofstadter appliqué) |
|---|---|---|
| 0 — Setup | 0.5j | 1j |
| 1 — Profiling | 1j | 1.5j |
| 2 — Rust ciblé | 3j | 5-7j |
| 3 — Implémentation core | 5j | 8-10j |
| 4 — Non-régression | 2j | 3j |
| 5 — Bindings PyO3 | 2j | 3j |
| 6 — Intégration | 1j | 2j |
| 7 — Benchmark & doc | 1j | 2j |
| **Total** | **~15.5j** | **~25-30j** (calendaire, à temps partiel à côté des cours) |

La colonne "réaliste" suppose que tu avances dessus en parallèle de tes cours EFREI/CentraleSupélec, pas à temps plein. Si tu comptes le faire à temps plein sur une période dédiée (vacances par ex.), la fourchette basse est plus atteignable.

## Dépendances et points de blocage à anticiper

- **Phase 3 dépend de Phase 2** mais peut démarrer avant que Phase 2 soit "terminée" — apprentissage et implémentation avancent ensemble
- **Phase 4 est bloquante avant Phase 6** — n'intègre jamais dans le pipeline réel un module non validé par non-régression, tu perdrais du temps à débugger deux problèmes en même temps (bug de tracking + bug d'intégration)
- **Risque externe unique** : aucune dépendance API tierce ou donnée externe non disponible — le projet est entièrement autonome sur tes propres vidéos de test, ce qui est un vrai avantage par rapport à d'autres projets qui dépendraient d'une API externe instable
