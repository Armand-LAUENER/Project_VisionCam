# Progress — VisionCam

## État actuel
- Milestone : pipeline complet (YOLOv8-Pose TensorRT → DeepSORT → InsightFace,
  vote 3 reco/piste) ; projet portfolio public, `master` à jour avec `origin`.
- Ce qui marche : reco + tracking temps réel, continuité par le corps sous
  masque, correctif anti-clonage des votes périmés, mesures d'identité
  republiées à cadence fixe. Checks : ruff ✅, 567 tests ✅ (0 GPU requis).
- Ce qui est incomplet : les faux noms sur non-enrôlés à distance sont une
  limite de reconnaissance (faux positif ArcFace), pas un bug — non corrigé
  par choix (on ne touche pas au seuil).

## Roadmap
- [x] Tracker Rust (`deepsort-rs`, `TRACKER_BACKEND=rust`) — terminé
- [x] Continuité du nom sous masque (corps, une caméra)
- [x] Bug « votes périmés après l'anti-clonage » (corrigé + mesuré)
- [x] Republier l'identité de bout en bout à cadence fixe
- [ ] 1.4 roadmap : 3 séquences webcam annotées (entrées/sorties, croisements,
      lunettes) via `annotate_sequence`, puis rejeu des réglages retenus
- [ ] Réduire les faux positifs de reco à distance sans baisser le seuil
      (exiger un visage plus grand, ou plus de voix pour un premier nom) — à mesurer

## Prochaines étapes
1. Décider la suite : soit l'annotation des séquences webcam (roadmap 1.4),
   soit explorer la piste « visage plus grand / plus de voix avant le 1er nom »
   pour les faux positifs à distance (tools/eval_identity.py, cadence fixe).
2. Si reprise de l'identité : réutiliser le protocole cadence fixe
   (`FACE_RECOGNITION_SKIP=2`, `every_frame`) documenté dans docs/performance.md.

## Décisions
- 2026-10-20 — Mesures d'identité toujours à cadence fixe (`FACE_RECOGNITION_SKIP=2`) :
  la cadence adaptative n'est pas reproductible d'un rejeu à l'autre.
- 2026-10-20 — Le faux nom sur non-enrôlés à distance (seq_025, 8,2 %) est un
  faux positif de reconnaissance, pas un bug de suivi ; seuil inchangé.

## Journal
### 2026-10-20 — anti-clonage, mesures d'identité à cadence fixe
- Fait : finalisé `tools/eval_swap_check.py` (test relatif corps après contact)
  + mesure dans docs/tracking.md ; corrigé le bug « votes périmés après
  l'anti-clonage » (`core/face_body_tracker.py`, purge `_vote_buffer` +
  `_unknown_streak`) avec régression `test_stale_votes_after_anticlone.py` ;
  mesuré avant/après sur seq_026_camera_3 (non-enrôlés nommés à tort
  19,7 % → 0,0 %) ; republié l'identité de bout en bout à cadence fixe dans
  docs/performance.md + README ; élucidé le faux nom de seq_025 (faux positif
  ArcFace, piste 8 sur personne 14 nommée id_5 ~6 s, personne 5 absente).
- Notes : commits 9fd2e16, 8b17422, 52c50dc, 71a290f, 049bd68 (poussés).
  Galerie d'éval, journaux et scripts d'analyse dans le scratchpad (hors repo).
