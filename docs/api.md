# API

Routes HTTP de l'application. Si un mot de passe est configuré, toutes
demandent une session (cf. [README](../README.md#accès-protégé)).

| Méthode | Chemin | Description |
|:--------|:-------|:------------|
| `GET` | `/` | Interface web (stream + liste de présence) |
| `GET` `POST` | `/login` | Connexion, si un mot de passe est configuré |
| `POST` | `/logout` | Déconnexion |
| `GET` | `/video` | Flux MJPEG `multipart/x-mixed-replace` |
| `GET` | `/status` | `{ currently_present[], fps, total_known, tracks[], frame_size }` — `tracks` : boîtes des personnes visibles |
| `POST` | `/api/capture` | Enrôle la personne cliquée (`{name, track_id, label?}`), label `Face`/`ProfilG`/`ProfilD` en mode guidé |
| `POST` | `/rebuild` | Reconstruction base embeddings depuis `known_faces/` (409 si déjà en cours) |
| `POST` | `/bench/pose` | Compare les deux sources d'orientation sur le flux |
| `GET` | `/api/people` | Personnes connues : photos, labels multitemplate, miniature |
| `GET` | `/api/people/<nom>/thumbnail` | Miniature de la première photo |
| `POST` | `/api/people/<nom>/rename` | Renommer (`{"new_name": …}`) : dossiers et base ensemble |
| `DELETE` | `/api/people/<nom>` | Supprimer photos et entrées (droit à l'effacement) |
| `POST` | `/api/people/<nom>/photos` | Ajouter des photos ; l'embedding est recalculé sur toutes |
| `GET` | `/api/history` | Sessions de présence (`name`, `from`, `to` en AAAA-MM-JJ, `limit`), recalculées du journal d'événements |
| `GET` | `/api/history.csv` | Mêmes filtres, export CSV |
| `GET` | `/api/events` | Server-Sent Events : état toutes les 0,5 s, arrivées, départs, inconnus, enrôlements |
| `GET` | `/api/diagnostics` | FPS, temps par étape (médiane/p95), GPU, modèles et réglages actifs |
