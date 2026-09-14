import glob
import logging
import os
import re
import shutil
import tempfile
import threading
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config

logger = logging.getLogger(__name__)

# Noms de personne et labels de template : ils deviennent des noms de dossier
# et de fichier sous known_faces/. Lettres (accents compris), chiffres, '_',
# '-', espace et point, sans point initial : ni séparateur de chemin, ni '..',
# ni '#' (séparateur nom#label des entrées multitemplate).
_SAFE_NAME = re.compile(r"[\w-][\w\-. ]{0,63}")


def _is_safe_name(value: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(value))


_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# Issues des opérations de gestion des personnes : l'API web en déduit le code HTTP.
OK, INVALID, NOT_FOUND, CONFLICT, NO_FACE = "ok", "invalid", "not_found", "conflict", "no_face"


def _flatten_model_dir(name, root="~/.insightface"):
    """Remonte les .onnx d'un pack de modèles extrait dans un sous-dossier homonyme.

    Certains packs (antelopev2) sont distribués dans un zip contenant déjà un
    dossier à leur nom : l'extraction produit <pack>/<pack>/*.onnx alors
    qu'InsightFace ne cherche les .onnx que directement dans <pack>/.

    Retourne True si un aplatissement a été effectué, False sinon.
    """
    model_dir = os.path.join(os.path.expanduser(root), "models", name)
    nested = os.path.join(model_dir, name)
    if glob.glob(os.path.join(model_dir, "*.onnx")) or not os.path.isdir(nested):
        return False

    logger.warning("Pack de modèles imbriqué détecté, aplatissement : %s", nested)
    for filename in os.listdir(nested):
        shutil.move(os.path.join(nested, filename), os.path.join(model_dir, filename))
    os.rmdir(nested)
    return True


class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", threshold=0.45, cache_path="data/embeddings.npz"):
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.cache_path = cache_path
        self.known_embeddings = []
        self.known_names = []
        self._lock = threading.Lock()   # Protège known_embeddings/known_names (accès multi-thread)
        # Sérialise ce qui modifie la base (enrôlement, renommage, suppression,
        # reconstruction) : un enrôlement pendant un rebuild était écrasé.
        self._write_lock = threading.RLock()

        # ── Charger InsightFace ──
        # allowed_modules : détection + reconnaissance uniquement.
        # Sans cette restriction, buffalo_l charge aussi landmark_2d_106,
        # landmark_3d_68 et genderage — inutiles ici et responsables du pic
        # de latence toutes les FACE_RECOGNITION_SKIP frames.
        logger.info("Chargement InsightFace (%s, det+rec only)...", config.INSIGHTFACE_MODEL)
        try:
            self.app = self._new_face_analysis()
        except AssertionError:
            # Aucun modèle chargé : le pack vient probablement d'être téléchargé
            # et extrait dans un sous-dossier imbriqué. On aplatit puis on réessaie
            # une fois — inutile de retélécharger, les .onnx sont déjà là.
            if not _flatten_model_dir(config.INSIGHTFACE_MODEL):
                raise
            self.app = self._new_face_analysis()
        self.app.prepare(ctx_id=0, det_size=config.INSIGHTFACE_DET_SIZE)
        logger.info("InsightFace chargé")

        # ── Charger ou construire la base ──
        self._load_or_build_database()

    def _new_face_analysis(self):
        """Instancie FaceAnalysis avec la configuration du projet."""
        return FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            allowed_modules=['detection', 'recognition'],
            providers=config.ONNX_PROVIDERS,
        )

    # =========================================================================
    # CHARGEMENT / CONSTRUCTION
    # =========================================================================

    def _load_or_build_database(self):
        """Charge le cache ou reconstruit la base d'embeddings.

        Un cache illisible est traité comme absent : il est entièrement
        reconstructible depuis known_faces/, il n'y a donc aucune raison
        d'empêcher le démarrage pour ça.

        Format .npz lu avec allow_pickle=False : contrairement à pickle, un
        fichier altéré ne peut pas exécuter de code au chargement, il est
        seulement refusé.
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        if not os.path.exists(self.cache_path):
            self._build_database()
            return

        logger.info("Chargement du cache : %s", self.cache_path)
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                names = [str(name) for name in data['names']]
                embeddings = list(data['embeddings'])
            if len(names) != len(embeddings):
                raise ValueError(f"{len(names)} nom(s) pour {len(embeddings)} embedding(s)")
        except (EOFError, ValueError, KeyError, OSError, zipfile.BadZipFile) as e:
            logger.warning(
                "Cache illisible (%s: %s) — reconstruction depuis %s",
                type(e).__name__, e, self.known_faces_dir,
            )
            self._build_database()
            return

        self.known_embeddings = embeddings
        self.known_names = names
        logger.info("%d entrée(s) chargée(s) depuis le cache", len(self.known_names))

    def _build_database(self):
        """Parcourt known_faces/ et calcule l'embedding moyen par personne.

        La nouvelle base est construite à part puis remplace l'ancienne d'un
        seul coup sous _lock : pendant un rebuild, la reconnaissance continue
        de lire l'ancienne base complète au lieu d'une base vide ou partielle.
        """
        logger.info("Construction de la base depuis : %s", self.known_faces_dir)
        known_embeddings = []
        known_names = []

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)
            with self._lock:
                self.known_embeddings, self.known_names = known_embeddings, known_names
            logger.warning("Dossier known_faces vide, aucune personne enregistrée.")
            return

        for person_name in sorted(os.listdir(self.known_faces_dir)):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            images = []
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                img = cv2.imread(os.path.join(person_dir, img_file))
                if img is not None:
                    images.append(img)

            embeddings = self._extract_best_embeddings(images)
            if embeddings:
                avg = np.mean(embeddings, axis=0)
                avg = avg / np.linalg.norm(avg)
                known_embeddings.append(avg)
                known_names.append(person_name)
                logger.debug("OK %s : %d image(s) traitée(s)", person_name, len(embeddings))
            else:
                logger.warning("SKIP %s : aucun embedding valide", person_name)

        with self._lock:
            self.known_embeddings, self.known_names = known_embeddings, known_names
        self._save_cache()
        logger.info("Base construite : %d entrée(s)", len(self.known_names))

    def _save_cache(self):
        """Sauvegarde les embeddings en cache (appeler sous lock si partagé).

        Écriture atomique : sans elle, un arrêt du processus pendant le dump
        laisse un fichier tronqué (voire vide) qui remplace un cache valide.
        os.replace est atomique tant que le temporaire est sur le même
        système de fichiers, d'où le dossier de destination.

        Effet de bord assumé : mkstemp crée en 0600, le cache n'est donc plus
        lisible que par son propriétaire. C'est le bon défaut pour un fichier
        d'embeddings faciaux ; à revoir si le service tourne un jour sous un
        autre utilisateur que celui qui a construit la base.
        """
        directory = os.path.dirname(self.cache_path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            if self.known_embeddings:
                embeddings = np.stack(self.known_embeddings)
            else:
                embeddings = np.empty((0, 0), dtype=np.float32)
            with os.fdopen(fd, 'wb') as f:
                # Sur un objet fichier, savez n'ajoute pas d'extension au nom.
                np.savez(f, names=np.array(self.known_names, dtype=str),
                         embeddings=embeddings)
            os.replace(tmp_path, self.cache_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        logger.debug("Cache sauvegardé : %s", self.cache_path)

    def rebuild_database(self):
        """Force la reconstruction complète de la base depuis les images sur disque."""
        with self._maintenance():
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
            self._build_database()

    def _maintenance(self) -> threading.RLock:
        """Verrou des opérations qui modifient la base.

        Créé à la demande pour les instances construites sans __init__ (tests).
        """
        lock = self.__dict__.get("_write_lock")
        if lock is None:
            lock = self.__dict__.setdefault("_write_lock", threading.RLock())
        return lock

    # =========================================================================
    # HELPERS INTERNES
    # =========================================================================

    def _extract_best_embeddings(self, images: list) -> list:
        """
        Pour chaque image, détecte les visages et retourne l'embedding
        du plus grand visage détecté. Ignore silencieusement les images
        sans visage.

        Returns:
            Liste de np.ndarray (un embedding par image valide).
        """
        embeddings = []
        for img in images:
            if img is None:
                continue
            faces = self.app.get(img)
            if not faces:
                continue
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if best.embedding is not None:
                embeddings.append(best.embedding)
        return embeddings

    def _upsert_embedding(self, name: str, embedding: np.ndarray) -> None:
        """
        Insère ou met à jour une entrée dans la base en mémoire (thread-safe).
        Doit être appelée DEPUIS un bloc `with self._lock`.
        """
        if name in self.known_names:
            idx = self.known_names.index(name)
            self.known_embeddings[idx] = embedding
        else:
            self.known_embeddings.append(embedding)
            self.known_names.append(name)

    def _save_images_to_disk(self, dir_name: str, images: list, prefix: str = "enroll") -> None:
        """
        Sauvegarde les images source sur disque.

        Le séparateur '#' des noms multi-template est conservé tel quel : il est
        valide sur NTFS comme sur ext4, et c'est lui qui permet à _identify de
        retrouver le nom de base après un rebuild_database. Le remplacer par '_'
        rendait les entrées reconstruites indistinguables d'un nom ordinaire.

        Args:
            dir_name : sous-dossier dans known_faces/ (ex: "Armand#Face")
            images   : liste d'images BGR
            prefix   : préfixe du nom de fichier

        Raises:
            ValueError : le dossier résolu sort de known_faces/. Les méthodes
                         d'enrôlement valident le nom avant d'arriver ici ; ce
                         contrôle couvre un appelant qui ne l'aurait pas fait.
        """
        person_dir = self._entry_dir(dir_name)
        os.makedirs(person_dir, exist_ok=True)
        # Numéro libre suivant : réenrôler une personne ajoute des photos au
        # lieu d'écraser avg_000.jpg, avg_001.jpg… de l'enrôlement précédent.
        idx = 0
        for img in images:
            while os.path.exists(os.path.join(person_dir, f"{prefix}_{idx:03d}.jpg")):
                idx += 1
            cv2.imwrite(os.path.join(person_dir, f"{prefix}_{idx:03d}.jpg"), img)

    def _entry_dir(self, entry: str) -> str:
        """Dossier d'une entrée de la base, garanti directement sous known_faces/."""
        base_dir = os.path.realpath(self.known_faces_dir)
        entry_dir = os.path.realpath(os.path.join(base_dir, entry))
        if os.path.dirname(entry_dir) != base_dir:
            raise ValueError(f"Dossier d'enrôlement hors de known_faces/ : {entry!r}")
        return entry_dir

    def _entry_images(self, entry: str) -> list[str]:
        """Chemins des photos d'une entrée, triés."""
        entry_dir = self._entry_dir(entry)
        if not os.path.isdir(entry_dir):
            return []
        return sorted(os.path.join(entry_dir, f) for f in os.listdir(entry_dir)
                      if f.lower().endswith(_IMAGE_EXTENSIONS))

    def _person_entries(self, name: str) -> list[str]:
        """Entrées d'une personne sur disque : « Nom » et « Nom#label »."""
        if not os.path.isdir(self.known_faces_dir):
            return []
        return sorted(e for e in os.listdir(self.known_faces_dir)
                      if (e == name or e.startswith(name + "#"))
                      and os.path.isdir(os.path.join(self.known_faces_dir, e)))

    def _refresh_entry(self, entry: str) -> int:
        """Recalcule l'embedding d'une entrée depuis toutes ses photos sur disque.

        Même calcul que _build_database : la base en mémoire reste celle qu'un
        rebuild produirait. Retourne le nombre de photos où un visage a été
        trouvé ; à 0, l'entrée est retirée de la base.
        """
        images = [cv2.imread(path) for path in self._entry_images(entry)]
        embeddings = self._extract_best_embeddings(images)
        with self._lock:
            if embeddings:
                avg = np.mean(embeddings, axis=0)
                self._upsert_embedding(entry, avg / np.linalg.norm(avg))
            elif entry in self.known_names:
                idx = self.known_names.index(entry)
                self.known_names = self.known_names[:idx] + self.known_names[idx + 1:]
                self.known_embeddings = self.known_embeddings[:idx] + self.known_embeddings[idx + 1:]
        return len(embeddings)

    # =========================================================================
    # RECONNAISSANCE
    # =========================================================================

    def _identify(self, embedding: np.ndarray) -> tuple:
        """
        Compare un embedding à toute la base.
        Retourne (nom_nettoyé, score).

        Nettoyage multi-template : "Armand#Profil" → "Armand".
        Cela permet à la reconnaissance de rester transparente pour l'UI
        quelle que soit la méthode d'enrôlement utilisée.
        """
        with self._lock:
            if not self.known_embeddings:
                return "Inconnu", 0.0
            names = list(self.known_names)
            # Une matrice (N, 512) : un seul produit au lieu d'une boucle Python.
            known = np.stack(self.known_embeddings)

        scores = known @ (embedding / np.linalg.norm(embedding))
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if best_score <= 0.0:
            return "Inconnu", 0.0
        best_name = names[best] if best_score >= self.threshold else "Inconnu"

        # Nettoyer le suffixe multi-template avant de retourner
        clean_name = best_name.split('#')[0] if '#' in best_name else best_name
        return clean_name, best_score

    def recognize_center_face(self, crop: np.ndarray) -> dict | None:
        """
        Reconnaît uniquement le visage le plus proche du centre d'un crop de tête.

        Le crop est centré sur les keypoints faciaux d'un track : son visage est
        celui du centre. Dans une scène dense, le crop attrape aussi les visages
        des voisins (jusqu'à 5 par paire de crops mesurés sur MOT17). Les
        reconnaître coûtait ~10 ms chacun, et leur nom était attribué au track
        du crop — un voisin connu pouvait ainsi prêter son identité au mauvais
        corps. Tous les visages sont détectés, un seul est reconnu.

        Returns:
            { 'bbox': [x1,y1,x2,y2], 'name': str, 'confidence': float,
              'embedding': np.ndarray }, ou None si aucun visage.
        """
        bboxes, kpss = self.app.det_model.detect(crop, max_num=0, metric='default')
        if bboxes.shape[0] == 0 or kpss is None:
            return None

        h, w = crop.shape[:2]
        centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2
        idx = int(np.argmin(np.sum((centers - (w / 2, h / 2)) ** 2, axis=1)))

        # ArcFaceONNX.get n'utilise que face.kps et renseigne face.embedding.
        face = SimpleNamespace(kps=kpss[idx], embedding=None)
        self.app.models['recognition'].get(crop, face)

        x1, y1, x2, y2 = [int(v) for v in bboxes[idx, 0:4]]
        name, confidence = self._identify(face.embedding)
        return {
            'bbox': [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
            'name': name,
            'confidence': confidence,
            'embedding': face.embedding,
        }

    # =========================================================================
    # ENRÔLEMENT — Méthode 1 : Averaging L2 (plusieurs photos, même angle)
    # =========================================================================

    def enroll_person_average(self, name: str, frames: list) -> tuple:
        """
        Enrôle une personne en calculant la moyenne L2 de ses embeddings.

        Cas d'usage idéal : plusieurs photos frontales du même sujet
        (variations d'éclairage, légères rotations).

        Algorithme :
            1. Extraire le meilleur embedding de chaque image.
            2. Calculer la moyenne arithmétique des vecteurs d'embedding.
            3. Normaliser le vecteur moyen (L2) pour qu'il reste sur la sphère unité.
               Note : mean(unit_vectors) ≠ unit_vector, la normalisation est obligatoire.
            4. Insérer ou mettre à jour l'entrée dans la base.

        Args:
            name   : identifiant (ex: "Armand_Lauener")
            frames : liste d'images BGR

        Returns:
            (success: bool, message: str)
        """
        if not _is_safe_name(name):
            return False, f"Nom invalide : {name!r}."

        # Seules les images où un visage est trouvé sont gardées sur disque.
        kept = [img for img in frames if img is not None and self._extract_best_embeddings([img])]
        if not kept:
            return False, "Aucun visage détecté dans les images fournies."

        with self._maintenance():
            # Les nouvelles photos s'ajoutent aux précédentes, et l'embedding
            # est la moyenne de toutes : comme après un rebuild_database.
            self._save_images_to_disk(name, kept, prefix="avg")
            total = self._refresh_entry(name)
            self._save_cache()
        logger.info("[AVG] '%s' enrôlé : %d nouvelle(s) photo(s), %d au total.",
                    name, len(kept), total)
        return True, f"'{name}' enrôlé : {len(kept)} photo(s) ajoutée(s), {total} au total."

    # =========================================================================
    # ENRÔLEMENT — Méthode 2 : Multi-Template (angles distincts)
    # =========================================================================

    def enroll_person_multitemplate(self, base_name: str, frames_dict: dict) -> tuple:
        """
        Enrôle une personne sous plusieurs sous-identités (une par angle/label).

        Cas d'usage idéal : angles très différents — face, profil gauche,
        profil droit — qui ne peuvent pas être moyennés sans perte de signal.

        Chaque entrée est stockée comme "{base_name}#{label}" dans la base.
        La méthode _identify nettoie automatiquement ce suffixe lors
        de la reconnaissance → l'UI ne voit que "Armand".

        Args:
            base_name   : nom de base de la personne (ex: "Armand")
            frames_dict : dict { label: np.ndarray } — une image par template
                          ex: { "Face": img_front, "ProfilG": img_left }

        Returns:
            (success: bool, message: str)
        """
        # Tout valider avant la première écriture : un label refusé en cours de
        # boucle laisserait un enrôlement à moitié fait.
        if not _is_safe_name(base_name):
            return False, f"Nom invalide : {base_name!r}."
        bad_labels = [label for label in frames_dict if not _is_safe_name(label)]
        if bad_labels:
            return False, f"Label(s) invalide(s) : {bad_labels!r}."

        enrolled_labels = []

        with self._maintenance():
            for label, frame in frames_dict.items():
                if not self._extract_best_embeddings([frame]):
                    logger.warning("[MT] Aucun visage dans le template '%s', ignoré.", label)
                    continue

                template_name = f"{base_name}#{label}"
                self._save_images_to_disk(template_name, [frame], prefix=label.lower())
                self._refresh_entry(template_name)
                enrolled_labels.append(label)

            if not enrolled_labels:
                return False, "Aucun visage détecté dans les images fournies."

            self._save_cache()
        logger.info("[MT] '%s' : %d template(s) -> %s", base_name, len(enrolled_labels), enrolled_labels)
        return True, f"'{base_name}' enrôlé en {len(enrolled_labels)} template(s) : {enrolled_labels}."

    # =========================================================================
    # GESTION DES PERSONNES
    # =========================================================================

    def list_people(self) -> list[dict]:
        """Personnes connues, regroupées par nom de base.

        Chaque entrée : name, templates (labels multitemplate), photos (nombre
        de photos sur disque), thumbnail (chemin de la première photo ou None),
        enrolled (présente dans la base de reconnaissance).
        """
        people: dict[str, dict] = {}

        def person(name):
            return people.setdefault(name, {"name": name, "templates": [], "photos": 0,
                                            "thumbnail": None, "enrolled": False})

        if os.path.isdir(self.known_faces_dir):
            for entry in sorted(os.listdir(self.known_faces_dir)):
                if not os.path.isdir(os.path.join(self.known_faces_dir, entry)):
                    continue
                name, _, label = entry.partition("#")
                if not _is_safe_name(name):
                    continue
                p = person(name)
                if label:
                    p["templates"].append(label)
                images = self._entry_images(entry)
                p["photos"] += len(images)
                if images and p["thumbnail"] is None:
                    p["thumbnail"] = images[0]
        with self._lock:
            names = list(self.known_names)
        for entry in names:
            person(entry.split("#")[0])["enrolled"] = True
        return sorted(people.values(), key=lambda p: p["name"].lower())

    def rename_person(self, old: str, new: str) -> tuple[str, str]:
        """Renomme une personne : ses dossiers dans known_faces/ et ses entrées en base."""
        if not _is_safe_name(old) or not _is_safe_name(new):
            return INVALID, f"Nom invalide : {old!r} ou {new!r}."
        with self._maintenance():
            entries = self._person_entries(old)
            with self._lock:
                in_memory = [n for n in self.known_names if n.split("#")[0] == old]
                taken = any(n.split("#")[0] == new for n in self.known_names)
            if not entries and not in_memory:
                return NOT_FOUND, f"'{old}' n'existe pas."
            if old == new:
                return OK, f"'{old}' garde son nom."
            if taken or self._person_entries(new):
                return CONFLICT, f"'{new}' existe déjà."
            for entry in entries:
                os.rename(self._entry_dir(entry), self._entry_dir(new + entry[len(old):]))
            with self._lock:
                self.known_names = [new + n[len(old):] if n.split("#")[0] == old else n
                                    for n in self.known_names]
            self._save_cache()
        logger.info("Personne renommée : '%s' -> '%s'", old, new)
        return OK, f"'{old}' renommé en '{new}'."

    def delete_person(self, name: str) -> tuple[str, str]:
        """Supprime une personne : photos sur disque et entrées en base (droit à l'effacement)."""
        if not _is_safe_name(name):
            return INVALID, f"Nom invalide : {name!r}."
        with self._maintenance():
            entries = self._person_entries(name)
            with self._lock:
                keep = [i for i, n in enumerate(self.known_names) if n.split("#")[0] != name]
                found = len(keep) != len(self.known_names)
                if found:
                    self.known_names = [self.known_names[i] for i in keep]
                    self.known_embeddings = [self.known_embeddings[i] for i in keep]
            if not entries and not found:
                return NOT_FOUND, f"'{name}' n'existe pas."
            for entry in entries:
                shutil.rmtree(self._entry_dir(entry))
            self._save_cache()
        logger.info("Personne supprimée : '%s' (%d dossier(s))", name, len(entries))
        return OK, f"'{name}' supprimé."
