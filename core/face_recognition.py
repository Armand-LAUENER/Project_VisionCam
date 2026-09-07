import glob
import logging
import os
import shutil
import threading
import cv2
import numpy as np
import pickle
import config
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


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
    def __init__(self, known_faces_dir="known_faces", threshold=0.45, cache_path="data/embeddings.pkl"):
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.cache_path = cache_path
        self.known_embeddings = []
        self.known_names = []
        self._lock = threading.Lock()   # Protège known_embeddings/known_names (accès multi-thread)

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
        """Charge le cache ou reconstruit la base d'embeddings."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        if os.path.exists(self.cache_path):
            logger.info("Chargement du cache : %s", self.cache_path)
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self.known_embeddings = data['embeddings']
                self.known_names = data['names']
            logger.info("%d entrée(s) chargée(s) depuis le cache", len(self.known_names))
        else:
            self._build_database()

    def _build_database(self):
        """Parcourt known_faces/ et calcule l'embedding moyen par personne."""
        logger.info("Construction de la base depuis : %s", self.known_faces_dir)
        self.known_embeddings = []
        self.known_names = []

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)
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
                self.known_embeddings.append(avg)
                self.known_names.append(person_name)
                logger.debug("OK %s : %d image(s) traitée(s)", person_name, len(embeddings))
            else:
                logger.warning("SKIP %s : aucun embedding valide", person_name)

        self._save_cache()
        logger.info("Base construite : %d entrée(s)", len(self.known_names))

    def _save_cache(self):
        """Sauvegarde les embeddings en cache (appeler sous lock si partagé)."""
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.known_embeddings,
                'names': self.known_names
            }, f)
        logger.debug("Cache sauvegardé : %s", self.cache_path)

    def rebuild_database(self):
        """Force la reconstruction complète de la base depuis les images sur disque."""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        self._build_database()

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

        Args:
            dir_name : sous-dossier dans known_faces/ (# remplacé par _ pour Windows)
            images   : liste d'images BGR
            prefix   : préfixe du nom de fichier
        """
        safe_dir = dir_name.replace('#', '_')
        person_dir = os.path.join(self.known_faces_dir, safe_dir)
        os.makedirs(person_dir, exist_ok=True)
        for idx, img in enumerate(images):
            cv2.imwrite(os.path.join(person_dir, f"{prefix}_{idx:03d}.jpg"), img)

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
            embeddings = list(self.known_embeddings)

        embedding_norm = embedding / np.linalg.norm(embedding)
        best_name = "Inconnu"
        best_score = 0.0

        for i, known_emb in enumerate(embeddings):
            score = float(np.dot(embedding_norm, known_emb))
            if score > best_score:
                best_score = score
                best_name = names[i] if score >= self.threshold else "Inconnu"

        # Nettoyer le suffixe multi-template avant de retourner
        clean_name = best_name.split('#')[0] if '#' in best_name else best_name
        return clean_name, best_score

    def detect_and_recognize(self, frame: np.ndarray) -> list:
        """
        Détecte et reconnaît les visages dans une frame.

        Returns:
            [{ 'bbox': [x1,y1,x2,y2], 'name': str, 'confidence': float,
               'embedding': np.ndarray }, ...]
        """
        results = []
        faces = self.app.get(frame)
        if not faces:
            return results

        h, w = frame.shape[:2]
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            name, confidence = self._identify(face.embedding)
            results.append({
                'bbox': [x1, y1, x2, y2],
                'name': name,
                'confidence': confidence,
                'embedding': face.embedding
            })

        return results

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
        embeddings = self._extract_best_embeddings(frames)
        if not embeddings:
            return False, "Aucun visage détecté dans les images fournies."

        # Moyenne puis normalisation L2
        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)

        # Persistance disque (source pour rebuild_database futur)
        self._save_images_to_disk(name, frames, prefix="avg")

        # Mise à jour atomique de la base
        with self._lock:
            self._upsert_embedding(name, avg)

        self._save_cache()
        logger.info("[AVG] '%s' enrôlé : %d embedding(s) -> moyenne L2.", name, len(embeddings))
        return True, f"'{name}' enrôlé par averaging sur {len(embeddings)} image(s)."

    # Alias rétro-compatible avec l'ancienne méthode enroll()
    def enroll(self, name: str, images: list) -> tuple:
        return self.enroll_person_average(name, images)

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
        enrolled_labels = []

        for label, frame in frames_dict.items():
            embeddings = self._extract_best_embeddings([frame])
            if not embeddings:
                logger.warning("[MT] Aucun visage dans le template '%s', ignoré.", label)
                continue

            # Un seul embedding par template → normalisation directe (pas de moyenne)
            emb = embeddings[0]
            emb = emb / np.linalg.norm(emb)

            template_name = f"{base_name}#{label}"
            self._save_images_to_disk(template_name, [frame], prefix=label.lower())

            with self._lock:
                self._upsert_embedding(template_name, emb)

            enrolled_labels.append(label)

        if not enrolled_labels:
            return False, "Aucun visage détecté dans les images fournies."

        self._save_cache()
        logger.info("[MT] '%s' : %d template(s) -> %s", base_name, len(enrolled_labels), enrolled_labels)
        return True, f"'{base_name}' enrôlé en {len(enrolled_labels)} template(s) : {enrolled_labels}."
