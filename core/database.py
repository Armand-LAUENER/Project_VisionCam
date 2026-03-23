"""
core/database.py — Gestion de la base de données d'embeddings faciaux.

Cette classe gère :
  1. Le chargement depuis un cache pickle (démarrage rapide).
  2. La compilation complète depuis les images sur disque (premier lancement ou mise à jour).
  3. La sauvegarde / mise à jour incrémentale du cache.

Les embeddings sont normalisés (norme L2 = 1) pour permettre une comparaison
par distance cosinus via un simple produit scalaire.
"""

import os
import pickle
import glob
from typing import Dict, Optional

import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Import de la configuration centralisée
import config


class FaceDatabase:
    """
    Base de données d'embeddings faciaux avec système de mise en cache.

    Attributs:
        db (dict): Dictionnaire {nom_personne: embedding_moyen_normalisé}
        analyzer (FaceAnalysis): Instance InsightFace pour l'extraction d'embeddings
    """

    def __init__(self):
        """Initialise la base de données et le modèle InsightFace."""

        # --- Initialisation du modèle InsightFace sur GPU ---
        print("[DATABASE] Chargement du modèle InsightFace...")
        self.analyzer = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            providers=config.ONNX_PROVIDERS
        )
        self.analyzer.prepare(
            ctx_id=0,  # GPU index 0
            det_size=config.INSIGHTFACE_DET_SIZE
        )
        print("[DATABASE] Modèle InsightFace chargé avec succès (GPU).")

        # Dictionnaire principal : {"Prenom_Nom": np.array(512,)}
        self.db: Dict[str, np.ndarray] = {}

        # Chargement automatique au démarrage
        self.load_database()

    # =========================================================================
    # MÉTHODE PRINCIPALE : Chargement intelligent
    # =========================================================================

    def load_database(self) -> None:
        """
        Charge la base de données d'embeddings.

        Stratégie :
          1. Tente de charger le cache pickle (rapide).
          2. Si le cache est absent ou corrompu, recompile depuis les images.
        """
        cache_path = config.EMBEDDINGS_CACHE_PATH

        if os.path.exists(cache_path):
            try:
                self.db = self._load_from_cache(cache_path)
                print(f"[DATABASE] Cache chargé : {len(self.db)} personne(s) trouvée(s).")
                self._print_summary()
                return
            except (pickle.UnpicklingError, EOFError, KeyError, Exception) as e:
                print(f"[DATABASE] Cache corrompu ({e}). Recompilation en cours...")

        # Pas de cache ou cache invalide → compilation complète
        self.db = self._compile_from_disk()

        if self.db:
            self._save_to_cache(self.db, cache_path)
            print(f"[DATABASE] Base compilée et sauvegardée : {len(self.db)} personne(s).")
        else:
            print("[DATABASE] ATTENTION : Aucune personne trouvée dans le dossier d'images.")
            print(f"[DATABASE] Ajoutez des sous-dossiers dans : {config.CAPTURED_FACES_DIR}")

        self._print_summary()

    # =========================================================================
    # CHARGEMENT DEPUIS LE CACHE PICKLE
    # =========================================================================

    @staticmethod
    def _load_from_cache(cache_path: str) -> Dict[str, np.ndarray]:
        """
        Charge le dictionnaire d'embeddings depuis un fichier pickle.

        Args:
            cache_path: Chemin vers le fichier .pkl

        Returns:
            Dictionnaire {nom: embedding}
        """
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        # Vérification de l'intégrité minimale
        if not isinstance(data, dict):
            raise ValueError("Le cache ne contient pas un dictionnaire valide.")

        for name, emb in data.items():
            if not isinstance(emb, np.ndarray):
                raise ValueError(f"L'embedding de '{name}' n'est pas un np.ndarray.")

        return data

    # =========================================================================
    # COMPILATION DEPUIS LES IMAGES SUR DISQUE
    # =========================================================================

    def _compile_from_disk(self) -> Dict[str, np.ndarray]:
        """
        Parcourt les sous-dossiers de captured_faces/, extrait les embeddings
        de chaque image, calcule la moyenne par personne et normalise.

        Structure attendue :
            data/captured_faces/
            ├── Jean_Dupont/
            │   ├── 01.jpg
            │   ├── 02.jpg
            │   └── 03.jpg
            └── Marie_Martin/
                ├── 01.jpg
                └── 02.jpg

        Returns:
            Dictionnaire {nom_personne: embedding_moyen_normalisé}
        """
        faces_dir = config.CAPTURED_FACES_DIR
        database = {}

        # Liste des sous-dossiers (chaque sous-dossier = une personne)
        person_dirs = [
            d for d in os.listdir(faces_dir)
            if os.path.isdir(os.path.join(faces_dir, d))
        ]

        if not person_dirs:
            return database

        print(f"[DATABASE] Compilation de {len(person_dirs)} personne(s)...")

        for person_name in sorted(person_dirs):
            person_path = os.path.join(faces_dir, person_name)

            # Récupère toutes les images (jpg, jpeg, png)
            image_paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                image_paths.extend(glob.glob(os.path.join(person_path, ext)))

            if not image_paths:
                print(f"  [SKIP] '{person_name}' — Aucune image trouvée.")
                continue

            # Extraction des embeddings pour chaque image
            embeddings = self._extract_embeddings_from_images(image_paths, person_name)

            if not embeddings:
                print(f"  [SKIP] '{person_name}' — Aucun visage détecté dans les images.")
                continue

            # Calcul de l'embedding moyen puis normalisation L2
            mean_embedding = np.mean(embeddings, axis=0)
            normalized_embedding = self._normalize(mean_embedding)

            database[person_name] = normalized_embedding
            print(f"  [OK] '{person_name}' — {len(embeddings)} visage(s) extraits sur {len(image_paths)} image(s).")

        return database

    def _extract_embeddings_from_images(
        self, image_paths: list, person_name: str
    ) -> list:
        """
        Extrait un embedding par image via InsightFace.

        Args:
            image_paths: Liste des chemins d'images
            person_name: Nom de la personne (pour les logs)

        Returns:
            Liste de np.ndarray (un embedding par image valide)
        """
        import cv2  # Import local pour éviter un import global inutile si non utilisé

        embeddings = []

        for img_path in image_paths:
            # Lecture de l'image en BGR (format attendu par InsightFace)
            img = cv2.imread(img_path)

            if img is None:
                print(f"    [WARN] Impossible de lire : {img_path}")
                continue

            # Détection des visages dans l'image
            faces = self.analyzer.get(img)

            if not faces:
                print(f"    [WARN] Aucun visage détecté dans : {os.path.basename(img_path)}")
                continue

            # On prend le visage avec le score de détection le plus élevé
            best_face = max(faces, key=lambda f: f.det_score)

            # Vérification que l'embedding existe
            if best_face.embedding is not None:
                embeddings.append(best_face.embedding)

        return embeddings

    # =========================================================================
    # SAUVEGARDE DU CACHE
    # =========================================================================

    @staticmethod
    def _save_to_cache(database: Dict[str, np.ndarray], cache_path: str) -> None:
        """
        Sérialise le dictionnaire d'embeddings dans un fichier pickle.

        Args:
            database: Dictionnaire à sauvegarder
            cache_path: Chemin de destination
        """
        # Création du dossier parent si nécessaire
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, "wb") as f:
            pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[DATABASE] Cache sauvegardé dans : {cache_path}")

    # =========================================================================
    # MISE À JOUR INCRÉMENTALE (utilisée par enroll_user.py)
    # =========================================================================

    def add_person(self, name: str, embedding: np.ndarray) -> None:
        """
        Ajoute ou met à jour une personne dans la base et sauvegarde le cache.

        Args:
            name: Identifiant de la personne (ex: "Jean_Dupont")
            embedding: Embedding moyen déjà calculé (sera normalisé)
        """
        normalized = self._normalize(embedding)
        self.db[name] = normalized
        self._save_to_cache(self.db, config.EMBEDDINGS_CACHE_PATH)
        print(f"[DATABASE] '{name}' ajouté(e) à la base ({len(self.db)} personne(s) au total).")

    # =========================================================================
    # RECONNAISSANCE : Recherche du plus proche voisin
    # =========================================================================

    def recognize(self, embedding: np.ndarray) -> Optional[str]:
        """
        Compare un embedding à la base et retourne le nom de la personne
        la plus proche si la distance cosinus est sous le seuil.

        La distance cosinus est calculée comme : 1 - (a · b)
        où a et b sont normalisés (norme L2 = 1).
        Donc le produit scalaire donne directement la similarité cosinus.

        Args:
            embedding: Embedding du visage à identifier (512-d)

        Returns:
            Nom de la personne reconnue, ou None si aucun match
        """
        if not self.db:
            return None

        # Normalisation de l'embedding entrant
        query = self._normalize(embedding)

        best_name = None
        best_similarity = -1.0  # Similarité cosinus : -1 (opposé) à +1 (identique)

        for name, ref_embedding in self.db.items():
            # Produit scalaire entre vecteurs normalisés = similarité cosinus
            similarity = float(np.dot(query, ref_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        # Conversion en "distance" pour comparer au seuil
        # Seuil de 0.65 signifie : similarité minimale de 0.35 ? Non.
        # Convention InsightFace : on compare directement la similarité au seuil.
        # Si similarité >= seuil → match validé.
        if best_similarity >= config.RECOGNITION_THRESHOLD:
            return best_name

        return None

    # =========================================================================
    # UTILITAIRES
    # =========================================================================

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """
        Normalise un vecteur à norme L2 = 1 (nécessaire pour la distance cosinus).

        Args:
            embedding: Vecteur brut

        Returns:
            Vecteur normalisé
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _print_summary(self) -> None:
        """Affiche un résumé formaté de la base de données."""
        if not self.db:
            print("[DATABASE] Base vide.")
            return

        print("[DATABASE] ┌─────────────────────────────────────┐")
        print("[DATABASE] │     PERSONNES ENREGISTRÉES          │")
        print("[DATABASE] ├─────────────────────────────────────┤")
        for i, name in enumerate(sorted(self.db.keys()), 1):
            print(f"[DATABASE] │  {i:>2}. {name:<32}│")
        print("[DATABASE] └─────────────────────────────────────┘")

    def get_names(self) -> list:
        """Retourne la liste triée des noms enregistrés."""
        return sorted(self.db.keys())

    def __len__(self) -> int:
        """Retourne le nombre de personnes dans la base."""
        return len(self.db)

    def __contains__(self, name: str) -> bool:
        """Vérifie si une personne est dans la base."""
        return name in self.db


# =============================================================================
# POINT D'ENTRÉE POUR TEST INDÉPENDANT
# =============================================================================

if __name__ == "__main__":
    """
    Permet de tester la compilation de la base en lançant directement :
        python -m core.database
    ou :
        python core/database.py
    """
    import sys
    # Ajout du répertoire parent au path pour l'import de config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 50)
    print("  TEST DE LA BASE DE DONNÉES FACIALE")
    print("=" * 50)

    # Force la recompilation en supprimant le cache
    if "--recompile" in sys.argv:
        if os.path.exists(config.EMBEDDINGS_CACHE_PATH):
            os.remove(config.EMBEDDINGS_CACHE_PATH)
            print("[TEST] Cache supprimé. Recompilation forcée.\n")

    # Instanciation (déclenche le chargement automatique)
    db = FaceDatabase()

    print(f"\n[TEST] Nombre de personnes : {len(db)}")
    print(f"[TEST] Noms : {db.get_names()}")
