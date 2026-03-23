import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", threshold=0.45, cache_path="data/embeddings.pkl"):
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.cache_path = cache_path
        self.known_embeddings = []
        self.known_names = []

        # ── Charger InsightFace ──
        print("[FACE] Chargement InsightFace (buffalo_l)...")
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[FACE] InsightFace chargé ✅")

        # ── Charger ou construire la base ──
        self._load_or_build_database()

    def _load_or_build_database(self):
        """Charge le cache ou reconstruit la base d'embeddings."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        if os.path.exists(self.cache_path):
            print(f"[FACE] Chargement du cache : {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self.known_embeddings = data['embeddings']
                self.known_names = data['names']
            print(f"[FACE] {len(self.known_names)} personnes chargées depuis le cache")
        else:
            self._build_database()

    def _build_database(self):
        """Parcourt known_faces/ et calcule l'embedding moyen par personne."""
        print(f"[FACE] Construction de la base depuis : {self.known_faces_dir}")
        self.known_embeddings = []
        self.known_names = []

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)
            print("[FACE] Dossier known_faces vide, aucune personne enregistrée.")
            return

        for person_name in sorted(os.listdir(self.known_faces_dir)):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"  [WARN] Impossible de lire : {img_path}")
                    continue

                faces = self.app.get(img)
                if faces:
                    # Prendre le visage le plus grand (principal)
                    best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    embeddings.append(best_face.embedding)
                else:
                    print(f"  [WARN] Aucun visage détecté : {img_path}")

            if embeddings:
                # Embedding moyen normalisé
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                self.known_embeddings.append(avg_embedding)
                self.known_names.append(person_name)
                print(f"  ✅ {person_name} : {len(embeddings)} images traitées")
            else:
                print(f"  ❌ {person_name} : aucun embedding valide")

        # Sauvegarder le cache
        self._save_cache()
        print(f"[FACE] Base construite : {len(self.known_names)} personnes")

    def _save_cache(self):
        """Sauvegarde les embeddings en cache."""
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.known_embeddings,
                'names': self.known_names
            }, f)
        print(f"[FACE] Cache sauvegardé : {self.cache_path}")

    def rebuild_database(self):
        """Force la reconstruction de la base."""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        self._build_database()

    def _identify(self, embedding):
        """Compare un embedding avec la base. Retourne (nom, score)."""
        if not self.known_embeddings:
            return "Inconnu", 0.0

        embedding_norm = embedding / np.linalg.norm(embedding)
        best_name = "Inconnu"
        best_score = 0.0

        for i, known_emb in enumerate(self.known_embeddings):
            # Similarité cosinus
            score = np.dot(embedding_norm, known_emb)
            if score > best_score:
                best_score = score
                if score >= self.threshold:
                    best_name = self.known_names[i]

        return best_name, float(best_score)

    def detect_and_recognize(self, frame):
        """
        Détecte et reconnaît les visages dans une frame.
        Retourne une liste de dict :
            [{ 'bbox': [x1,y1,x2,y2], 'name': str, 'confidence': float, 'embedding': np.array }, ...]
        """
        results = []

        faces = self.app.get(frame)
        if not faces:
            return results

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]

            # Clamp aux dimensions de la frame
            h, w = frame.shape[:2]
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
