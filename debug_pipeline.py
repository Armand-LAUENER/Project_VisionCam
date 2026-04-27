"""
debug_pipeline.py — Diagnostic du pipeline de reconnaissance.

Lance chaque étage en isolation et affiche exactement où ça bloque.
Usage : python debug_pipeline.py
        python debug_pipeline.py --image chemin/vers/photo.jpg
"""

import argparse
import sys
import cv2
import numpy as np
import pickle
import os

# ──────────────────────────────────────────────────────────────────────────────
# Étape 0 — Vérification des embeddings
# ──────────────────────────────────────────────────────────────────────────────

def check_embeddings():
    print("\n" + "="*60)
    print("ÉTAPE 0 — Base d'embeddings")
    print("="*60)

    import config
    if not os.path.exists(config.EMBEDDINGS_CACHE_PATH):
        print(f"[FAIL] Fichier embeddings introuvable : {config.EMBEDDINGS_CACHE_PATH}")
        return False

    with open(config.EMBEDDINGS_CACHE_PATH, 'rb') as f:
        db = pickle.load(f)

    names = list(db.keys())
    print(f"[OK]   {len(names)} entrée(s) dans la base : {names}")
    for name, emb in db.items():
        print(f"       '{name}' → embedding shape={np.array(emb).shape}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Étape 1 — YOLO-Pose : corps + keypoints nez
# ──────────────────────────────────────────────────────────────────────────────

def check_yolo(frame):
    print("\n" + "="*60)
    print("ÉTAPE 1 — YOLO-Pose (corps + keypoints)")
    print("="*60)

    import config
    from ultralytics import YOLO

    model = YOLO(config.YOLO_MODEL)
    results = model.predict(frame, classes=[0], conf=config.YOLO_CONF_THRESHOLD,
                            verbose=False)

    detections = []
    for result in results:
        kps_data = result.keypoints.data if result.keypoints is not None else None
        n_persons = len(result.boxes)
        print(f"[OK]   {n_persons} personne(s) détectée(s)")

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            print(f"\n  Personne #{i+1}  bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]  conf={conf:.2f}")

            nose = None
            if kps_data is not None and i < len(kps_data):
                kp = kps_data[i]
                nx, ny, nc = float(kp[0][0]), float(kp[0][1]), float(kp[0][2])
                nose = (nx, ny, nc)
                status = "OK" if nc >= config.POSE_NOSE_CONF_THRESHOLD else "SKIP (conf trop basse)"
                print(f"  Nez  : x={nx:.1f}  y={ny:.1f}  conf={nc:.3f}  → {status}")
                print(f"  Seuil POSE_NOSE_CONF_THRESHOLD = {config.POSE_NOSE_CONF_THRESHOLD}")
            else:
                print(f"  Nez  : [FAIL] keypoints absents pour cette personne")

            detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'nose': nose})

    if not detections:
        print("[FAIL] Aucune personne détectée — vérifie que tu es bien dans le champ")

    return detections


# ──────────────────────────────────────────────────────────────────────────────
# Étape 2 — Crop nez → InsightFace
# ──────────────────────────────────────────────────────────────────────────────

def check_insightface(frame, detections):
    print("\n" + "="*60)
    print("ÉTAPE 2 — InsightFace sur crop nez")
    print("="*60)

    import config
    from core.face_recognition import FaceRecognizer

    recognizer = FaceRecognizer(
        config.KNOWN_FACES_DIR,
        threshold=config.RECOGNITION_THRESHOLD,
        cache_path=config.EMBEDDINGS_CACHE_PATH,
    )
    print(f"[OK]   Personnes connues : {list(recognizer.known_names)}")
    print(f"       Seuil RECOGNITION_THRESHOLD = {config.RECOGNITION_THRESHOLD}")

    h_img, w_img = frame.shape[:2]

    for i, det in enumerate(detections):
        nose = det['nose']
        x1, y1, x2, y2 = det['bbox']
        print(f"\n  Personne #{i+1}")

        if nose is None or nose[2] < config.POSE_NOSE_CONF_THRESHOLD:
            print(f"  [SKIP] Nez absent ou conf={nose[2] if nose else 'N/A':.3f} < {config.POSE_NOSE_CONF_THRESHOLD}")
            print(f"         → InsightFace ne sera JAMAIS appelé pour cette personne")
            continue

        nx, ny, nc = nose
        body_height = max(1, y2 - y1)
        half = max(config.POSE_CROP_HALF_SIZE, int(body_height * 0.25))
        cx, cy = int(nx), int(ny)

        crop_x1 = max(0, cx - half)
        crop_y1 = max(0, cy - half)
        crop_x2 = min(w_img, cx + half)
        crop_y2 = min(h_img, cy + half)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        print(f"  Crop  : [{crop_x1},{crop_y1},{crop_x2},{crop_y2}]  ({crop_w}x{crop_h}px)  half={half}")

        if crop_w < 20 or crop_h < 20:
            print(f"  [FAIL] Crop trop petit ({crop_w}x{crop_h}) → skip")
            continue

        head_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.imwrite(f"debug_crop_person{i+1}.jpg", head_crop)
        print(f"  Crop sauvegardé : debug_crop_person{i+1}.jpg")

        faces = recognizer.detect_and_recognize(head_crop)
        print(f"  InsightFace : {len(faces)} visage(s) trouvé(s)")

        for j, face in enumerate(faces):
            status = "OK" if face['confidence'] >= config.RECOGNITION_THRESHOLD else "SOUS SEUIL"
            print(f"    Visage #{j+1}  nom='{face['name']}'  conf={face['confidence']:.3f}  → {status}")

        if not faces:
            print(f"  [FAIL] InsightFace n'a détecté aucun visage dans le crop")
            print(f"         → Vérifie debug_crop_person{i+1}.jpg : le visage est-il visible ?")


# ──────────────────────────────────────────────────────────────────────────────
# Étape 3 — Vote buffer : simulation du consensus
# ──────────────────────────────────────────────────────────────────────────────

def check_vote_buffer():
    print("\n" + "="*60)
    print("ÉTAPE 3 — Vote buffer (consensus)")
    print("="*60)

    from core.face_body_tracker import FaceBodyTracker
    print(f"  VOTE_WINDOW = {FaceBodyTracker.VOTE_WINDOW}")
    required = (FaceBodyTracker.VOTE_WINDOW + 1) // 2
    print(f"  Votes requis pour consensus = {required}/{FaceBodyTracker.VOTE_WINDOW}")
    print(f"  FACE_RECOGNITION_SKIP = 5 → reconnaissance toutes les 5 frames")
    print(f"  → Il faut au minimum {required * 5} frames consécutives de détection")
    print(f"    pour qu'une identité soit confirmée (~{required * 5 / 30:.1f}s à 30 FPS)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Chemin vers une image de test (sinon webcam)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("DIAGNOSTIC PIPELINE VisionCam")
    print("="*60)

    # Récupérer une frame
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[FAIL] Impossible de lire l'image : {args.image}")
            sys.exit(1)
        print(f"Image chargée : {args.image}  ({frame.shape[1]}x{frame.shape[0]})")
    else:
        import config
        print(f"Ouverture caméra : {config.CAMERA_SOURCE}")
        cap = cv2.VideoCapture(config.CAMERA_SOURCE)
        if not cap.isOpened():
            print("[FAIL] Caméra inaccessible")
            sys.exit(1)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("[FAIL] Impossible de lire une frame")
            sys.exit(1)
        cv2.imwrite("debug_frame.jpg", frame)
        print(f"Frame capturée ({frame.shape[1]}x{frame.shape[0]}) → debug_frame.jpg")

    # Lancer les étapes
    ok = check_embeddings()
    if not ok:
        print("\n[STOP] Corrige les embeddings avant de continuer.")
        return

    detections = check_yolo(frame)
    check_insightface(frame, detections)
    check_vote_buffer()

    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print("Inspecte les lignes [FAIL] et [SKIP] ci-dessus.")
    print("Les crops sont sauvegardés en debug_crop_personN.jpg")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
