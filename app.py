from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import numpy as np
import os

# ── Imports de tes modules ──
from core.face_recognition import FaceRecognizer
from core.tracker import PersonTracker
from core.pose_estimation import PoseEstimator

app = Flask(__name__)

# ══════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════
CAMERA_SOURCE = "http://192.168.27.65:5000/video"         # 0 = webcam intégrée, ou URL IP "http://IP:5000/video"
KNOWN_FACES_DIR = "known_faces"
CONFIDENCE_THRESHOLD = 0.45
FRAME_SKIP = 2             # Traiter 1 frame sur 2 pour la perf

# ══════════════════════════════════════════
# INITIALISATION DES MODULES
# ══════════════════════════════════════════
print("[INIT] Chargement des modules...")

face_recognizer = FaceRecognizer(KNOWN_FACES_DIR, threshold=CONFIDENCE_THRESHOLD)
tracker = PersonTracker()
pose_estimator = PoseEstimator()

print(f"[INIT] Visages connus : {list(face_recognizer.known_names)}")
print("[INIT] Modules chargés ✅")

# ══════════════════════════════════════════
# ÉTAT GLOBAL (thread-safe)
# ══════════════════════════════════════════
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_frame = None
        self.currently_present = []
        self.fps = 0.0
        self.total_known = len(face_recognizer.known_names)
        self.running = True

state = AppState()

# ══════════════════════════════════════════
# BOUCLE DE TRAITEMENT (Thread séparé)
# ══════════════════════════════════════════
def processing_loop():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra !")
        return

    print(f"[CAM] Caméra ouverte : {int(cap.get(3))}x{int(cap.get(4))}")

    frame_count = 0
    fps_time = time.time()
    fps_counter = 0
    detections_cache = []

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("[CAM] Frame perdue, retry...")
            time.sleep(0.1)
            continue

        frame_count += 1
        fps_counter += 1
        display_frame = frame.copy()

        # ── Traitement lourd uniquement toutes les N frames ──
        if frame_count % FRAME_SKIP == 0:
            try:
                # 1) Détection + reconnaissance faciale
                faces = face_recognizer.detect_and_recognize(frame)

                # 2) Tracking
                detections_for_tracker = []
                for f in faces:
                    x1, y1, x2, y2 = f['bbox']
                    detections_for_tracker.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': f.get('confidence', 0),
                        'name': f.get('name', 'Inconnu')
                    })

                tracked = tracker.update(detections_for_tracker, frame)
                detections_cache = tracked

            except Exception as e:
                print(f"[ERREUR TRAITEMENT] {e}")

        # ── Dessiner les résultats sur chaque frame ──
        present_list = []

        for det in detections_cache:
            x1, y1, x2, y2 = det['bbox']
            name = det.get('name', 'Inconnu')
            track_id = det.get('track_id', -1)
            conf = det.get('confidence', 0)

            is_known = name != 'Inconnu'
            color = (0, 245, 160) if is_known else (100, 100, 255)

            # Rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{name} #{track_id}"
            if conf > 0:
                label += f" ({conf*100:.0f}%)"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Estimation de pose (optionnel)
            try:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    pose = pose_estimator.estimate(person_crop)
                    if pose:
                        cv2.putText(display_frame, f"Pose: {pose}",
                                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except:
                pass

            present_list.append({
                'name': name,
                'track_id': track_id,
                'confidence': conf
            })

        # ── FPS ──
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()
        else:
            current_fps = state.fps

        # Afficher FPS sur la frame
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)

        # Compteur de personnes
        cv2.putText(display_frame, f"Personnes: {len(present_list)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 245, 160), 2)

        # ── Mise à jour de l'état global ──
        with state.lock:
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            state.current_frame = buffer.tobytes()
            state.currently_present = present_list
            state.fps = current_fps

    cap.release()
    print("[CAM] Caméra fermée")


# ══════════════════════════════════════════
# ROUTES FLASK
# ══════════════════════════════════════════
def generate_frames():
    while True:
        with state.lock:
            frame = state.current_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # ~30 FPS max


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with state.lock:
        return jsonify({
            'currently_present': state.currently_present,
            'fps': state.fps,
            'total_known': state.total_known
        })


# ══════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════
if __name__ == '__main__':
    # Créer le dossier known_faces si absent
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    # Lancer le traitement dans un thread
    process_thread = threading.Thread(target=processing_loop, daemon=True)
    process_thread.start()
    print("[SERVER] Démarrage sur http://0.0.0.0:8080")

    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
