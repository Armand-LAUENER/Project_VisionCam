from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
import time

import config
from core.face_recognition import FaceRecognizer
from core.face_body_tracker import FaceBodyTracker
from core.pose_estimation import PoseEstimator

app = Flask(__name__)

# ══════════════════════════════════════════
# INITIALISATION DES MODULES
# ══════════════════════════════════════════
print("[INIT] Chargement des modules...")
print(f"[INIT] Source vidéo : {'WEBCAM' if config.USE_LOCAL_CAM else config.REMOTE_SOURCE}")

face_recognizer = FaceRecognizer(
    config.KNOWN_FACES_DIR,
    threshold=config.RECOGNITION_THRESHOLD,
    cache_path=config.EMBEDDINGS_CACHE_PATH,
)
tracker = FaceBodyTracker(face_recognizer)
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
# BOUCLE DE TRAITEMENT (thread séparé)
# ══════════════════════════════════════════
def _estimate_pose_safe(frame, bbox, track_id):
    """Calcule la pose depuis le crop du corps, avec log explicite en cas d'échec."""
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        return pose_estimator.estimate(crop)
    except Exception as e:
        print(f"[POSE] Échec sur track #{track_id} : {type(e).__name__}: {e}")
        return None


def _draw_person(display_frame, person, pose, frame_count):
    """
    Dessine bbox + label + pose pour une TrackedPerson.

    Modes d'affichage selon l'état d'identification :
      - Vert  (0, 245, 160) : visage reconnu récemment (< FACE_FRESHNESS_FRAMES)
      - Orange(0, 165, 255) : identité connue mais visage perdu — tracking par corps
      - Bleu  (100, 100, 255): personne inconnue
    """
    x1, y1, x2, y2 = person.body_bbox

    frames_since_face = frame_count - person.last_face_frame

    if person.name == 'Inconnu':
        color = (100, 100, 255)     # Bleu — inconnu
        mode_tag = None
    elif frames_since_face <= config.FACE_FRESHNESS_FRAMES:
        color = (0, 245, 160)       # Vert — visage vu récemment
        mode_tag = None
    else:
        color = (0, 165, 255)       # Orange — tracking corps uniquement
        mode_tag = "BODY"

    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

    label = f"{person.name} #{person.track_id}"
    if person.confidence > 0:
        label += f" ({person.confidence * 100:.0f}%)"
    if mode_tag:
        label += f" [{mode_tag}]"

    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
    cv2.putText(display_frame, label, (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if pose:
        cv2.putText(display_frame, f"Pose: {pose}",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _open_camera():
    """
    Ouvre la caméra avec un timeout court pour ne pas bloquer le thread.
    Retourne un objet VideoCapture ouvert, ou None si l'ouverture échoue.
    """
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)

    # Timeout de connexion et de lecture (ms). Sans ça, VideoCapture sur HTTP
    # peut bloquer jusqu'à 30s sans aucun log — invisible pour l'utilisateur.
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.CAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.CAM_READ_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DISPLAY_HEIGHT)

    if not cap.isOpened():
        cap.release()
        return None
    print(f"[CAM] Caméra ouverte : {int(cap.get(3))}x{int(cap.get(4))}")
    return cap


def _reconnect_camera(cap):
    """
    Tente de rouvrir la caméra avec exponential backoff (1s → 30s max).
    Bloque jusqu'à la reconnexion ou l'arrêt de l'application.
    Retourne un nouveau VideoCapture ouvert, ou None si state.running devient False.
    """
    if cap is not None:
        cap.release()

    delay = 1.0
    attempt = 0
    while state.running:
        attempt += 1
        print(f"[CAM] Reconnexion tentative #{attempt} dans {delay:.0f}s...")
        time.sleep(delay)

        new_cap = _open_camera()
        if new_cap is not None:
            print(f"[CAM] Reconnexion réussie après {attempt} tentative(s) ✅")
            return new_cap

        # Exponential backoff plafonné à 30s
        delay = min(delay * 2, 30.0)

    return None


def processing_loop():
    print("[CAM] Thread de traitement démarré.")
    cap = _open_camera()
    if cap is None:
        print(f"[ERREUR] Impossible d'ouvrir la caméra au démarrage : {config.CAMERA_SOURCE}")
        # Attendre une reconnexion plutôt que de mourir silencieusement
        cap = _reconnect_camera(None)
        if cap is None:
            return

    frame_count = 0
    fps_time = time.time()
    fps_counter = 0
    consecutive_failures = 0
    persons_cache = []
    pose_cache = {}

    while state.running:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures == 1:
                print("[CAM] Frame perdue...")
            elif consecutive_failures >= config.CAM_MAX_FAILURES:
                # Trop d'échecs consécutifs → vraie déconnexion, tenter une reconnexion
                print(f"[CAM] {consecutive_failures} échecs consécutifs — déconnexion détectée.")
                cap = _reconnect_camera(cap)
                if cap is None:
                    break
                consecutive_failures = 0
            else:
                time.sleep(0.05)
            continue

        consecutive_failures = 0

        frame_count += 1
        fps_counter += 1
        display_frame = frame.copy()

        # ── Pipeline Body-First (YOLO + DeepSORT + InsightFace) ──────────────
        # FaceBodyTracker gère en interne sa propre cadence pour InsightFace
        # (toutes les FACE_RECOGNITION_SKIP frames). YOLO tourne à chaque appel.
        try:
            persons_cache = tracker.update(frame, frame_count)
        except Exception as e:
            print(f"[ERREUR TRAITEMENT] {type(e).__name__}: {e}")

        # ── Pose estimation (toutes les FRAME_SKIP frames, sur le crop corps) ─
        if frame_count % config.FRAME_SKIP == 0:
            pose_cache = {
                p.track_id: _estimate_pose_safe(frame, p.body_bbox, p.track_id)
                for p in persons_cache
            }

        # ── Dessin + construction de la liste de présence ────────────────────
        present_list = []
        for person in persons_cache:
            pose = pose_cache.get(person.track_id)
            _draw_person(display_frame, person, pose, frame_count)
            present_list.append({
                'name': person.name,
                'track_id': person.track_id,
                'confidence': person.confidence,
                'pose': pose,
                'last_face_frame': person.last_face_frame,
            })

        # ── FPS ──────────────────────────────────────────────────────────────
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()
        else:
            current_fps = state.fps

        cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
        cv2.putText(display_frame, f"Personnes: {len(present_list)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 245, 160), 2)

        # ── Mise à jour de l'état global ─────────────────────────────────────
        with state.lock:
            _, buffer = cv2.imencode(
                '.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
            )
            state.current_frame = buffer.tobytes()
            state.currently_present = present_list
            state.fps = current_fps

    tracker.release()
    cap.release()
    print("[CAM] Caméra fermée")


# ══════════════════════════════════════════
# ROUTES FLASK
# ══════════════════════════════════════════
def generate_frames():
    # Cadence navigateur — bornée par config.MJPEG_FPS_LIMIT
    interval = 1.0 / max(1, config.MJPEG_FPS_LIMIT)
    while True:
        with state.lock:
            frame = state.current_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(interval)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/enroll', methods=['POST'])
def enroll():
    """
    Enrôle une personne à chaud sans redémarrer l'application.

    Entrée (multipart/form-data) :
        name  : str — identifiant de la personne (ex: "Armand_Lauener")
        images: fichier(s) image JPG/PNG — au moins 1, idéalement 3-5

    Réponse :
        200 { success: true,  message: str, total_known: int }
        400 { success: false, message: str }
    """
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': 'Champ "name" manquant ou vide.'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'success': False, 'message': 'Aucune image fournie (champ "images").'}), 400

    images = []
    for f in files:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if not images:
        return jsonify({'success': False, 'message': 'Impossible de décoder les images reçues.'}), 400

    success, message = face_recognizer.enroll(name, images)

    if success:
        with state.lock:
            state.total_known = len(face_recognizer.known_names)
        return jsonify({'success': True, 'message': message,
                        'total_known': state.total_known}), 200

    return jsonify({'success': False, 'message': message}), 422


@app.route('/status')
def status():
    with state.lock:
        return jsonify({
            'currently_present': state.currently_present,
            'fps': state.fps,
            'total_known': state.total_known,
        })


# ══════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════
if __name__ == '__main__':
    process_thread = threading.Thread(target=processing_loop, daemon=True)
    process_thread.start()
    print(f"[SERVER] Démarrage sur http://{config.FLASK_HOST}:{config.FLASK_PORT}")

    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=config.DEBUG_MODE, threaded=True)
