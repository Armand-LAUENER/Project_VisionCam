from flask import Flask, render_template, Response, jsonify, request
import cv2
import logging
import numpy as np
import os
import queue
import threading
import time

import config
from core.face_recognition import FaceRecognizer
from core.face_body_tracker import FaceBodyTracker
from core.pose_estimation import PoseEstimator
from core.pose_from_keypoints import KeypointPoseEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ══════════════════════════════════════════
# INITIALISATION DES MODULES
# ══════════════════════════════════════════
logger.info("Chargement des modules...")
logger.info("Source vidéo : %s", "WEBCAM" if config.USE_LOCAL_CAM else config.REMOTE_SOURCE)

face_recognizer = FaceRecognizer(
    config.KNOWN_FACES_DIR,
    threshold=config.RECOGNITION_THRESHOLD,
    cache_path=config.EMBEDDINGS_CACHE_PATH,
)
tracker = FaceBodyTracker(face_recognizer)
if config.POSE_SOURCE == "yolo":
    pose_estimator = KeypointPoseEstimator()
elif config.POSE_SOURCE == "mediapipe":
    pose_estimator = PoseEstimator()
else:
    raise ValueError(
        f"POSE_SOURCE invalide : {config.POSE_SOURCE!r} — attendu 'mediapipe' ou 'yolo'."
    )
logger.info("Source d'orientation : %s", config.POSE_SOURCE)

logger.info("Visages connus : %s", list(face_recognizer.known_names))
logger.info("Modules chargés")


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
        # PRESENCE_TIMEOUT : { track_id: (person_dict, last_seen_timestamp) }
        self.last_seen: dict = {}

state = AppState()

# Queue inter-thread caméra → AI. Taille 2 : on garde toujours la frame la plus fraîche.
_frame_queue: queue.Queue = queue.Queue(maxsize=2)


# ══════════════════════════════════════════
# HELPERS PIPELINE
# ══════════════════════════════════════════
def _estimate_pose_safe(frame, person):
    """
    Calcule l'orientation d'une personne, avec log explicite en cas d'échec.

    Deux sources selon config.POSE_SOURCE : les keypoints YOLO déjà produits
    sur GPU, ou une inférence Mediapipe sur le crop du corps.
    """
    try:
        if config.POSE_SOURCE == "yolo":
            return pose_estimator.estimate(person.pose_kps)

        x1, y1, x2, y2 = person.body_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return pose_estimator.estimate(crop)
    except Exception as e:
        logger.warning("Échec pose sur track #%d : %s: %s",
                       person.track_id, type(e).__name__, e)
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
        color = (100, 100, 255)
        mode_tag = None
    elif frames_since_face <= config.FACE_FRESHNESS_FRAMES:
        color = (0, 245, 160)
        mode_tag = None
    else:
        color = (0, 165, 255)
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
    logger.info("Caméra ouverte : %dx%d", int(cap.get(3)), int(cap.get(4)))
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
        logger.warning("Reconnexion tentative #%d dans %.0fs...", attempt, delay)
        time.sleep(delay)

        new_cap = _open_camera()
        if new_cap is not None:
            logger.info("Reconnexion réussie après %d tentative(s)", attempt)
            return new_cap

        delay = min(delay * 2, 30.0)

    return None


# ══════════════════════════════════════════
# THREAD A — LECTURE CAMÉRA
# ══════════════════════════════════════════
def camera_loop():
    """
    Lit les frames depuis la caméra et les pousse dans _frame_queue.
    Entièrement découplé du pipeline AI : cap.read() ne bloque jamais
    le calcul YOLO/InsightFace. La queue conserve toujours la frame
    la plus récente (drop oldest si pleine).
    """
    logger.info("Thread caméra démarré.")
    cap = _open_camera()
    if cap is None:
        logger.error("Impossible d'ouvrir la caméra au démarrage : %s", config.CAMERA_SOURCE)
        cap = _reconnect_camera(None)
        if cap is None:
            return

    consecutive_failures = 0

    while state.running:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures == 1:
                logger.warning("Frame perdue...")
            elif consecutive_failures >= config.CAM_MAX_FAILURES:
                logger.error("%d échecs consécutifs — déconnexion détectée.", consecutive_failures)
                cap = _reconnect_camera(cap)
                if cap is None:
                    break
                consecutive_failures = 0
            else:
                time.sleep(0.05)
            continue

        consecutive_failures = 0

        # Toujours garder la frame la plus fraîche : drop oldest si queue pleine.
        try:
            _frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                _frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                _frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    cap.release()
    logger.info("Thread caméra arrêté.")


# ══════════════════════════════════════════
# THREAD B — PIPELINE AI
# ══════════════════════════════════════════
def processing_loop():
    """
    Consomme les frames de _frame_queue et applique le pipeline AI.
    Ne touche plus à la caméra — entièrement découplé de camera_loop.
    """
    logger.info("Thread de traitement démarré.")

    frame_count = 0
    fps_time = time.time()
    fps_counter = 0
    persons_cache = []
    pose_cache = {}

    while state.running:
        try:
            frame = _frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_count += 1
        fps_counter += 1
        display_frame = frame.copy()

        # ── Pipeline Body-First (YOLO + DeepSORT + InsightFace) ──────────────
        try:
            persons_cache = tracker.update(frame, frame_count)
        except Exception as e:
            logger.error("Erreur traitement : %s: %s", type(e).__name__, e)

        # ── Pose estimation (toutes les FRAME_SKIP frames, sur le crop corps) ─
        if frame_count % config.FRAME_SKIP == 0:
            pose_cache = {
                p.track_id: _estimate_pose_safe(frame, p)
                for p in persons_cache
            }

        # ── Dessin + construction de la liste de présence ────────────────────
        now = time.time()
        present_list = []
        present_ids = set()

        for person in persons_cache:
            pose = pose_cache.get(person.track_id)
            _draw_person(display_frame, person, pose, frame_count)
            entry = {
                'name': person.name,
                'track_id': person.track_id,
                'confidence': person.confidence,
                'pose': pose,
                'last_face_frame': person.last_face_frame,
            }
            present_list.append(entry)
            present_ids.add(person.track_id)
            state.last_seen[person.track_id] = (entry, now)

        # ── PRESENCE_TIMEOUT : réinjecter les personnes récemment vues ───────
        for tid, (person_entry, last_time) in list(state.last_seen.items()):
            if tid not in present_ids:
                if now - last_time <= config.PRESENCE_TIMEOUT:
                    present_list.append(person_entry)
                else:
                    del state.last_seen[tid]

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
    logger.info("Thread de traitement arrêté.")


# ══════════════════════════════════════════
# ROUTES FLASK
# ══════════════════════════════════════════
def generate_frames():
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
        name   : str — identifiant de la personne (ex: "Armand_Lauener")
        method : 'average' (défaut) | 'multitemplate'
        images : fichier(s) image JPG/PNG

    Réponse :
        200 { success: true,  message: str, total_known: int }
        400 { success: false, message: str }
        422 { success: false, message: str }  ← aucun visage trouvé
    """
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': 'Champ "name" manquant ou vide.'}), 400

    method = request.form.get('method', 'average').strip().lower()
    if method not in ('average', 'multitemplate'):
        return jsonify({'success': False,
                        'message': 'Champ "method" invalide : attendu "average" ou "multitemplate".'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'success': False, 'message': 'Aucune image fournie (champ "images").'}), 400

    decoded = []
    for f in files:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        label = os.path.splitext(f.filename)[0] if f.filename else f"img{len(decoded)}"
        decoded.append((label, img))

    if not decoded:
        return jsonify({'success': False, 'message': 'Impossible de décoder les images reçues.'}), 400

    if method == 'average':
        images = [img for _, img in decoded]
        success, message = face_recognizer.enroll_person_average(name, images)
    else:
        frames_dict = {label: img for label, img in decoded}
        success, message = face_recognizer.enroll_person_multitemplate(name, frames_dict)

    if success:
        with state.lock:
            state.total_known = len(face_recognizer.known_names)
        return jsonify({'success': True, 'message': message,
                        'total_known': state.total_known}), 200

    return jsonify({'success': False, 'message': message}), 422


@app.route('/capture', methods=['POST'])
def capture():
    """
    Enrôle une personne depuis la frame courante du flux caméra.

    Entrée (multipart/form-data) :
        name : str — identifiant de la personne

    Réponse :
        200 { success: true,  message: str, total_known: int }
        400 { success: false, message: str }
        503 { success: false, message: str }  ← pas de frame disponible
        422 { success: false, message: str }  ← aucun visage détecté
    """
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': 'Champ "name" manquant ou vide.'}), 400

    with state.lock:
        frame_bytes = state.current_frame

    if not frame_bytes:
        return jsonify({'success': False, 'message': 'Aucune frame disponible (caméra déconnectée ?)'}), 503

    # Décoder le JPEG actuellement streamé pour en extraire le visage
    buf = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    success, message = face_recognizer.enroll_person_average(name, [frame])

    if success:
        with state.lock:
            state.total_known = len(face_recognizer.known_names)
        return jsonify({'success': True, 'message': message,
                        'total_known': state.total_known}), 200

    return jsonify({'success': False, 'message': message}), 422


@app.route('/rebuild', methods=['POST'])
def rebuild():
    """
    Reconstruit la base d'embeddings depuis known_faces/ à chaud.
    Lance le rebuild dans un thread daemon pour ne pas bloquer la réponse HTTP.

    Réponse immédiate :
        202 { started: true, message: str }
    """
    def _do_rebuild():
        logger.info("Rebuild base embeddings demandé via UI...")
        face_recognizer.rebuild_database()
        with state.lock:
            state.total_known = len(face_recognizer.known_names)
        logger.info("Rebuild terminé : %d entrée(s)", state.total_known)

    threading.Thread(target=_do_rebuild, daemon=True).start()
    return jsonify({'started': True, 'message': 'Reconstruction lancée en arrière-plan.'}), 202


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
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    process_thread = threading.Thread(target=processing_loop, daemon=True)
    camera_thread.start()
    process_thread.start()
    logger.info("Démarrage sur http://%s:%d", config.FLASK_HOST, config.FLASK_PORT)

    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=config.DEBUG_MODE, threaded=True)
