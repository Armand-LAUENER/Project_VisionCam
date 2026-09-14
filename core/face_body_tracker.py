"""
face_body_tracker.py — Pipeline Body-First avec keypoints YOLO-Pose.

Architecture :
  1. YOLOv8-Pose → détection corps + keypoints squelette COCO (17 pts)
  2. DeepSORT    → tracking corps + ReID apparence (chaque frame)
                   Les keypoints sont transmis via `others` du tuple DeepSORT.
  3. InsightFace → crop dynamique centré sur les keypoints faciaux visibles
                   (COCO 0-4). Sans aucun keypoint visible, le crop retombe
                   sur le haut de la bbox corps : InsightFace tourne quand même.
  4. Association → face_center ↔ nose_keypoint (proximité euclidienne)
  5. Persistance → identity_map[track_id] conserve le nom même sans visage visible
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from collections import deque

import numpy as np
from ultralytics import YOLO

import config
from core.face_recognition import FaceRecognizer
from core.tracker_backends import build_body_tracker

logger = logging.getLogger(__name__)

# 384x640 : la taille letterbox d'une image 16:9 à 640 px de large. Un moteur
# 640x640 ajoute du padding qui change les détections (cf. README).
YOLO_ENGINE_EXPORT_COMMAND = (
    "uv run yolo export model=yolov8s-pose.pt format=engine half=True device=0 imgsz=384,640"
)


# IoU minimale pour qu'une piste soit considérée comme détectée sur l'image.
TRACK_DETECTION_MIN_IOU = 0.3


def match_tracks_to_detections(track_boxes: list, detection_boxes: list,
                               min_iou: float = TRACK_DETECTION_MIN_IOU) -> dict[int, int]:
    """Appariement un-pour-un piste → détection, par IoU décroissante.

    `track_boxes` en (x1, y1, x2, y2), `detection_boxes` en (x, y, w, h).
    Retourne {indice de piste: indice de détection}. Une détection ne sert
    qu'une piste : sa vraie piste (IoU proche de 1) l'emporte sur une piste en
    roue libre dont la boîte prédite la chevauche. Une piste absente du
    résultat n'a pas été détectée sur cette image.
    """
    pairs = []
    for i, (tx1, ty1, tx2, ty2) in enumerate(track_boxes):
        for j, (dx, dy, dw, dh) in enumerate(detection_boxes):
            iw = min(tx2, dx + dw) - max(tx1, dx)
            ih = min(ty2, dy + dh) - max(ty1, dy)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            union = (tx2 - tx1) * (ty2 - ty1) + dw * dh - inter
            iou = inter / union if union > 0 else 0.0
            if iou > min_iou:
                pairs.append((iou, i, j))
    matches: dict[int, int] = {}
    used_detections = set()
    for _iou, i, j in sorted(pairs, reverse=True):
        if i not in matches and j not in used_detections:
            matches[i] = j
            used_detections.add(j)
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

class TrackedPerson:
    """
    Résultat final pour une personne trackée à un instant donné.
    Contient les infos de position (corps) et d'identité (visage).
    """
    __slots__ = ('track_id', 'body_bbox', 'name', 'confidence', 'last_face_frame',
                 'pose_kps')

    def __init__(self, track_id: int, body_bbox: list[int],
                 name: str = "Inconnu", confidence: float = 0.0,
                 last_face_frame: int = -1, pose_kps: list | None = None) -> None:
        self.track_id = track_id
        self.body_bbox = body_bbox
        self.name = name
        self.confidence = confidence
        self.last_face_frame = last_face_frame
        # Keypoints COCO 0-6 du dernier match YOLO, pour POSE_SOURCE="yolo".
        self.pose_kps = pose_kps


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

class FaceBodyTracker:
    """
    Orchestre le pipeline Body-First avec keypoints YOLO-Pose.

    Usage:
        tracker = FaceBodyTracker(FaceRecognizer(...))
        for frame_count, frame in enumerate(camera):
            persons = tracker.update(frame, frame_count)
            for p in persons:
                draw_box(frame, p.body_bbox, p.name)
    """

    # Lissage temporel : consensus requis avant de confirmer une identité.
    # VOTE_WINDOW=3, majorité = 2/3 → ≈ 100ms à 30 FPS.
    VOTE_WINDOW = 3

    def __init__(self, face_recognizer: FaceRecognizer) -> None:
        logger.info("Chargement YOLO-Pose (%s)...", config.YOLO_MODEL)
        # Sans ce contrôle, ultralytics tente de télécharger le .engine absent
        # et échoue avec un message qui ne parle pas de TensorRT.
        if config.YOLO_MODEL.endswith(".engine") and not os.path.exists(config.YOLO_MODEL):
            raise FileNotFoundError(
                f"Moteur TensorRT introuvable : {config.YOLO_MODEL}. Il se construit sur "
                f"la machine qui l'utilise : {YOLO_ENGINE_EXPORT_COMMAND}"
            )
        self.yolo = YOLO(config.YOLO_MODEL)
        logger.info("YOLO-Pose chargé")

        # deep_sort_realtime ou deepsort-rs selon config.TRACKER_BACKEND.
        self.body_tracker = build_body_tracker()

        self.face_recognizer = face_recognizer

        # { track_id: { 'name': str, 'confidence': float, 'last_face_frame': int } }
        self._identity_map: dict[int, dict] = {}

        # { track_id: deque([(name, confidence), ...], maxlen=VOTE_WINDOW) }
        self._vote_buffer: dict[int, deque] = {}

        # { track_id: frame_count } — dernière soumission à InsightFace.
        # Fait tourner les tracks quand ils sont plus nombreux que les places.
        self._last_attempt_frame: dict[int, int] = {}

        # { track_id: (nose_x, nose_y, nose_conf) }
        # Mis à jour par IoU matching YOLO↔DeepSORT chaque frame.
        # Contourne le fait que track.others n'est pas fiable dans DeepSORT 1.3.x.
        self._nose_map: dict[int, tuple] = {}

        # { track_id: list[(x, y, conf) × 5] } — keypoints COCO 0-4 (nose, eyes, ears)
        self._face_kps_map: dict[int, list] = {}

        # { track_id: list[(x, y, conf) × 7] } — COCO 0-6, ajoute les épaules.
        # Alimente l'estimation d'orientation quand POSE_SOURCE="yolo".
        self._pose_kps_map: dict[int, list] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Point d'entrée public
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, frame_count: int) -> list[TrackedPerson]:
        """Pipeline complet pour une frame."""

        # Étape 1 : Détection YOLO-Pose (corps + keypoints)
        body_detections = self._detect_bodies(frame)

        # Étape 2 : Tracking DeepSORT
        raw_tracks = self.body_tracker.update(body_detections, frame)
        active_tracks = [t for t in raw_tracks if t.is_confirmed()]

        # Étape 2b : Mise à jour du nose_map par IoU matching YOLO↔tracks
        # track.others n'est pas fiable dans DeepSORT 1.3.x → on maintient
        # notre propre dict { track_id: (nx, ny, nc) }.
        visible_ids = self._update_nose_map(active_tracks, body_detections)

        # Une piste confirmée mais non détectée sur cette image est en roue
        # libre (personne occultée ou sortie) : sa boîte n'est qu'une
        # prédiction. Elle garde son identité en mémoire jusqu'à max_age, mais
        # n'est ni affichée ni soumise à la reconnaissance, dont le crop
        # montrerait l'obstacle. Mesuré sur MOT17 (validation, cf. README) :
        # MOTA 21,5 → 44,2 %, IDF1 48,3 → 54,4 %.
        visible_tracks = [t for t in active_tracks if t.track_id in visible_ids]

        # Étapes 3 & 4 : Reconnaissance faciale intelligente (cadencée)
        if frame_count % config.FACE_RECOGNITION_SKIP == 0 and visible_tracks:

            tracks_to_recognize = []
            for track in visible_tracks:
                identity = self._identity_map.get(track.track_id, {})
                name = identity.get('name', 'Inconnu')
                last_face_frame = identity.get('last_face_frame', -1)

                if name == 'Inconnu' or (frame_count - last_face_frame > config.FACE_FRESHNESS_FRAMES):
                    tracks_to_recognize.append(track)

            # Max 2 InsightFace/frame pour garantir les FPS. La tentative la plus
            # ancienne passe en premier (jamais tenté = -1) : sinon deux tracks
            # qui restent « Inconnu » monopolisent les deux places.
            tracks_to_recognize.sort(
                key=lambda t: self._last_attempt_frame.get(t.track_id, -1))
            tracks_to_recognize = tracks_to_recognize[:2]
            for track in tracks_to_recognize:
                self._last_attempt_frame[track.track_id] = frame_count

            if tracks_to_recognize:
                faces = self._recognize_faces_for_tracks(frame, tracks_to_recognize)
                self._associate_faces_to_tracks(visible_tracks, faces, frame_count)

        # Étape 5 : Construire les résultats
        results = self._build_results(visible_tracks)

        # Nettoyage : purger les entrées des tracks disparus. Les pistes en
        # roue libre restent : leur identité doit survivre à l'occlusion.
        active_ids = {t.track_id for t in active_tracks}
        self._identity_map = {tid: v for tid, v in self._identity_map.items() if tid in active_ids}
        self._vote_buffer  = {tid: v for tid, v in self._vote_buffer.items()  if tid in active_ids}
        self._last_attempt_frame = {tid: v for tid, v in self._last_attempt_frame.items()
                                    if tid in active_ids}
        # _nose_map et _face_kps_map sont déjà purgés dans _update_nose_map

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 2b — Mise à jour du nose_map (bypass track.others)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_nose_map(self, active_tracks: list, body_detections: list) -> set:
        """
        Associe chaque track confirmé à sa détection YOLO (IoU, un-pour-un) et
        met à jour ses keypoints. Retourne les identifiants des tracks
        détectés sur cette image.

        Nécessaire car track.others n'est pas propagé de façon fiable par
        deep_sort_realtime 1.3.x pour les tracks confirmés. L'appariement
        un-pour-un empêche une piste en roue libre de récupérer les keypoints
        de la personne dont sa boîte prédite chevauche la détection.
        """
        matches = match_tracks_to_detections(
            [track.to_ltrb() for track in active_tracks],
            [det_bbox for det_bbox, *_ in body_detections],
        )
        visible_ids = set()
        for track_index, det_index in matches.items():
            track_id = active_tracks[track_index].track_id
            det_others = body_detections[det_index][3]
            visible_ids.add(track_id)
            if det_others.get('nose') is not None:
                self._nose_map[track_id] = det_others['nose']
            if det_others.get('face_kps') is not None:
                self._face_kps_map[track_id] = det_others['face_kps']
            if det_others.get('pose_kps') is not None:
                self._pose_kps_map[track_id] = det_others['pose_kps']
        # Sans match, un track garde ses keypoints précédents (roue libre).

        # Purger les tracks disparus
        active_ids = {t.track_id for t in active_tracks}
        self._nose_map    = {tid: v for tid, v in self._nose_map.items()    if tid in active_ids}
        self._face_kps_map = {tid: v for tid, v in self._face_kps_map.items() if tid in active_ids}
        self._pose_kps_map = {tid: v for tid, v in self._pose_kps_map.items() if tid in active_ids}
        return visible_ids

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 1 — Détection YOLO-Pose
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_bodies(self, frame: np.ndarray) -> list[tuple]:
        """
        Détecte les personnes avec YOLOv8-Pose et formate les résultats pour DeepSORT.

        Chaque détection est un 4-tuple : ([x, y, w, h], conf, 'person', others).
        `others` contient le keypoint Nez sous la forme (x, y, conf) ou None.
        DeepSORT stocke `others` dans track.others → disponible dans les étapes suivantes.

        Returns:
            Liste de 4-tuples au format DeepSORT avec keypoints embarqués.
        """
        yolo_results = self.yolo.predict(
            frame,
            classes=[0],
            conf=config.YOLO_CONF_THRESHOLD,
            device=0,
            verbose=False,
        )

        detections = []
        for result in yolo_results:
            detections.extend(self._parse_result(result))

        return detections

    @staticmethod
    def _parse_result(result) -> list[tuple]:
        """Convertit un résultat YOLO-Pose en détections au format DeepSORT.

        Les tenseurs sont rapatriés du GPU en une seule copie par champ. Lire
        chaque coordonnée avec float() forçait une synchronisation GPU par
        valeur, soit ~40 par personne : 16 ms par frame à 8 personnes, plus que
        l'inférence YOLO elle-même, contre 0,3 ms ainsi.
        """
        boxes = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        # Keypoints COCO 0-6 (nez, yeux, oreilles, épaules), (N, 7, 3) : (x, y, conf).
        kps_rows: list = []
        if result.keypoints is not None:
            kps_rows = result.keypoints.data[:, :7].cpu().tolist()

        detections = []
        for i, ((x1, y1, x2, y2), conf) in enumerate(zip(boxes, confs)):
            nose = face_kps = pose_kps = None
            if i < len(kps_rows):
                pose_kps = [tuple(kp) for kp in kps_rows[i]]
                nose = pose_kps[0]
                # 0-4 seulement : sert au comptage des keypoints faciaux visibles.
                face_kps = pose_kps[:5]

            others = {'nose': nose, 'face_kps': face_kps, 'pose_kps': pose_kps}
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person', others))

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 3 — Reconnaissance faciale par crop dynamique centré sur le Nez
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_half_size(visible_face_kps: list, body_height: int) -> int:
        """Demi-taille du crop tête soumis à InsightFace.

        SCRFD ne détecte pas un visage qui remplit son entrée : le crop doit
        grandir avec le visage, pas avec le corps. L'écart maximal entre deux
        keypoints faciaux visibles mesure directement cette taille (cf.
        POSE_CROP_KP_SPAN_RATIO pour les mesures qui calent le facteur).

        Avec moins de deux keypoints il n'y a pas d'écart à mesurer : on
        retombe sur la hauteur du corps, moins fiable mais toujours là.
        """
        if len(visible_face_kps) >= 2:
            span = max(math.dist(a, b)
                       for a, b in itertools.combinations(visible_face_kps, 2))
            scaled = int(span * config.POSE_CROP_KP_SPAN_RATIO)
        else:
            scaled = int(body_height * config.POSE_CROP_BODY_RATIO)
        return max(config.POSE_CROP_HALF_SIZE, scaled)

    def _recognize_faces_for_tracks(
            self,
            frame: np.ndarray,
            tracks_to_recognize: list,
    ) -> list[dict]:
        """
        Pour chaque track, découpe un carré centré sur le centroïde des
        keypoints faciaux visibles (_face_kps_map) et le soumet à InsightFace.

        Avantages vs. crop "60% supérieur" :
          - Fonctionne quelle que soit la distance à la caméra.
          - Crop plus petit et plus précis → InsightFace plus rapide.

        Sans keypoint facial visible (dos tourné, très loin, occulté), le crop
        est centré sur le haut de la bbox corps. Ce cas n'est pas sauté : un
        visage petit ou lointain que YOLO ne repère pas reste reconnaissable.

        Returns:
            Liste de visages détectés avec bbox remises aux coordonnées globales
            et `source_track_id` indiquant le track d'origine.
        """
        all_faces = []
        h_img, w_img = frame.shape[:2]

        for track in tracks_to_recognize:
            bx1, by1, bx2, by2 = [int(v) for v in track.to_ltrb()]
            body_height = max(1, by2 - by1)

            # Centroïde des keypoints faciaux COCO 0-4 visibles (nez, yeux, oreilles).
            # Résistant aux lunettes et aux occlusions partielles : 1 keypoint suffit.
            crop_center = None
            visible: list[tuple[float, float]] = []
            face_kps = self._face_kps_map.get(track.track_id)
            if face_kps:
                visible = [(x, y) for x, y, c in face_kps if c >= config.POSE_NOSE_CONF_THRESHOLD]
                if len(visible) >= config.POSE_FACE_KP_MIN_VISIBLE:
                    cx = int(sum(x for x, y in visible) / len(visible))
                    cy = int(sum(y for x, y in visible) / len(visible))
                    crop_center = (cx, cy)
                    if len(visible) < 5:
                        logger.debug("track #%s : %d/5 keypoints faciaux visibles → centroïde",
                                     track.track_id, len(visible))

            # Fallback : aucun keypoint visible (personne de dos, très loin, très occulté)
            # → on estime la position de la tête depuis le haut de la bbox corps.
            if crop_center is None:
                cx = (bx1 + bx2) // 2
                cy = by1 + int(body_height * 0.15)
                crop_center = (cx, cy)
                logger.debug("track #%s : aucun keypoint facial → fallback body-top", track.track_id)

            cx, cy = crop_center
            half = self._crop_half_size(visible, body_height)
            crop_x1 = max(0, cx - half)
            crop_y1 = max(0, cy - half)
            crop_x2 = min(w_img, cx + half)
            crop_y2 = min(h_img, cy + half)

            if crop_x2 - crop_x1 < 20 or crop_y2 - crop_y1 < 20:
                continue

            head_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            try:
                # Un seul visage par crop : celui du centre, qui est celui du
                # track. Les visages voisins attrapés par le crop ne doivent
                # pas recevoir son source_track_id.
                face = self.face_recognizer.recognize_center_face(head_crop)
            except Exception as e:
                logger.warning("InsightFace erreur track #%s : %s: %s",
                               track.track_id, type(e).__name__, e)
                continue

            if face is None:
                continue

            fx1, fy1, fx2, fy2 = face['bbox']
            # Trop petit, l'embedding ne sépare plus les personnes : mieux vaut
            # ne rien décider que risquer un mauvais nom (cf. config).
            if min(fx2 - fx1, fy2 - fy1) < config.RECOGNITION_MIN_FACE_PX:
                continue
            # Remettre les coordonnées du crop à l'échelle de l'image globale
            face['bbox'] = [fx1 + crop_x1, fy1 + crop_y1, fx2 + crop_x1, fy2 + crop_y1]
            # Association directe : on sait déjà à quel track ce visage appartient
            face['source_track_id'] = track.track_id
            all_faces.append(face)

        return all_faces

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 4 — Association Visage → Corps (basée sur le nez)
    # ─────────────────────────────────────────────────────────────────────────

    def _associate_faces_to_tracks(
            self,
            active_tracks: list,
            faces: list[dict],
            frame_count: int,
    ) -> None:
        """
        Associe chaque visage reconnu à son track et met à jour _identity_map.

        Priorité d'association :
          1. source_track_id (défini au crop → association directe, sans calcul).
          2. Proximité euclidienne face_center ↔ nose_keypoint (fallback).

        Règles absolues :
          - 'Inconnu' ne remplace JAMAIS une identité établie.
          - Confiance < RECOGNITION_THRESHOLD → rejeté.
          - Anti-clonage : une identité ne peut appartenir qu'à un seul corps.
          - Vote buffer : consensus requis avant confirmation.
        """
        active_track_map = {t.track_id: t for t in active_tracks}

        for face in faces:
            if face['name'] == 'Inconnu':
                continue
            if face['confidence'] < config.RECOGNITION_THRESHOLD:
                continue

            # Association directe via source_track_id (chemin nominal)
            best_track_id = face.get('source_track_id')
            if best_track_id not in active_track_map:
                # Fallback géométrique si le track a disparu entre le crop et l'association
                best_track_id = self._find_containing_track(face['bbox'], active_tracks)

            if best_track_id is None:
                continue

            # ── Vote buffer ────────────────────────────────────────────────────
            if best_track_id not in self._vote_buffer:
                self._vote_buffer[best_track_id] = deque(maxlen=self.VOTE_WINDOW)
            self._vote_buffer[best_track_id].append((face['name'], face['confidence']))

            votes: dict[str, list[float]] = {}
            for name, conf in self._vote_buffer[best_track_id]:
                votes.setdefault(name, []).append(conf)

            top_name = max(votes, key=lambda n: len(votes[n]))
            top_count = len(votes[top_name])
            if top_count < (self.VOTE_WINDOW + 1) // 2:
                continue  # pas encore de consensus

            avg_confidence = sum(votes[top_name]) / top_count
            new_name = top_name

            # ── Anti-clonage ────────────────────────────────────────────────
            for old_tid, identity in list(self._identity_map.items()):
                if old_tid != best_track_id and identity.get('name') == new_name:
                    self._identity_map[old_tid] = {
                        'name': 'Inconnu', 'confidence': 0.0, 'last_face_frame': -1
                    }
                    logger.warning(
                        "Correction usurpation : '%s' passe du corps #%d au corps #%d",
                        new_name, old_tid, best_track_id,
                    )

            self._identity_map[best_track_id] = {
                'name': new_name,
                'confidence': avg_confidence,
                'last_face_frame': frame_count,
            }

    def _find_containing_track(
            self,
            face_bbox: list[int],
            active_tracks: list,
    ) -> int | None:
        """
        Fallback d'association : retourne le track dont le nez est le plus proche
        du centre de la bbox du visage.

        Seuil de proximité : 80% de la largeur du corps (tolérant aux imprécisions
        YOLO vs InsightFace sur la localisation du nez).
        """
        face_cx = (face_bbox[0] + face_bbox[2]) / 2
        face_cy = (face_bbox[1] + face_bbox[3]) / 2

        best_track_id = None
        best_distance = float('inf')

        for track in active_tracks:
            nose = self._nose_map.get(track.track_id)
            if nose is None or nose[2] < config.POSE_NOSE_CONF_THRESHOLD:
                continue

            nose_x, nose_y, _ = nose
            dist = ((face_cx - nose_x) ** 2 + (face_cy - nose_y) ** 2) ** 0.5

            bx1, by1, bx2, by2 = [int(v) for v in track.to_ltrb()]
            body_width = max(1, bx2 - bx1)
            max_dist = body_width * 0.8

            if dist < max_dist and dist < best_distance:
                best_distance = dist
                best_track_id = track.track_id

        return best_track_id

    # ─────────────────────────────────────────────────────────────────────────
    # Étape 5 — Construction du résultat
    # ─────────────────────────────────────────────────────────────────────────

    def _build_results(self, active_tracks: list) -> list[TrackedPerson]:
        """Assemble les TrackedPerson depuis les tracks actifs + l'identity_map."""
        results = []
        for track in active_tracks:
            tid = track.track_id
            x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
            identity = self._identity_map.get(tid, {})

            results.append(TrackedPerson(
                track_id=tid,
                body_bbox=[x1, y1, x2, y2],
                name=identity.get('name', 'Inconnu'),
                confidence=identity.get('confidence', 0.0),
                last_face_frame=identity.get('last_face_frame', -1),
                pose_kps=self._pose_kps_map.get(tid),
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Libère les ressources."""
        self._identity_map.clear()
        self._vote_buffer.clear()
        self._last_attempt_frame.clear()
        self._face_kps_map.clear()
        logger.info("Ressources libérées")
