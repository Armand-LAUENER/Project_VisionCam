"""
bench_face.py — Compare des variantes de la reconnaissance faciale : vitesse et noms.

La reconnaissance est l'étape la plus lourde du pipeline (~20 ms par image
traitée). Chaque variante combine :

- la taille d'entrée du détecteur SCRFD : les crops de tête font souvent
  moins de 300 px et étaient agrandis en 640×640 ;
- le moteur d'exécution de SCRFD et de glintr100 : CUDA (FP32) ou TensorRT FP16,
  via le TensorrtExecutionProvider d'onnxruntime.

Les crops sont ceux de l'application : YOLOv8-Pose (config.YOLO_MODEL) donne
les keypoints, FaceBodyTracker.head_crop_box le carré de tête, pour chaque
personne annotée appariée à une détection. Les identités viennent de la vérité
terrain, comme tools/recognition_threshold_study.py : enrôlement sur
`--gallery`, reconnaissance sur `--probe`, visages d'au moins
RECOGNITION_MIN_FACE_PX, seuil RECOGNITION_THRESHOLD.

Les variantes tournent en alternance par paquets de crops : un autre programme
qui charge le GPU pèse sur toutes de la même façon.

Depuis la racine du projet :

    uv run -m tools.bench_face \\
        --gallery ~/datasets/CHIRLA/mot/seq_004_camera_2 ~/datasets/CHIRLA/mot/seq_020_camera_4 \\
        --probe ~/datasets/CHIRLA/mot/seq_025_camera_2 ~/datasets/CHIRLA/mot/seq_026_camera_3
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import statistics
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np

import config
from tools import eval_mot
from tools.recognition_threshold_study import enroll, load_gt_boxes, outcomes

MODEL_DIR = os.path.expanduser(f"~/.insightface/models/{config.INSIGHTFACE_MODEL}")
TRT_CACHE_DIR = os.path.join(config.DATA_DIR, "trt_cache")
DEFAULT_VARIANTS = ["cuda:640/cuda", "cuda:320/cuda", "cuda:640/trt", "trt:640/trt", "trt:320/trt"]
CHUNK = 50


def _session(model_path, engine, input_name=None, shape=None, max_batch=1):
    import onnxruntime as ort

    if engine == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        os.makedirs(TRT_CACHE_DIR, exist_ok=True)
        dims = "x".join(str(v) for v in shape[1:])
        providers = [("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": TRT_CACHE_DIR,
            "trt_profile_min_shapes": f"{input_name}:1x{dims}",
            "trt_profile_opt_shapes": f"{input_name}:1x{dims}",
            "trt_profile_max_shapes": f"{input_name}:{max_batch}x{dims}",
        }), "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


@dataclass
class Variant:
    spec: str
    det_ms: list = field(default_factory=list)
    rec_ms: list = field(default_factory=list)
    results: list = field(default_factory=list)   # un par crop : (embedding, taille) ou None

    def __post_init__(self):
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.model_zoo.scrfd import SCRFD

        det, rec_engine = self.spec.split("/")
        det_engine, size = det.split(":")
        self.size = int(size)
        det_path = os.path.join(MODEL_DIR, "scrfd_10g_bnkps.onnx")
        rec_path = os.path.join(MODEL_DIR, "glintr100.onnx")
        self.detector = SCRFD(det_path, session=_session(
            det_path, det_engine, "input.1", (1, 3, self.size, self.size)))
        self.detector.prepare(0, input_size=(self.size, self.size), det_thresh=0.5)
        self.recognizer = ArcFaceONNX(rec_path, session=_session(
            rec_path, rec_engine, "input.1", (1, 3, 112, 112)))
        self.recognizer.prepare(0)

    def recognize(self, crop):
        """Même règle que FaceRecognizer.recognize_center_face, chronométrée."""
        start = time.perf_counter()
        bboxes, kpss = self.detector.detect(crop, max_num=0, metric="default")
        self.det_ms.append(1000 * (time.perf_counter() - start))
        if bboxes.shape[0] == 0 or kpss is None:
            return None
        h, w = crop.shape[:2]
        centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2
        idx = int(np.argmin(np.sum((centers - (w / 2, h / 2)) ** 2, axis=1)))
        face = SimpleNamespace(kps=kpss[idx], embedding=None)
        start = time.perf_counter()
        self.recognizer.get(crop, face)
        self.rec_ms.append(1000 * (time.perf_counter() - start))
        x1, y1, x2, y2 = bboxes[idx, :4]
        return face.embedding / np.linalg.norm(face.embedding), float(min(x2 - x1, y2 - y1))


def extract_crops(yolo, seq_dir, step):
    """[(identité, image, crop de tête)] des personnes annotées détectées par YOLO."""
    from core.face_body_tracker import FaceBodyTracker, match_tracks_to_detections

    crops = []
    gt = load_gt_boxes(seq_dir)
    for frame in range(1, eval_mot.sequence_length(seq_dir) + 1, step):
        if not gt.get(frame):
            continue
        image = cv2.imread(eval_mot.frame_path(seq_dir, frame))
        result = yolo.predict(image, classes=[0], conf=config.YOLO_CONF_THRESHOLD,
                              device=0, verbose=False)[0]
        detections = FaceBodyTracker._parse_result(result)
        gt_boxes = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in gt[frame]]
        matches = match_tracks_to_detections(gt_boxes, [d[0] for d in detections], min_iou=0.5)
        for gt_index, det_index in matches.items():
            (x, y, w, h), _, _, others = detections[det_index]
            box = FaceBodyTracker.head_crop_box(image.shape, (x, y, x + w, y + h),
                                                others["face_kps"])
            if box is None:
                continue
            cx1, cy1, cx2, cy2 = box
            crops.append((gt[frame][gt_index][0], frame, image[cy1:cy2, cx1:cx2].copy()))
    return crops


def ms(values):
    if not values:
        return "—"
    return f"{statistics.median(values):5.1f} / {np.percentile(values, 95):5.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gallery", nargs="+", required=True, help="séquences d'enrôlement")
    parser.add_argument("--probe", nargs="+", required=True, help="séquences de test")
    parser.add_argument("--step", type=int, default=15, help="une image sur N")
    parser.add_argument("--photos", type=int, default=5, help="photos par personne enrôlée")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="moteur_détecteur:taille/moteur_reconnaissance, moteurs cuda ou trt")
    args = parser.parse_args()

    from ultralytics import YOLO

    yolo = YOLO(config.YOLO_MODEL, task="pose")
    gallery_crops = [c for s in args.gallery
                     for c in extract_crops(yolo, os.path.expanduser(s), args.step)]
    probe_crops = [c for s in args.probe
                   for c in extract_crops(yolo, os.path.expanduser(s), args.step)]
    crops = gallery_crops + probe_crops
    sizes = [max(c[2].shape[:2]) for c in crops]
    print(f"{len(gallery_crops)} crops d'enrôlement, {len(probe_crops)} de test — "
          f"côté médian {statistics.median(sizes):.0f} px (5e-95e centile "
          f"{np.percentile(sizes, 5):.0f}-{np.percentile(sizes, 95):.0f})")

    print("Chargement des variantes (premier lancement TensorRT : construction des moteurs)…")
    variants = [Variant(spec) for spec in args.variants]
    for variant in variants:   # préchauffage, hors mesure
        for _, _, crop in crops[:20]:
            variant.recognize(crop)
        variant.det_ms.clear()
        variant.rec_ms.clear()

    for start in range(0, len(crops), CHUNK):
        for variant in variants:
            variant.results += [variant.recognize(crop) for _, _, crop in crops[start:start + CHUNK]]

    reference = variants[0]
    threshold = config.RECOGNITION_THRESHOLD
    min_face = config.RECOGNITION_MIN_FACE_PX
    print(f"\nSeuil {threshold}, visages d'au moins {min_face} px. "
          f"Temps en ms, médiane / 95e centile, par crop.")
    print(f"{'variante':<14} │ {'détection':>13} {'embedding':>13} │ {'test':>5} "
          f"{'bon nom':>8} {'mauvais':>8} {'Inconnu':>8} │ {'cos vs réf.':>13}")
    n_gallery = len(gallery_crops)
    for variant in variants:
        faces = [(crop[0], crop[1], r[0], r[1]) if r and r[1] >= min_face else None
                 for crop, r in zip(crops, variant.results)]
        gallery = enroll([f for f in faces[:n_gallery] if f], args.photos)
        probes = [f for f in faces[n_gallery:] if f]
        _, _, rows = outcomes(gallery, probes, [threshold])
        _, right, wrong, rejected, _ = rows[0]
        similarity = [float(a[0] @ b[0]) for a, b in zip(reference.results, variant.results)
                      if a and b]
        print(f"{variant.spec:<14} │ {ms(variant.det_ms):>13} {ms(variant.rec_ms):>13} │ "
              f"{len(probes):>5} {right:>8} {wrong:>8} {rejected:>8} │ "
              f"{np.median(similarity):.3f} / {np.percentile(similarity, 5):.3f}")
    print("\ntest : visages de test d'au moins RECOGNITION_MIN_FACE_PX ; bon nom, mauvais, "
          "Inconnu : nombres de visages.\ncos vs réf. : similarité avec la première variante "
          "(médiane / 5e centile), sur les crops où les deux trouvent un visage.")


if __name__ == "__main__":
    main()
