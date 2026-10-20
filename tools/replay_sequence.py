"""
replay_sequence.py — Rejoue une séquence MOT dans le pipeline de l'application.

Fait passer une séquence annotée (dossier MOT17 avec `img1/`) par le pipeline
complet (`core.frame_pipeline.FramePipeline` : YOLOv8-Pose → DeepSORT →
InsightFace, vote par piste), sans Flask ni dessin, et écrit le journal des
pistes affichées au format `TRACKS_LOG_PATH` — celui qu'attend
`tools/eval_identity.py`. C'est le chemin mono-processus de l'application, isolé
pour la mesure.

**Cadence fixe.** La cadence de reconnaissance n'est pas ajustée en marche
(`tuning=None`) : chaque piste garde `--skip` (`FACE_RECOGNITION_SKIP`) tout du
long. Deux rejeux de la même séquence produisent donc le même journal, ce que
la cadence adaptative ne garantit pas. C'est la condition pour comparer deux
versions du code (avant/après un correctif) ou deux réglages.

Enrôler sur d'autres séquences avec `tools/enroll_from_sequence.py`, dans un
dossier à part (jamais le `known_faces/` de l'application), puis :

    D=~/datasets/CHIRLA/mot
    uv run -m tools.enroll_from_sequence --output /tmp/known_faces_chirla \\
        $D/seq_004_camera_2 $D/seq_020_camera_4
    uv run -m tools.replay_sequence $D/seq_026_camera_3 \\
        --known-faces /tmp/known_faces_chirla --output /tmp/tracks_026.csv --skip 2
    uv run -m tools.eval_identity --tracks /tmp/tracks_026.csv \\
        --sequence $D/seq_026_camera_3 --known-faces /tmp/known_faces_chirla
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequence", help="dossier de séquence MOT17 (img1/ + seqinfo.ini)")
    parser.add_argument("--output", required=True, help="journal de pistes à écrire (CSV)")
    parser.add_argument("--known-faces", help="galerie d'enrôlement (format known_faces/)")
    parser.add_argument("--skip", type=int,
                        help="FACE_RECOGNITION_SKIP (cadence fixe ; défaut : config)")
    parser.add_argument("--cache", help="cache d'embeddings de la galerie "
                        "(défaut : <galerie>.npz ; jamais celui de l'application)")
    args = parser.parse_args()

    # Fixer l'environnement avant d'importer config (qui le lit à l'import).
    os.environ["VIDEO_MODE"] = "every_frame"
    os.environ["MJPEG_ANNOTATE"] = "false"
    os.environ["VIDEO_FILE"] = os.path.expanduser(args.sequence)
    os.environ["TRACKS_LOG_PATH"] = os.path.expanduser(args.output)
    if args.known_faces:
        os.environ["KNOWN_FACES_DIR"] = os.path.expanduser(args.known_faces)
    if args.skip is not None:
        os.environ["FACE_RECOGNITION_SKIP"] = str(args.skip)

    import config
    from core.face_body_tracker import FaceBodyTracker
    from core.face_recognition import FaceRecognizer
    from core.frame_pipeline import FramePipeline, build_pose_estimator
    from core.video_source import FileSource

    # Cache propre à la galerie : ne jamais réutiliser celui de l'application
    # (data/embeddings.npz), qui correspond à un autre dossier known_faces/.
    cache = os.path.expanduser(args.cache) if args.cache \
        else config.KNOWN_FACES_DIR.rstrip("/") + ".npz"

    recognizer = FaceRecognizer(config.KNOWN_FACES_DIR,
                                threshold=config.RECOGNITION_THRESHOLD, cache_path=cache)
    tracker = FaceBodyTracker(recognizer)
    pipeline = FramePipeline(config.CAMERA_ID, tracker, record=lambda *a: None,
                             pose_estimator=build_pose_estimator(),
                             tracks_path=config.TRACKS_LOG_PATH, tuning=None)
    source = FileSource(config.VIDEO_FILE, mode=config.VIDEO_MODE)
    print(f"Rejeu {os.path.basename(config.VIDEO_FILE.rstrip('/'))} "
          f"(skip={tracker.recognition_skip}, galerie={config.KNOWN_FACES_DIR}) "
          f"→ {config.TRACKS_LOG_PATH}", flush=True)

    frames = 0
    while True:
        ok, frame = source.read()
        if not ok:
            break
        pipeline.process(0.0, frame)
        frames += 1
        if frames % 1000 == 0:
            print(f"  {frames} images", flush=True)
    pipeline.close()
    source.release()
    print(f"Terminé : {frames} images écrites dans {config.TRACKS_LOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
