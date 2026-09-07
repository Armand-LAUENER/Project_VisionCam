"""webcam_bridge.py — Expose une webcam USB en flux MJPEG sur le réseau local.

À LANCER CÔTÉ WINDOWS, pas dans WSL :

    py -3.11 tools\\webcam_bridge.py

WSL2 n'a aucun accès aux périphériques USB — la webcam n'existe que pour
Windows. Ce pont la lit côté Windows et la rediffuse en HTTP ; l'application
la consomme ensuite comme une caméra IP via REMOTE_SOURCE dans .env.

La capture tourne dans un thread dédié qui ne conserve que la dernière frame :
un client lent ne peut donc pas faire accumuler de retard sur la webcam.
"""

import argparse
import logging
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

logger = logging.getLogger("webcam_bridge")

BOUNDARY = "visioncam-frame"


class CameraStream:
    """Lecture continue de la webcam dans un thread, dernière frame disponible."""

    def __init__(self, index, width, height, fps):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la webcam d'index {index}. "
                f"Vérifie qu'aucune autre application ne l'utilise, "
                f"ou essaie un autre index (--index 1)."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def resolution(self):
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _loop(self):
        failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures >= 30:
                    logger.error("30 lectures consécutives échouées, arrêt de la capture.")
                    self._running = False
                continue

            failures = 0
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


class MJPEGHandler(BaseHTTPRequestHandler):
    """Sert /video (flux MJPEG), /snapshot (une image) et / (page de test)."""

    camera = None
    jpeg_quality = 80
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.client_address[0], fmt % args)

    def _encode(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes() if ok else None

    def do_GET(self):
        if self.path.startswith("/video"):
            self._serve_stream()
        elif self.path.startswith("/snapshot"):
            self._serve_snapshot()
        elif self.path in ("/", "/index.html"):
            self._serve_index()
        else:
            self.send_error(404)

    def _serve_index(self):
        page = (
            b"<html><body style='margin:0;background:#111'>"
            b"<img src='/video' style='width:100%'>"
            b"</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def _serve_snapshot(self):
        frame = self.camera.read()
        if frame is None:
            self.send_error(503, "Aucune frame disponible")
            return

        jpeg = self._encode(frame)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        logger.info("Client connecté : %s", self.client_address[0])
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue

                jpeg = self._encode(frame)
                if jpeg is None:
                    continue

                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Déconnexion normale du client (onglet fermé, app arrêtée).
            logger.info("Client déconnecté : %s", self.client_address[0])


def local_ips():
    """Adresses IPv4 de la machine, pour afficher des URLs utilisables."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    return sorted(ips)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, default=0, help="Index de la webcam (défaut : 0)")
    parser.add_argument("--host", default="0.0.0.0", help="Interface d'écoute (défaut : 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port HTTP (défaut : 8080)")
    parser.add_argument("--width", type=int, default=1280, help="Largeur demandée (défaut : 1280)")
    parser.add_argument("--height", type=int, default=720, help="Hauteur demandée (défaut : 720)")
    parser.add_argument("--fps", type=int, default=30, help="FPS demandés (défaut : 30)")
    parser.add_argument("--quality", type=int, default=80, help="Qualité JPEG 1-100 (défaut : 80)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    camera = CameraStream(args.index, args.width, args.height, args.fps)
    width, height = camera.resolution
    logger.info("Webcam %d ouverte en %dx%d", args.index, width, height)

    MJPEGHandler.camera = camera
    MJPEGHandler.jpeg_quality = args.quality
    server = ThreadingHTTPServer((args.host, args.port), MJPEGHandler)
    server.daemon_threads = True

    logger.info("Flux MJPEG disponible sur :")
    for ip in local_ips():
        logger.info("    http://%s:%d/video", ip, args.port)
    logger.info("Ctrl+C pour arrêter.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé.")
    finally:
        server.shutdown()
        camera.stop()


if __name__ == "__main__":
    main()
