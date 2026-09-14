"""webcam_bridge.py — Expose une webcam USB en flux MJPEG sur le réseau local.

À LANCER CÔTÉ WINDOWS, pas dans WSL :

    py -3.11 tools\\webcam_bridge.py

WSL2 n'a aucun accès aux périphériques USB — la webcam n'existe que pour
Windows. Ce pont la lit côté Windows et la rediffuse en HTTP ; l'application
la consomme ensuite comme une caméra IP via REMOTE_SOURCE dans .env.

La capture tourne dans un thread dédié qui encode chaque image une seule fois ;
les clients reçoivent chaque nouvelle image une fois, et un client lent saute
des images au lieu d'accumuler du retard. La webcam est ouverte en MJPG pour
limiter la bande passante USB (cf. CameraStream).
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
    """Lecture continue de la webcam dans un thread ; chaque image est encodée une fois.

    Les clients attendent l'image suivante (wait_jpeg) au lieu de renvoyer la
    dernière en boucle. Avant, le flux ré-encodait et renvoyait la même image
    sans pause : 556 images/s envoyées pour 30 réelles, un cœur du processeur
    Windows occupé en permanence, et une application qui traitait des doublons.
    """

    def __init__(self, index, width, height, fps, jpeg_quality=80, capture=None):
        if capture is None:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            capture = cv2.VideoCapture(index, backend)
        self.cap = capture
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la webcam d'index {index}. "
                f"Vérifie qu'aucune autre application ne l'utilise, "
                f"ou essaie un autre index (--index 1)."
            )

        # MJPG avant la résolution : DirectShow négocie le format à ce moment.
        # En non compressé, 1280x720 à 30 i/s sature l'USB 2.0 et peut couper
        # les autres périphériques du même contrôleur (casque, micro).
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.jpeg_quality = jpeg_quality

        self._jpeg = None
        self._seq = 0
        self._new_frame = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def resolution(self):
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fourcc(self):
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip() or "?"

    @property
    def running(self):
        return self._running

    def _loop(self):
        failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures >= 30:
                    logger.error("30 lectures consécutives échouées, arrêt de la capture.")
                    self._running = False
                    with self._new_frame:
                        self._new_frame.notify_all()
                continue

            failures = 0
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                continue
            with self._new_frame:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self._new_frame.notify_all()

    def wait_jpeg(self, last_seq, timeout=2.0):
        """(numéro, JPEG) de la première image plus récente que last_seq, ou (last_seq, None)."""
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._seq != last_seq or not self._running, timeout)
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._jpeg

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


class MJPEGHandler(BaseHTTPRequestHandler):
    """Sert /video (flux MJPEG), /snapshot (une image) et / (page de test)."""

    camera = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.client_address[0], fmt % args)

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
        _, jpeg = self.camera.wait_jpeg(last_seq=-1)
        if jpeg is None:
            self.send_error(503, "Aucune frame disponible")
            return
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
        seq = 0
        try:
            while self.camera.running:
                # Attend une image nouvelle : jamais deux fois la même.
                seq, jpeg = self.camera.wait_jpeg(seq)
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

    camera = CameraStream(args.index, args.width, args.height, args.fps, jpeg_quality=args.quality)
    width, height = camera.resolution
    logger.info("Webcam %d ouverte en %dx%d, format %s, %.0f i/s demandées par le pilote",
                args.index, width, height, camera.fourcc, camera.cap.get(cv2.CAP_PROP_FPS))

    MJPEGHandler.camera = camera
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
