"""
tests/test_camera_worker.py

Processus de caméras (core/camera_worker.py), sans GPU : le Worker tourne ici
dans le processus du test, avec un tracker factice et une séquence d'images
rejouée ; les résultats passent par un vrai Pipe multiprocessing, comme vers
l'application.
"""

import threading
from multiprocessing import Pipe
from unittest.mock import MagicMock

import pytest

from core import camera_worker
from core.camera_worker import Worker, WorkerCamera, decode_specs, encode_specs
from tests.test_video_source import N_FRAMES, make_sequence


def fake_tracker():
    tracker = MagicMock()
    tracker.update.return_value = []
    tracker.last_timings = {'detection': 0.002}
    tracker.state_sizes.return_value = {'vote_buffer': 0}
    return tracker


@pytest.fixture
def every_frame(monkeypatch):
    monkeypatch.setattr(camera_worker.config, "VIDEO_MODE", "every_frame")
    monkeypatch.setattr(camera_worker.config, "VIDEO_LOOP", False)
    monkeypatch.setattr(camera_worker.config, "MJPEG_ANNOTATE", False)


@pytest.fixture
def worker(tmp_path, every_frame):
    app_end, worker_end = Pipe(duplex=False)
    w = Worker([WorkerCamera("porte", make_sequence(tmp_path), True)], worker_end,
               fake_tracker, MagicMock())
    yield w, app_end
    w.stop()


def receive(conn, n):
    results = []
    for _ in range(n):
        assert conn.poll(5), "pas de résultat du worker"
        results.append(conn.recv())
    return results


def test_specs_survive_the_command_line():
    cameras = [WorkerCamera("hall", 0, False), WorkerCamera("porte", "/v.mp4", True, "t.csv")]

    assert decode_specs(encode_specs(cameras)) == cameras


def test_results_carry_measures_but_no_image(worker):
    w, conn = worker
    w.start()

    first = receive(conn, 1)[0]

    assert first.camera_id == "porte"
    assert first.frame is None and first.display_frame is None and first.jpeg is None
    assert first.frame_size == (64, 48)
    assert ('detection', 0.002) in first.stats['timings']
    assert first.stats['sizes'] == {'vote_buffer': 0, 'last_seen': 0}


def test_every_frame_of_the_sequence_comes_back(worker):
    w, conn = worker
    w.start()

    assert len(receive(conn, N_FRAMES)) == N_FRAMES


def test_a_jpeg_is_sent_only_while_the_stream_is_watched(worker):
    w, conn = worker
    w.handle(("stream", "porte", True))
    w.start()

    assert all(r.jpeg and r.jpeg[:2] == b'\xff\xd8' for r in receive(conn, 3))


def test_stop_command_stops_the_threads(worker):
    w, conn = worker
    threads = w.start()

    w.handle(("stop",))
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)


def test_the_worker_stops_when_the_app_stops_listening(tmp_path, every_frame):
    app_end, worker_end = Pipe(duplex=False)
    app_end.close()
    w = Worker([WorkerCamera("porte", make_sequence(tmp_path), True)], worker_end,
               fake_tracker, MagicMock())

    threads = w.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not w.running()


def test_unknown_commands_are_ignored(worker):
    w, _ = worker

    w.handle(("reload",))
    w.handle(("stream", "ailleurs", True))

    assert w.running()


def test_only_one_thread_sends_at_a_time(worker):
    w, conn = worker
    sent = []
    w._results = MagicMock(send=lambda r: sent.append(r))
    threads = [threading.Thread(target=w.send, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(sent) == list(range(20))
