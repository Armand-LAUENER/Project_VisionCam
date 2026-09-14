"""
tests/test_appearance_embedder.py

Couvre core/appearance_embedder.py sans GPU : le prétraitement groupé et
l'embedder PyTorch doivent reproduire la référence de deep_sort_realtime
(`MobileNetv2_Embedder`), dont ils reprennent le réseau et les poids. Le moteur
TensorRT, lui, ne se teste qu'avec un GPU (tools/bench_embedder.py).

Lancer : pytest tests/test_appearance_embedder.py -v
"""

import numpy as np
import pytest
import torch
from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

import config
from core import appearance_embedder
from core.appearance_embedder import EMBED_DIM, TorchEmbedder, build_embedder, preprocess

CPU = torch.device("cpu")


def random_crops(seed=0):
    """Crops BGR de tailles variées, dont un plus petit que l'entrée réseau."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=shape, dtype=np.uint8)
            for shape in ((180, 70, 3), (33, 15, 3), (400, 260, 3))]


class TestMatchesDeepSortRealtime:

    def test_preprocess_matches_the_reference(self):
        reference = object.__new__(MobileNetv2_Embedder)
        reference.bgr = True
        crops = random_crops()

        expected = torch.cat([reference.preprocess(crop) for crop in crops])

        torch.testing.assert_close(preprocess(crops, CPU), expected, rtol=0, atol=1e-6)

    def test_cpu_embeddings_match_the_reference(self):
        crops = random_crops(seed=1)
        reference = MobileNetv2_Embedder(half=False, gpu=False)

        expected = np.asarray(reference.predict(crops))
        actual = TorchEmbedder(gpu=False).predict(crops)

        assert actual.shape == (len(crops), EMBED_DIM)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_no_crop_gives_an_empty_batch(self):
        assert TorchEmbedder(gpu=False).predict([]).shape == (0, EMBED_DIM)


class TestBuildEmbedder:

    def test_pytorch_is_the_default(self, monkeypatch):
        monkeypatch.setattr(config, "DEEPSORT_EMBEDDER_ENGINE", "")
        monkeypatch.setattr(appearance_embedder, "TorchEmbedder", lambda gpu: ("torch", gpu))

        assert build_embedder() == ("torch", config.DEEPSORT_EMBEDDER_GPU)

    def test_missing_engine_explains_how_to_build_it(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DEEPSORT_EMBEDDER_ENGINE", str(tmp_path / "absent.engine"))

        with pytest.raises(FileNotFoundError, match="export_embedder_engine"):
            build_embedder()

    def test_existing_engine_selects_tensorrt(self, monkeypatch, tmp_path):
        engine = tmp_path / "embedder.engine"
        engine.write_bytes(b"")
        monkeypatch.setattr(config, "DEEPSORT_EMBEDDER_ENGINE", str(engine))
        monkeypatch.setattr(appearance_embedder, "TensorRTEmbedder", lambda path: ("trt", path))

        assert build_embedder() == ("trt", str(engine))
