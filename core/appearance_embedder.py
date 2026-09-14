"""
appearance_embedder.py — Embeddings d'apparence (MobileNetV2) pour DeepSORT.

Même réseau et mêmes poids que l'embedder de deep_sort_realtime
(`MobileNetv2_Embedder`), mais calculé autrement :

- prétraitement groupé : chaque crop est redimensionné en 224x224 sur le CPU,
  puis le lot entier part en un seul transfert et est normalisé sur le GPU.
  La référence normalisait crop par crop sur le CPU (~3 ms à 8 personnes) ;
- `TorchEmbedder` (défaut) : PyTorch FP16 en channels_last ;
- `TensorRTEmbedder` : moteur TensorRT FP16, activé par
  `config.DEEPSORT_EMBEDDER_ENGINE`.

Mesures (tools/bench_embedder.py, tools/eval_mot.py) : voir le README,
section Accélération TensorRT.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import torch

import config

logger = logging.getLogger(__name__)

INPUT_SIZE = 224
EMBED_DIM = 1280
# Lot maximal du profil d'optimisation du moteur TensorRT : les lots plus
# grands sont découpés.
ENGINE_MAX_BATCH = 32
ENGINE_EXPORT_COMMAND = "uv run -m tools.export_embedder_engine"

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def preprocess(crops: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Crops BGR → tenseur (N, 3, 224, 224) float32 normalisé ImageNet.

    Redimensionner puis inverser les canaux donne le même résultat que la
    référence, qui inverse d'abord : le resize traite chaque canal à part.
    """
    batch = np.stack([cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE)) for crop in crops])
    rgb = np.ascontiguousarray(batch[..., ::-1])
    tensor = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def load_mobilenetv2() -> torch.nn.Module:
    """MobileNetV2-bottleneck avec les poids fournis par deep_sort_realtime."""
    from deep_sort_realtime.embedder.embedder_pytorch import MOBILENETV2_BOTTLENECK_WTS
    from deep_sort_realtime.embedder.mobilenetv2_bottle import MobileNetV2_bottle

    model = MobileNetV2_bottle(input_size=INPUT_SIZE, width_mult=1.0)
    model.load_state_dict(torch.load(MOBILENETV2_BOTTLENECK_WTS, weights_only=True))
    return model.eval()


class TorchEmbedder:
    """MobileNetV2 en PyTorch : FP16 channels_last sur GPU, FP32 sur CPU."""

    def __init__(self, gpu: bool = True) -> None:
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.model = load_mobilenetv2().to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half().to(memory_format=torch.channels_last)
        logger.info("Embedder d'apparence : MobileNetV2 PyTorch (%s)", self.device)
        self.predict([np.zeros((100, 100, 3), dtype=np.uint8)])  # warm-up

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            batch = preprocess(crops, self.device)
            if self.device.type == "cuda":
                batch = batch.half().contiguous(memory_format=torch.channels_last)
            return self.model(batch).cpu().numpy()


class TensorRTEmbedder:
    """MobileNetV2 compilé en moteur TensorRT FP16 (lot dynamique)."""

    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(
                f"Moteur TensorRT illisible : {engine_path}. Il a peut-être été construit "
                f"avec une autre version de TensorRT ou un autre GPU : {ENGINE_EXPORT_COMMAND}"
            )
        self._context = self._engine.create_execution_context()
        # Flux CUDA dédié : sur le flux par défaut, TensorRT ajoute des
        # synchronisations à chaque exécution.
        self._stream = torch.cuda.Stream()
        self.device = torch.device("cuda")
        logger.info("Embedder d'apparence : MobileNetV2 TensorRT (%s)", engine_path)
        self.predict([np.zeros((100, 100, 3), dtype=np.uint8)])  # warm-up

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        outputs = []
        with torch.cuda.stream(self._stream):
            batch = preprocess(crops, self.device)
            for start in range(0, len(crops), ENGINE_MAX_BATCH):
                chunk = batch[start:start + ENGINE_MAX_BATCH].contiguous()
                out = torch.empty((chunk.shape[0], EMBED_DIM), device=self.device,
                                  dtype=torch.float32)
                self._context.set_input_shape("images", tuple(chunk.shape))
                self._context.set_tensor_address("images", chunk.data_ptr())
                self._context.set_tensor_address("features", out.data_ptr())
                self._context.execute_async_v3(self._stream.cuda_stream)
                outputs.append(out)
            # FP16 comme TorchEmbedder sur GPU : deep_sort_realtime stocke les
            # vecteurs tels quels, les deux chemins restent interchangeables.
            result = torch.cat(outputs).half().cpu().numpy()
        return result


def build_embedder() -> TorchEmbedder | TensorRTEmbedder:
    """Instancie l'embedder désigné par la configuration."""
    engine = config.DEEPSORT_EMBEDDER_ENGINE
    if not engine:
        return TorchEmbedder(gpu=config.DEEPSORT_EMBEDDER_GPU)
    if not os.path.exists(engine):
        raise FileNotFoundError(
            f"Moteur TensorRT de l'embedder introuvable : {engine}. Il se construit sur "
            f"la machine qui l'utilise : {ENGINE_EXPORT_COMMAND}"
        )
    return TensorRTEmbedder(engine)
