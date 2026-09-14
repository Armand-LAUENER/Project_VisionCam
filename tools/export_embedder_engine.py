"""
export_embedder_engine.py — Construit le moteur TensorRT FP16 de l'embedder.

Exporte MobileNetV2 (réseau et poids de deep_sort_realtime) en ONNX, puis le
compile en TensorRT FP16 avec un lot dynamique de 1 à ENGINE_MAX_BATCH crops.
Le moteur ne vaut que pour le GPU et la version de TensorRT qui l'ont
construit : à relancer sur chaque machine, et après une mise à jour de
TensorRT ou du pilote.

Depuis la racine du projet (~1 min sur une RTX 4060) :

    uv run -m tools.export_embedder_engine
    # puis DEEPSORT_EMBEDDER_ENGINE=mobilenetv2-embedder.engine dans .env
"""

import argparse
import os
import tempfile
import time

import tensorrt as trt
import torch

from core.appearance_embedder import ENGINE_MAX_BATCH, INPUT_SIZE, load_mobilenetv2

# Lot pour lequel TensorRT choisit ses noyaux : l'ordre de grandeur du nombre
# de personnes par image dans les séquences mesurées.
OPTIMAL_BATCH = 8


def export_onnx(path):
    model = load_mobilenetv2()
    torch.onnx.export(
        model, torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE), path,
        input_names=["images"], output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )


def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX illisible par TensorRT :\n{errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    shape = (3, INPUT_SIZE, INPUT_SIZE)
    profile.set_shape("images", (1, *shape), (OPTIMAL_BATCH, *shape), (ENGINE_MAX_BATCH, *shape))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT n'a pas pu construire le moteur (voir les messages ci-dessus).")
    with open(engine_path, "wb") as f:
        f.write(serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="mobilenetv2-embedder.engine")
    args = parser.parse_args()

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "mobilenetv2.onnx")
        export_onnx(onnx_path)
        build_engine(onnx_path, args.output)
    print(f"Moteur écrit : {args.output} ({time.time() - start:.0f} s, "
          f"TensorRT {trt.__version__}, {torch.cuda.get_device_name(0)})")


if __name__ == "__main__":
    main()
