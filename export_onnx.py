"""
export_onnx.py
--------------
Step 3 of the pipeline: prepare the trained model for edge / Arm-style deployment.

  1. Export the PyTorch model to ONNX (a portable format many runtimes can run,
     including ONNX Runtime on Arm CPUs).
  2. Quantise it to 8-bit integers (INT8) with static quantisation, calibrated
     on a sample of validation frames. INT8 models are ~4x smaller and usually
     faster on CPUs, which is why it's standard practice on low-power devices.
  3. Benchmark FP32 vs INT8 on the CPU: file size, latency per frame, and
     accuracy / macro-F1 on the test videos, so the trade-off is measured,
     not assumed.

Outputs:
    models/NAME_fp32.onnx
    models/NAME_int8.onnx
    results/NAME/edge_benchmark.json

Usage (from the project folder, with the virtual environment active):
    python export_onnx.py --checkpoint models/mobilenet_v2.pt
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from sklearn.metrics import f1_score
from torchvision import datasets, transforms

from train import build_model  # reuse the exact same model definitions


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def eval_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class FrameCalibrationReader(CalibrationDataReader):
    """Feeds a sample of validation frames to the quantiser so it can measure
    the typical range of values flowing through each layer."""

    def __init__(self, dataset, input_name, n_samples=300, seed=0):
        rng = random.Random(seed)
        idx = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
        self.samples = [dataset[i][0].unsqueeze(0).numpy() for i in idx]
        self.input_name = input_name
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.samples):
            return None
        item = {self.input_name: self.samples[self.pos]}
        self.pos += 1
        return item


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_to_onnx(model, size, path):
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
    try:
        # classic exporter (works on most PyTorch versions)
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        # older PyTorch without the 'dynamo' argument
        torch.onnx.export(model, dummy, str(path), **kwargs)
    except Exception as e:
        print(f"Classic exporter failed ({e.__class__.__name__}); trying the new exporter...")
        torch.onnx.export(model, (dummy,), str(path), dynamo=True,
                          input_names=["input"], output_names=["logits"])


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def make_session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def latency_ms(session, size, runs=200, warmup=20):
    """Average time to classify ONE frame (batch size 1), as in a live camera feed."""
    name = session.get_inputs()[0].name
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {name: x})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {name: x})
    return (time.perf_counter() - start) / runs * 1000


def evaluate(session, dataset, batch_size=64):
    name = session.get_inputs()[0].name
    preds, labels = [], []
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        x = np.stack([b[0].numpy() for b in batch]).astype(np.float32)
        out = session.run(None, {name: x})[0]
        preds.extend(out.argmax(1).tolist())
        labels.extend(b[1] for b in batch)
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    return acc, float(f1_score(labels, preds, average="macro"))


def main():
    ap = argparse.ArgumentParser(description="Export, quantise and benchmark a model")
    ap.add_argument("--checkpoint", default="models/mobilenet_v2.pt")
    ap.add_argument("--data", default="data/frames")
    ap.add_argument("--calib-samples", type=int, default=300)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    name = ckpt_path.stem
    size, mean, std, classes = ckpt["size"], ckpt["mean"], ckpt["std"], ckpt["classes"]

    model = build_model(ckpt["arch"], len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    fp32_path = Path("models") / f"{name}_fp32.onnx"
    int8_path = Path("models") / f"{name}_int8.onnx"
    out_dir = Path("results") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export ---------------------------------------------------------------
    print("1/4 Exporting to ONNX...")
    export_to_onnx(model, size, fp32_path)

    # sanity check: ONNX must give the same answers as PyTorch
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        torch_out = model(x).numpy()
    onnx_out = make_session(fp32_path, 1).run(None, {"input": x.numpy()})[0]
    print(f"    max difference PyTorch vs ONNX: {np.abs(torch_out - onnx_out).max():.2e} "
          "(should be tiny, e.g. < 1e-3)")

    # 2. Quantise -------------------------------------------------------------
    print("2/4 Quantising to INT8 (calibrating on validation frames)...")
    tf = eval_transform(size, mean, std)
    val_ds = datasets.ImageFolder(Path(args.data) / "val", transform=tf)
    test_ds = datasets.ImageFolder(Path(args.data) / "test", transform=tf)

    prepped = fp32_path
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        prepped = Path("models") / f"{name}_prep.onnx"
        quant_pre_process(str(fp32_path), str(prepped))
    except Exception as e:
        print(f"    (pre-processing skipped: {e.__class__.__name__}; continuing)")
        prepped = fp32_path

    quantize_static(
        str(prepped), str(int8_path),
        FrameCalibrationReader(val_ds, "input", args.calib_samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    if prepped != fp32_path and prepped.exists():
        prepped.unlink()

    # 3. Speed ----------------------------------------------------------------
    print("3/4 Measuring CPU latency (batch size 1)...")
    all_threads = os.cpu_count() or 1
    rows = {}
    for label, path in (("FP32", fp32_path), ("INT8", int8_path)):
        rows[label] = {
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "latency_ms_1_thread": round(latency_ms(make_session(path, 1), size), 2),
            f"latency_ms_{all_threads}_threads": round(latency_ms(make_session(path, all_threads), size), 2),
        }

    # 4. Accuracy -------------------------------------------------------------
    print("4/4 Checking accuracy on the test videos (CPU, may take a minute)...")
    for label, path in (("FP32", fp32_path), ("INT8", int8_path)):
        acc, f1 = evaluate(make_session(path, all_threads), test_ds)
        rows[label]["test_accuracy"] = round(acc, 4)
        rows[label]["test_macro_f1"] = round(f1, 4)

    (out_dir / "edge_benchmark.json").write_text(json.dumps(rows, indent=2))

    f, i = rows["FP32"], rows["INT8"]
    k = f"latency_ms_{all_threads}_threads"
    print(f"\nEDGE BENCHMARK - {name} (CPU, ONNX Runtime)")
    print(f"| Model | Size (MB) | Latency 1 thread (ms) | Latency {all_threads} threads (ms) | Test acc | Macro-F1 |")
    print("|---|---|---|---|---|---|")
    for label, r in rows.items():
        print(f"| {label} | {r['size_mb']} | {r['latency_ms_1_thread']} | {r[k]} | "
              f"{r['test_accuracy']:.1%} | {r['test_macro_f1']:.3f} |")
    print(f"\nINT8 is {f['size_mb'] / i['size_mb']:.1f}x smaller and "
          f"{f['latency_ms_1_thread'] / i['latency_ms_1_thread']:.1f}x faster on 1 thread, "
          f"accuracy change {100 * (i['test_accuracy'] - f['test_accuracy']):+.1f} points.")
    print(f"Saved: {fp32_path}, {int8_path}, {out_dir / 'edge_benchmark.json'}")


if __name__ == "__main__":
    main()
