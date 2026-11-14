"""
app.py
------
Browser demo for the anomaly detector.

Upload a surveillance clip and the app:
  * runs the INT8 ONNX model on the CPU, as it would on an edge device
  * smooths predictions over several frames so one odd frame doesn't raise an alarm
  * returns the video with a live overlay (class, confidence, red ALERT banner)
  * plots each class's confidence over time
  * lists alerts with timestamps, like a security system log

Usage (from the project folder, with the virtual environment active):
    pip install gradio
    python app.py
Then open http://localhost:7860 in your browser.
"""

import argparse
import csv
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

MIN_CONSECUTIVE = 2   # abnormal predictions in a row needed before raising an alert
MAX_OUTPUT_WIDTH = 640

# Filled in at startup
SESSION = None
INPUT_NAME = None
CLASSES, SIZE, MEAN, STD = None, None, None, None


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def load_model(onnx_path: Path, ckpt_path: Path):
    """Load the ONNX model, plus class names and preprocessing settings
    saved in the PyTorch checkpoint during training."""
    global SESSION, INPUT_NAME, CLASSES, SIZE, MEAN, STD
    ckpt = torch.load(ckpt_path, map_location="cpu")
    CLASSES = ckpt["classes"]
    SIZE = ckpt["size"]
    MEAN = np.array(ckpt["mean"], dtype=np.float32)
    STD = np.array(ckpt["std"], dtype=np.float32)
    SESSION = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess(frame_bgr):
    """Same steps as training: RGB, resize, scale to 0-1, normalise, CHW."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # shape (1, 3, SIZE, SIZE)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_overlay(frame, label, conf, alert, ms):
    h, w = frame.shape[:2]
    band = frame.copy()
    cv2.rectangle(band, (0, 0), (w, 42), (0, 0, 0), -1)
    frame = cv2.addWeighted(band, 0.55, frame, 0.45, 0)

    if alert:
        text, colour = f"ALERT: {label} {conf:.0%}", (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, 6)
    else:
        text, colour = f"{label} {conf:.0%}", (80, 200, 80)
    cv2.putText(frame, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    if w >= 500:
        cv2.putText(frame, f"INT8 ONNX | {ms:.1f} ms/frame", (w - 235, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def timeline_plot(times, history, threshold):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    if history:
        H = np.array(history)
        for i, c in enumerate(CLASSES):
            style = "--" if c == "Normal" else "-"
            ax.plot(times, H[:, i], style, linewidth=2, label=c)
    ax.axhline(threshold, color="red", linestyle=":", linewidth=1, label="alert threshold")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time in video (s)")
    ax.set_ylabel("smoothed probability")
    ax.set_title("Model confidence over time")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def to_browser_mp4(src: Path, dst: Path) -> Path:
    """Browsers need H.264 video; OpenCV writes a format they can't play."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(cmd, check=True)
        return dst
    except (subprocess.CalledProcessError, FileNotFoundError):
        return src  # fall back to the raw file


def fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyse(video_path, analysis_fps, threshold, window, max_seconds, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video (or pick an example) first.")

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    max_frames = min(total, int(max_seconds * src_fps))
    step = max(1, round(src_fps / analysis_fps))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_OUTPUT_WIDTH / max(w, 1))
    out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2  # H.264 needs even sizes

    tmp = Path(tempfile.mkdtemp())
    raw_path, out_path = tmp / "raw.mp4", tmp / "annotated.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (out_w, out_h))

    buffer = deque(maxlen=int(window))
    times, history, infer_ms, alerts = [], [], [], []
    label, conf, ms = "...", 0.0, 0.0
    consecutive, alert_active, alert_cls = 0, False, None
    normal_idx = CLASSES.index("Normal") if "Normal" in CLASSES else -1

    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            x = preprocess(frame)
            t0 = time.perf_counter()
            logits = SESSION.run(None, {INPUT_NAME: x})[0][0]
            ms = (time.perf_counter() - t0) * 1000
            infer_ms.append(ms)

            buffer.append(softmax(logits))
            smooth = np.mean(buffer, axis=0)  # moving average over recent frames
            top = int(np.argmax(smooth))
            label, conf = CLASSES[top], float(smooth[top])
            t = idx / src_fps
            times.append(t)
            history.append(smooth)

            abnormal = top != normal_idx and conf >= threshold
            consecutive = consecutive + 1 if abnormal else 0
            if consecutive >= MIN_CONSECUTIVE:
                if not alert_active or top != alert_cls:
                    alerts.append({"Time": fmt_time(t), "Event": label,
                                   "Confidence": f"{conf:.0%}"})
                alert_active, alert_cls = True, top
            else:
                alert_active, alert_cls = False, None

            if len(times) % 10 == 0:
                progress(idx / max_frames, desc="Analysing video")

        small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(draw_overlay(small, label, conf, alert_active, ms))
        idx += 1

    cap.release()
    writer.release()

    if not history:
        raise gr.Error("Couldn't read any frames from this video.")

    progress(1.0, desc="Encoding video for the browser")
    video_out = to_browser_mp4(raw_path, out_path)

    alert_df = pd.DataFrame(alerts) if alerts else pd.DataFrame(
        [{"Time": "-", "Event": "No alerts raised", "Confidence": "-"}])

    mean_probs = np.mean(history, axis=0)
    ranked = sorted(zip(CLASSES, mean_probs), key=lambda p: -p[1])
    summary = (
        f"### Summary\n"
        f"- **Most likely activity across the clip:** {ranked[0][0]} "
        f"({ranked[0][1]:.0%} average confidence), then {ranked[1][0]} ({ranked[1][1]:.0%})\n"
        f"- **Alerts raised:** {len(alerts)}\n"
        f"- **Analysed:** {len(history)} frames from {idx / src_fps:.1f} s of video "
        f"({analysis_fps} frames per second)\n"
        f"- **Average inference time:** {np.mean(infer_ms):.1f} ms per frame on CPU (INT8 ONNX)\n\n"
        f"*This is a frame-level model with ~55% accuracy on unseen test videos, "
        f"so treat predictions as indicative, not definitive.*"
    )
    return str(video_out), timeline_plot(times, history, threshold), alert_df, summary


# ---------------------------------------------------------------------------
# Example videos: one TEST video per class (never seen during training)
# ---------------------------------------------------------------------------
def find_examples(splits_csv=Path("data/frames/splits.csv"), raw_dir=Path("data/raw")):
    if not splits_csv.exists() or not raw_dir.exists():
        return []
    paths = {p.name: p for p in raw_dir.rglob("*") if p.suffix.lower() in {".mp4", ".avi"}}
    chosen = {}
    with open(splits_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row["label"] not in chosen and row["video"] in paths:
                chosen[row["label"]] = str(paths[row["video"]])
    return [[chosen[c]] for c in sorted(chosen)]


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Surveillance Anomaly Detector") as demo:
        gr.Markdown(
            "# Real-Time Anomaly Detection in Surveillance Video\n"
            "Upload a CCTV clip. A MobileNetV2 model, quantised to **INT8** and running on the "
            "**CPU** with ONNX Runtime, classifies frames as Arson, Burglary, Explosion, "
            "Fighting or Normal, and raises an alert when abnormal activity persists."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Input video", sources=["upload"])
                analysis_fps = gr.Slider(1, 15, value=5, step=1,
                                         label="Frames analysed per second")
                threshold = gr.Slider(0.3, 0.9, value=0.5, step=0.05,
                                      label="Alert threshold (confidence)")
                window = gr.Slider(1, 15, value=5, step=1,
                                   label="Smoothing window (analysed frames)")
                max_seconds = gr.Slider(10, 300, value=90, step=10,
                                        label="Maximum seconds to analyse")
                run_btn = gr.Button("Analyse video", variant="primary")
            with gr.Column(scale=2):
                video_out = gr.Video(label="Annotated output")
                summary = gr.Markdown()
        plot = gr.Plot(label="Confidence timeline")
        alert_table = gr.Dataframe(label="Alert log", interactive=False)

        examples = find_examples()
        if examples:
            gr.Examples(examples=examples, inputs=[video_in],
                        label="Test-set videos (never seen during training), one per class")

        run_btn.click(analyse,
                      inputs=[video_in, analysis_fps, threshold, window, max_seconds],
                      outputs=[video_out, plot, alert_table, summary])
    return demo


def main():
    ap = argparse.ArgumentParser(description="Browser demo for the anomaly detector")
    ap.add_argument("--model", default="models/mobilenet_v2_int8.onnx")
    ap.add_argument("--checkpoint", default="models/mobilenet_v2.pt")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    load_model(Path(args.model), Path(args.checkpoint))
    print(f"Loaded {args.model} | classes: {CLASSES}")
    build_ui().queue().launch(server_port=args.port)


if __name__ == "__main__":
    main()
