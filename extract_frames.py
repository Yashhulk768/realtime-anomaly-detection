"""
extract_frames.py
-----------------
Step 1 of the pipeline: turn UCF-Crime videos into labelled image frames.

Key design choice: the train / val / test split is done BY VIDEO, not by frame.
Frames from the same video look almost identical, so if one video's frames
ended up in both train and test, the model could "recognise the video" instead
of the activity and the test accuracy would be misleadingly high (data leakage).

Usage (from the project folder, with the virtual environment active):
    python extract_frames.py --raw data/raw --out data/frames
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# The five classes used in the original IJIRCCE 2024 paper
CLASSES = ["Arson", "Burglary", "Explosion", "Fighting", "Normal"]
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def label_for(path: Path):
    """Work out a video's class from the name of the folder it sits in."""
    if "__MACOSX" in path.parts or path.name.startswith("._"):
        return None  # junk files some zips contain
    parent = path.parent.name
    if "normal" in parent.lower():
        return "Normal"
    return parent if parent in CLASSES else None


def find_videos(raw_dir: Path):
    """Return a list of (video_path, label) for every usable video."""
    videos = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            label = label_for(p)
            if label:
                videos.append((p, label))
    return videos


def split_by_video(videos, val_frac, test_frac, seed):
    """Assign each whole video to train, val or test, separately per class."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for path, label in videos:
        by_class[label].append(path)

    rows = []
    for label, paths in sorted(by_class.items()):
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for i, path in enumerate(paths):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            rows.append((path, label, split))
    return rows


def extract_one(video_path: Path, label, split, out_dir: Path, max_frames, size):
    """Save up to max_frames evenly spaced frames from one video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return str(video_path), 0

    k = min(max_frames, total)
    wanted = set(np.linspace(0, total - 1, k, dtype=int).tolist())
    last_wanted = max(wanted)

    dest = out_dir / split / label
    dest.mkdir(parents=True, exist_ok=True)

    saved, idx = 0, 0
    while idx <= last_wanted:
        if not cap.grab():          # grab() moves to the next frame cheaply
            break
        if idx in wanted:
            ok, frame = cap.retrieve()  # only fully decode frames we keep
            if ok:
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                name = f"{video_path.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(dest / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1
        idx += 1

    cap.release()
    return str(video_path), saved


def main():
    ap = argparse.ArgumentParser(description="Extract frames from UCF-Crime videos")
    ap.add_argument("--raw", default="data/raw", help="folder with the unzipped videos")
    ap.add_argument("--out", default="data/frames", help="where to write the frames")
    ap.add_argument("--max-frames", type=int, default=150, help="frames kept per video")
    ap.add_argument("--size", type=int, default=128, help="output image size (pixels)")
    ap.add_argument("--val", type=float, default=0.15, help="fraction of videos for validation")
    ap.add_argument("--test", type=float, default=0.15, help="fraction of videos for testing")
    ap.add_argument("--seed", type=int, default=42, help="random seed, for a repeatable split")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    videos = find_videos(raw_dir)
    if not videos:
        raise SystemExit(f"No videos found under {raw_dir}. Check the folder names.")

    print("Videos found per class:")
    for label, count in sorted(Counter(l for _, l in videos).items()):
        print(f"  {label:<10} {count}")
    missing = set(CLASSES) - {l for _, l in videos}
    if missing:
        print(f"  (no videos found for: {', '.join(sorted(missing))})")

    rows = split_by_video(videos, args.val, args.test, args.seed)

    # Save the split so it can be reported and reproduced
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "label", "split"])
        for path, label, split in rows:
            w.writerow([path.name, label, split])

    frame_counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, p, l, s, out_dir, args.max_frames, args.size): (l, s)
            for p, l, s in rows
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Videos"):
            label, split = futures[fut]
            _, saved = fut.result()
            frame_counts[(split, label)] += saved

    video_counts = Counter((s, l) for _, l, s in rows)
    print("\nSummary (videos / frames):")
    print(f"  {'class':<10}" + "".join(f"{s:>16}" for s in ("train", "val", "test")))
    for label in CLASSES:
        cells = "".join(
            f"{video_counts[(s, label)]:>6} / {frame_counts[(s, label)]:<7}"
            for s in ("train", "val", "test")
        )
        print(f"  {label:<10}{cells}")
    print(f"\nFrames saved to {out_dir}/  |  split list saved to {out_dir}/splits.csv")


if __name__ == "__main__":
    main()
