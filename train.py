"""
train.py
--------
Train and evaluate a frame classifier on the extracted UCF-Crime frames.

Two models are available:
    --model baseline   small CNN in the spirit of the IJIRCCE 2024 paper
                       (Conv2D -> MaxPool2D -> Dense -> softmax), trained from scratch
    --model mobilenet  MobileNetV2 pretrained on ImageNet, fine-tuned on our frames
                       (transfer learning; also a typical model for Arm/edge devices)

The best epoch (by validation macro-F1) is kept and then tested ONCE on the
held-out test videos, which the model has never seen.

Outputs (NAME is cnn_baseline or mobilenet_v2):
    models/NAME.pt
    results/NAME/classification_report.txt
    results/NAME/confusion_matrix.png
    results/NAME/training_curves.png
    results/NAME/metrics.json

Usage (from the project folder, with the virtual environment active):
    python train.py --model baseline
    python train.py --model mobilenet
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Four Conv2D + MaxPool2D blocks, then Dense layers and a softmax output."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # halves the image size
            )

        self.features = nn.Sequential(
            block(3, 32),     # 128 -> 64
            block(32, 64),    # 64  -> 32
            block(64, 128),   # 32  -> 16
            block(128, 256),  # 16  -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 256 x 8 x 8 -> 256 x 1 x 1
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),      # Dense layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # softmax is applied inside the loss
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "baseline":
        return BaselineCNN(num_classes)
    # MobileNetV2 already knows general visual features (edges, objects, people)
    # from ImageNet; we only replace its final layer and fine-tune.
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(net.last_channel, num_classes),
    )
    return net


# Per-model defaults: name, epochs, learning rate, normalisation
CONFIG = {
    "baseline": dict(name="cnn_baseline", epochs=15, lr=1e-3,
                     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    "mobilenet": dict(name="mobilenet_v2", epochs=8, lr=3e-4,
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, acc, macro_f1, all_labels, all_preds


def main():
    ap = argparse.ArgumentParser(description="Train a frame classifier")
    ap.add_argument("--model", choices=["baseline", "mobilenet"], default="baseline")
    ap.add_argument("--data", default="data/frames")
    ap.add_argument("--epochs", type=int, default=None, help="default depends on model")
    ap.add_argument("--lr", type=float, default=None, help="default depends on model")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = CONFIG[args.model]
    epochs = args.epochs or cfg["epochs"]
    lr = args.lr or cfg["lr"]
    name, mean, std = cfg["name"], cfg["mean"], cfg["std"]
    out_dir = Path("results") / name

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {name} | device: {device} | epochs: {epochs} | lr: {lr}")

    # --- Data -------------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.7, 1.0)),  # random zoom/crop
        transforms.RandomHorizontalFlip(),                          # mirror image
        transforms.ColorJitter(brightness=0.3, contrast=0.3),       # lighting changes
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data = Path(args.data)
    train_ds = datasets.ImageFolder(data / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data / "test", transform=eval_tf)
    classes = train_ds.classes
    print(f"Classes: {classes}")
    print(f"Frames - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_dl = DataLoader(test_ds, shuffle=False, **loader_kw)

    # Burglary has twice as many videos as the other classes. Class weights
    # make mistakes on the smaller classes count more.
    counts = np.bincount(train_ds.targets, minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32)

    # --- Model, loss, optimiser ------------------------------------------
    model = build_model(args.model, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # label_smoothing stops the model becoming over-confident on noisy labels
    # (many frames in an anomaly video actually look normal)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training loop ----------------------------------------------------
    Path("models").mkdir(exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("models") / f"{name}.pt"
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "val_f1")}
    best_f1, best_epoch = -1.0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc, va_f1, _, _ = run_epoch(model, val_dl, criterion, device)
        scheduler.step()

        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc, va_f1)):
            history[k].append(v)

        marker = ""
        if va_f1 > best_f1:  # keep the model that does best on validation
            best_f1, best_epoch = va_f1, epoch
            torch.save({"arch": args.model, "model_state": model.state_dict(),
                        "classes": classes, "size": args.size, "mean": mean, "std": std},
                       ckpt_path)
            marker = "  <- best so far, saved"
        print(f"Epoch {epoch:2d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f} macro-F1 {va_f1:.3f}{marker}")

    minutes = (time.time() - start) / 60
    print(f"\nTraining finished in {minutes:.1f} min. Best validation macro-F1 "
          f"{best_f1:.3f} at epoch {best_epoch}.")

    # --- Final test on videos the model has never seen ---------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, te_acc, te_f1, y_true, y_pred = run_epoch(model, test_dl, criterion, device)

    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    print("\nTEST SET RESULTS (unseen videos)")
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - test set (accuracy {te_acc:.1%})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    ep = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, history["train_acc"], label="train acc")
    axes[1].plot(ep, history["val_acc"], label="validation acc")
    axes[1].plot(ep, history["val_f1"], label="validation macro-F1")
    axes[1].set_title("Accuracy / F1"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    metrics = {"model": name, "test_accuracy": te_acc, "test_macro_f1": te_f1,
               "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
               "epochs": epochs, "parameters": n_params,
               "train_minutes": round(minutes, 1)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test accuracy {te_acc:.1%} | test macro-F1 {te_f1:.3f}")
    print(f"Saved: {ckpt_path} and {out_dir}/")


if __name__ == "__main__":
    main()
