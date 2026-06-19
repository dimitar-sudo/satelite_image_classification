"""Evaluation and inference.

Loads the trained checkpoint and, on the held-out validation split, reports
overall accuracy, macro F1 and Cohen's kappa together with a full per-class
precision/recall/F1 table. It also writes:

* ``outputs/plots/training_curves.png``
* ``outputs/plots/confusion_matrix.png``
* ``outputs/plots/misclassified_samples.png``
* ``outputs/metrics.json``       - machine-readable metrics
* ``outputs/submission.csv``     - predictions on the unlabelled test set

Run from the repository root::

    python -m src.evaluate
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, f1_score, cohen_kappa_score,
)

from .config import (
    NUM_CLASSES, TRAIN_DIR, TEST_DIR, SEED,
    WEIGHTS_PATH, HISTORY_PATH, METRICS_PATH, PLOTS_DIR, SUBMISSION_PATH,
)
from .data import (
    preprocess, preprocess_test, split_data, EuroSATDataset, eval_transform,
)
from .model import SatelliteCNN
from .train import get_device


@torch.no_grad()
def predict_loader(model, loader, device, with_labels: bool = True):
    """Return model predictions (and labels, if available) over a loader."""
    model.eval()
    preds, labels = [], []
    for batch in loader:
        x = batch[0] if with_labels else batch
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        if with_labels:
            labels.append(batch[1].numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels) if with_labels else None
    return preds, labels


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_training_curves(history) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss"); ax1.legend()
    ax2.plot(epochs, history["val_acc"], color="green", label="Validation accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy over Epochs"); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "training_curves.png"),
                dpi=120, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, idx_to_class) -> None:
    cm = confusion_matrix(y_true, y_pred)
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix (Validation Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"),
                dpi=120, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_misclassified(val_df, y_true, y_pred, idx_to_class, n: int = 5) -> None:
    mis = np.where(y_true != y_pred)[0]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(mis, size=min(n, len(mis)), replace=False)
    fig, axes = plt.subplots(1, len(pick), figsize=(3 * len(pick), 3.2))
    if len(pick) == 1:
        axes = [axes]
    for ax, i in zip(axes, pick):
        row = val_df.iloc[i]
        img = Image.open(os.path.join(row.folder, row.file_name)).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"True: {idx_to_class[y_true[i]]}\n"
                     f"Pred: {idx_to_class[y_pred[i]]}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "misclassified_samples.png"),
                dpi=120, bbox_inches="tight")
    plt.show(); plt.close(fig)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, names) -> dict:
    """Build a machine-readable metrics dict and print a readable report."""
    report = classification_report(y_true, y_pred, target_names=names,
                                   digits=4, output_dict=True)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            n: {
                "precision": report[n]["precision"],
                "recall": report[n]["recall"],
                "f1": report[n]["f1-score"],
                "support": report[n]["support"],
            } for n in names
        },
    }
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    print(f"Accuracy {metrics['accuracy']:.4f} | "
          f"Macro-F1 {metrics['macro_f1']:.4f} | "
          f"Cohen's kappa {metrics['cohen_kappa']:.4f}")
    return metrics


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = get_device()
    df, label_dict = preprocess(TRAIN_DIR)
    idx_to_class = {v: k for k, v in label_dict.items()}
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    _, val_df = split_data(df)

    model = SatelliteCNN(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # Training curves (if a history file is available).
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as fh:
            plot_training_curves(json.load(fh))

    # Validation metrics + plots.
    val_ds = EuroSATDataset(val_df, eval_transform)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    val_pred, val_true = predict_loader(model, val_loader, device, with_labels=True)

    metrics = compute_metrics(val_true, val_pred, names)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    plot_confusion_matrix(val_true, val_pred, idx_to_class)
    plot_misclassified(val_df, val_true, val_pred, idx_to_class, 5)

    # Test-set inference + submission file.
    test_df = preprocess_test(TEST_DIR)
    test_ds = EuroSATDataset(test_df, eval_transform, has_labels=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    test_pred, _ = predict_loader(model, test_loader, device, with_labels=False)
    submission = pd.DataFrame({
        "file_name": test_df["file_name"],
        "label": [idx_to_class[p] for p in test_pred],
    })
    submission.to_csv(SUBMISSION_PATH, index=False, header=False)
    print(f"Wrote {SUBMISSION_PATH} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
