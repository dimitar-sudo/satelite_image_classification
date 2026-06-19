"""Training entry point.

Trains :class:`~src.model.SatelliteCNN` on the EuroSAT tiles, checkpoints the
best model by validation accuracy, and writes the training history.

Run from the repository root::

    python -m src.train
"""
from __future__ import annotations

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    CONFIG, SEED, NUM_CLASSES, TRAIN_DIR, WEIGHTS_PATH, HISTORY_PATH,
)
from .data import preprocess, split_data, EuroSATDataset, train_transform, eval_transform
from .model import SatelliteCNN


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(batch_size: int, num_workers: int):
    df, label_dict = preprocess(TRAIN_DIR)
    train_df, val_df = split_data(df)
    train_ds = EuroSATDataset(train_df, train_transform)
    val_ds = EuroSATDataset(val_df, eval_transform)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader, label_dict


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (average loss, accuracy) of ``model`` over ``loader``."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train() -> tuple[dict, float]:
    cfg = CONFIG["train"]
    epochs = cfg["epochs"]
    patience = cfg["patience"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = make_loaders(cfg["batch_size"], cfg["num_workers"])
    model = SatelliteCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, epochs_no_improve = 0.0, 0
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()

        tr_loss, tr_acc = loss_sum / total, correct / total
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        improved = va_acc > best_acc
        if improved:
            best_acc = va_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{'*' if improved else ' '} best {best_acc:.4f} | "
              f"{time.time() - t0:.1f}s", flush=True)

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(history, fh)
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Weights saved to {WEIGHTS_PATH}")
    return history, best_acc


if __name__ == "__main__":
    train()
