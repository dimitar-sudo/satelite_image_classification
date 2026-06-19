"""Exploratory data analysis.

Generates three figures into ``outputs/plots``:

* ``random_samples.png``             - a row of random labelled tiles,
* ``average_pixel_distribution.png`` - per-channel average-pixel histograms,
* ``average_brightness.png``         - per-class brightness box plots.

Run as a module from the repository root::

    python -m src.eda
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from .config import PLOTS_DIR, SEED, TRAIN_DIR
from .data import preprocess, split_data


def _load(row) -> Image.Image:
    return Image.open(os.path.join(row.folder, row.file_name)).convert("RGB")


def show_samples(df: pd.DataFrame, idx_to_class: dict, num_samples: int = 5) -> None:
    """Display ``num_samples`` random images in a row."""
    sample = df.sample(num_samples, random_state=SEED)
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    if num_samples == 1:
        axes = [axes]
    for ax, row in zip(axes, sample.itertuples(index=False)):
        ax.imshow(_load(row))
        ax.set_title(f"Label: {idx_to_class[row.label]}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "random_samples.png"),
                dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def average_pixel_plot(df: pd.DataFrame) -> None:
    """Histogram of per-image average pixel value for each RGB channel."""
    means = np.array([
        np.asarray(_load(row)).reshape(-1, 3).mean(axis=0)
        for row in df.itertuples(index=False)
    ])
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, color in enumerate(["red", "green", "blue"]):
        ax.hist(means[:, c], bins=40, color=color, alpha=0.5,
                label=f"{color.capitalize()} channel")
    ax.set_title("Distribution of Average Pixel Values")
    ax.set_xlabel("Average Pixel Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "average_pixel_distribution.png"),
                dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def average_brightness_per_class(df: pd.DataFrame, idx_to_class: dict) -> None:
    """Box plot of average brightness (mean over RGB) per class."""
    brightness = np.array([
        np.asarray(_load(row)).mean() for row in df.itertuples(index=False)
    ])
    labels_sorted = sorted(df["label"].unique())
    classes = [idx_to_class[i] for i in labels_sorted]
    data = [brightness[df["label"].values == i] for i in labels_sorted]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, labels=classes, patch_artist=True,
                    medianprops=dict(color="red"))
    for box in bp["boxes"]:
        box.set_facecolor("lightblue")
    ax.set_title("Average Brightness Distribution per Class")
    ax.set_ylabel("Average Brightness")
    ax.set_xticklabels(classes, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "average_brightness.png"),
                dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    df, label_dict = preprocess(TRAIN_DIR)
    idx_to_class = {v: k for k, v in label_dict.items()}
    train_df, _ = split_data(df)

    show_samples(df, idx_to_class, 5)
    average_pixel_plot(train_df)
    average_brightness_per_class(train_df, idx_to_class)
    print("EDA plots written to", PLOTS_DIR)


if __name__ == "__main__":
    main()
