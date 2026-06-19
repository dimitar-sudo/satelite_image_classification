"""Satellite terrain classification — source package.

Modules
-------
config    : typed access to ``configs/default.yaml``.
data      : dataset scanning, train/val split and the PyTorch ``Dataset``.
model     : the ``SatelliteCNN`` architecture.
eda       : exploratory data-analysis plots.
train     : training loop with checkpointing and early stopping.
evaluate  : validation metrics, plots and test-set submission.
"""

__all__ = ["config", "data", "model", "eda", "train", "evaluate"]
