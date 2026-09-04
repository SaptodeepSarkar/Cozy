"""Memory-mapped, class-balanced feature batches for wakeword training."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


def _batches(data_files: dict[str, str | Path], per_class: dict[str, int], labels: dict[str, Callable[[np.ndarray], int]]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    for name, raw_path in data_files.items():
        path = Path(raw_path)
        if not path.exists():
            logger.warning("Skipping missing %s features: %s", name, path)
            continue
        data = np.load(path, mmap_mode="r")
        if data.ndim == 2 and data.shape[1] == 96:
            data = data[: (len(data) // 16) * 16].reshape(-1, 16, 96)
        if data.ndim != 3 or data.shape[1:] != (16, 96):
            raise ValueError(f"{path} has shape {data.shape}; expected (N, 16, 96)")
        arrays[name] = data
    if not arrays:
        raise FileNotFoundError("No wakeword feature files found")
    positions = {name: 0 for name in arrays}
    while True:
        features, targets = [], []
        for name, data in arrays.items():
            count = per_class.get(name, 0)
            if count <= 0:
                continue
            start = positions[name]
            for sample in data[np.arange(start, start + count) % len(data)]:
                features.append(sample)
                targets.append(labels[name](sample))
            positions[name] = (start + count) % len(data)
        if not features:
            return
        order = np.random.permutation(len(targets))
        yield np.asarray(features, dtype=np.float32)[order], np.asarray(targets, dtype=np.float32)[order]


class WakeWordDataset(IterableDataset):
    def __init__(self, data_files, n_per_class, label_funcs):
        self.data_files, self.n_per_class, self.label_funcs = data_files, n_per_class, label_funcs

    def __iter__(self):
        for features, targets in _batches(self.data_files, self.n_per_class, self.label_funcs):
            yield torch.from_numpy(features.copy()), torch.from_numpy(targets.copy())


def create_dataloader(data_files, n_per_class, label_funcs, prefetch_factor=16, num_workers=0):
    dataset = WakeWordDataset(data_files, n_per_class, label_funcs)
    return DataLoader(dataset, batch_size=None, num_workers=num_workers,
                      prefetch_factor=prefetch_factor if num_workers else None,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
