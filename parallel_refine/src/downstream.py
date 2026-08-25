"""Common tabular data and DNN definitions for DNN/BDT comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from parallel_refine.src.cache import FrozenFeatureCache


def fit_normalization(
        cache: FrozenFeatureCache, columns: np.ndarray,
        *, chunk_size: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(columns), dtype=np.float64)
    square = np.zeros(len(columns), dtype=np.float64)
    count = 0
    for start in range(0, len(cache.labels), chunk_size):
        stop = min(start + chunk_size, len(cache.labels))
        values = np.asarray(cache.features[start:stop, columns], dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite cached feature encountered")
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std > 1e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


class CachedTabularDataset(Dataset):
    def __init__(self, cache: FrozenFeatureCache, columns: np.ndarray):
        self.cache = cache
        self.columns = np.asarray(columns, dtype=np.int64)

    def __len__(self):
        return len(self.cache.labels)

    def __getitem__(self, index):
        features = np.array(
            self.cache.features[index, self.columns], dtype=np.float32, copy=True)
        label = int(self.cache.labels[index])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


def create_tabular_loader(
        cache, columns, *, batch_size, shuffle, num_workers, seed):
    from src.config import dataloader_generator, seed_dataloader_worker

    return DataLoader(
        CachedTabularDataset(cache, columns), batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=dataloader_generator(seed),
        worker_init_fn=seed_dataloader_worker)


class TabularDNN(nn.Module):
    def __init__(
            self, input_dim: int, hidden_dims, dropout: float,
            mean, std, n_classes: int = 3):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        layers = []
        current = input_dim
        for width in hidden_dims:
            layers.extend([
                nn.Linear(current, int(width)), nn.ReLU(), nn.Dropout(dropout),
            ])
            current = int(width)
        layers.append(nn.Linear(current, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, values):
        return self.classifier((values - self.mean) / self.std)


def save_dnn_description(path, *, recipe, columns, feature_names, config):
    payload = {
        "model_type": "tabular_dnn",
        "recipe": recipe,
        "columns": np.asarray(columns, dtype=np.int64).tolist(),
        "feature_names": list(feature_names),
        "input_dim": int(len(columns)),
        "hidden_dims": list(config["hidden_dims"]),
        "dropout": float(config["dropout"]),
        "n_classes": 3,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dnn(directory: str | Path, device: torch.device):
    directory = Path(directory)
    description = json.loads(
        (directory / "model.json").read_text(encoding="utf-8"))
    normalization = np.load(directory / "normalization.npz")
    model = TabularDNN(
        description["input_dim"], description["hidden_dims"],
        description["dropout"], normalization["mean"], normalization["std"],
        description["n_classes"]).to(device)
    model.load_state_dict(torch.load(
        directory / "best_dnn.pt", map_location=device, weights_only=True))
    model.eval()
    return model, description

