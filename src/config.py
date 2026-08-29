"""Shared defaults and deterministic random-number helpers for Parallel."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch


_DEFAULTS = {
    "model_type": "parallel_origin_vertex_jet",
    "gpu_ids": [-1],
    "top_k": 40,
    "batch_size": 512,
    "use_pair_target": True,
    "d_model": 32,
    "n_heads": 2,
    "n_layers": 4,
    "d_ffn": 64,
    "dropout": 0.1,
    "n_origin_classes": 8,
    "n_jet_classes": 3,
    "jet_fields": ["pt_btagJes", "eta_btagJes"],
    "truth_vertex": {
        "field": "ftagTruthVertexIndex",
        "premerged": True,
        "merge_distance_mm": 0.1,
    },
    "track_fields": [
        "d0", "z0SinTheta", "dphi", "deta", "qOverP",
        "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
        "phiUncertainty", "thetaUncertainty", "qOverPUncertainty",
        "numberOfPixelHits", "numberOfSCTHits",
        "numberOfInnermostPixelLayerHits",
        "numberOfNextToInnermostPixelLayerHits",
        "numberOfInnermostPixelLayerSharedHits",
        "numberOfInnermostPixelLayerSplitHits",
        "numberOfPixelSharedHits", "numberOfPixelSplitHits",
        "numberOfSCTSharedHits",
    ],
    "flavour_to_label": {"5": 0, "4": 1, "0": 2},
    "jet_class_names": ["b-jet", "c-jet", "light-jet"],
    "origin_class_names": [
        "Pileup", "Fake", "Primary", "From b", "From b->c", "From c",
        "From tau", "Other secondary",
    ],
    "track_projection_dim": None,
    "task_head_hidden_dims": [16],
    "seed": 42,
    "data_seed": 42,
    "num_workers": 2,
    "lambda_jet": 1.0,
    "lambda_origin": 1.0,
    "lambda_pair": 1.0,
    "jet_class_weights": [2.0, 2.0, 1.0],
    "origin_class_weights": [
        3.08, 566.04, 1.0, 10.16, 6.56, 7.53, 83.66, 19.91,
    ],
    "tensorboard_enabled": True,
    "tensorboard_subdir": "tensorboard",
}


def seed_everything(seed: int, cuda_devices: Iterable[int] = ()) -> None:
    """Seed Python, NumPy, CPU PyTorch, and selected CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        for device_index in dict.fromkeys(cuda_devices):
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(seed)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    """Return an isolated generator for reproducible sampler order."""
    return torch.Generator().manual_seed(seed)
