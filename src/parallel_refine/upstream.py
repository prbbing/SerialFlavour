"""Parallel model construction, checkpoint metadata, and frozen readout."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from src.parallel_model import build_parallel_origin_vertex_jet
from src.parallel_refine.config import ParallelRuntimeConfig


def build_parallel(config):
    return build_parallel_origin_vertex_jet(config)


def checkpoint_config(checkpoint: str | Path, fallback: ParallelRuntimeConfig):
    path = Path(checkpoint).parent / "config.json"
    if not path.is_file():
        return fallback
    return ParallelRuntimeConfig(json.loads(path.read_text(encoding="utf-8")))


def _target(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def frozen_parallel_outputs(
        model: nn.Module, track_features: torch.Tensor,
        jet_features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return shared representations and task-head predictions efficiently."""
    target = _target(model)
    hidden, pooled, attention = target.representations(
        track_features, jet_features, mask)
    logits = target.task_logits(hidden, pooled)
    jet_logits = logits["jet_logits"]
    origin_logits = logits["origin_logits"]
    origin_probs = torch.softmax(origin_logits, dim=-1)
    origin_probs = origin_probs * mask.unsqueeze(-1).to(origin_probs.dtype)
    pair_logits = logits["pair_logits"]
    tracks = mask.shape[1]
    off_diagonal = ~torch.eye(
        tracks, dtype=torch.bool, device=mask.device).unsqueeze(0)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & off_diagonal
    pair_probs = torch.sigmoid(pair_logits) * pair_mask.to(pair_logits.dtype)
    return {
        "track_embedding": hidden,
        "pooled_embedding": pooled,
        "pool_attention": attention,
        "track_mask": mask,
        "jet_logits": jet_logits,
        "flavour_probs": torch.softmax(jet_logits, dim=-1),
        "origin_logits": origin_logits,
        "origin_probs": origin_probs,
        "pair_logits": pair_logits,
        "pair_probs": pair_probs,
        "pair_mask": pair_mask,
    }
