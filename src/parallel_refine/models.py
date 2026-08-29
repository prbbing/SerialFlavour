"""Thin adapters around the production Parallel model implementation."""

from __future__ import annotations

import json
from pathlib import Path

from src.parallel_model import build_parallel_origin_vertex_jet

from src.parallel_refine.config import ParallelRuntimeConfig


def build_parallel(config):
    return build_parallel_origin_vertex_jet(config)


def checkpoint_config(checkpoint: str | Path, fallback: ParallelRuntimeConfig):
    path = Path(checkpoint).parent / "config.json"
    if not path.is_file():
        return fallback
    values = json.loads(path.read_text(encoding="utf-8"))
    return ParallelRuntimeConfig(values)
