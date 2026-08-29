"""Truth-free, structured attention-pooled features for DNN refiners."""

from __future__ import annotations

from dataclasses import dataclass

import torch


FEATURE_SCHEMA_VERSION = "parallel_refine_structured_pool_v3"
EPS = 1e-12


@dataclass(frozen=True)
class FeatureTable:
    values: torch.Tensor
    names: tuple[str, ...]
    groups: dict[str, tuple[int, int]]


def _append_group(pieces, names, groups, group, values, group_names):
    start = sum(piece.shape[-1] for piece in pieces)
    pieces.append(values)
    names.extend(group_names)
    groups[group] = (start, start + values.shape[-1])


def _pair_track_features(
        probabilities: torch.Tensor, embedding: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
    """Return one relation vector per track, excluding diagonal self-pairs.

    The neighbour embedding is a probability-weighted *mean*.  The separate
    match-probability sum retains relation strength/track-multiplicity
    information without allowing it to rescale the neighbour representation.
    """
    _, tracks, _ = probabilities.shape
    eye = torch.eye(tracks, dtype=torch.bool, device=probabilities.device)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye.unsqueeze(0)
    weights = torch.where(pair_mask, probabilities, torch.zeros_like(probabilities))
    probability_sum = weights.sum(dim=-1, keepdim=True)
    neighbour_count = pair_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    probability_mean = probability_sum / neighbour_count.to(probabilities.dtype)
    probability_max = probabilities.masked_fill(~pair_mask, float("-inf")).max(
        dim=-1, keepdim=True).values
    probability_max = torch.where(
        torch.isfinite(probability_max), probability_max,
        torch.zeros_like(probability_max))

    weighted_sum = torch.matmul(weights, embedding)
    weighted_embedding = torch.where(
        probability_sum > 0,
        weighted_sum / probability_sum.clamp(min=EPS),
        torch.zeros_like(weighted_sum))
    return torch.cat([
        weighted_embedding, probability_mean, probability_max, probability_sum,
    ], dim=-1)


def build_feature_table(output: dict[str, torch.Tensor]) -> FeatureTable:
    """Build the complete F4 table without consulting labels or truth fields."""
    mask = output["track_mask"].bool()
    jet_probability = output["flavour_probs"]
    jet_probability_names = tuple(
        f"jet_prob_{name}" for name in ("b", "c", "light"))

    embedding = output["track_embedding"]
    attention = output["pool_attention"]
    embedding_features = output["pooled_embedding"]
    dimension = embedding.shape[-1]
    embedding_names = tuple(f"pooled_{index}" for index in range(dimension))

    origin = output["origin_probs"]
    origin_features = (origin * attention).sum(dim=1)
    classes = origin.shape[-1]
    origin_names = tuple(f"origin_attention_{index}" for index in range(classes))
    pair_per_track = _pair_track_features(
        output["pair_probs"], embedding, mask)
    pair_features = (pair_per_track * attention).sum(dim=1)
    pair_names = tuple(
        [f"pair_weighted_embedding_{index}" for index in range(dimension)]
        + ["pair_match_mean", "pair_match_max", "pair_match_sum"])
    aux = torch.cat([origin_features, pair_features], dim=-1)
    aux_names = origin_names + pair_names

    pieces = []
    names = []
    groups = {}
    _append_group(
        pieces, names, groups, "jet_probability", jet_probability,
        jet_probability_names)
    _append_group(
        pieces, names, groups, "embedding", embedding_features, embedding_names)
    _append_group(pieces, names, groups, "aux", aux, aux_names)
    values = torch.cat(pieces, dim=-1)
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite structured pooled feature encountered")
    return FeatureTable(values, tuple(names), groups)
