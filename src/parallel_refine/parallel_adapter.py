"""Expose frozen Parallel representations with a memory-efficient pair readout."""

from __future__ import annotations

import torch
import torch.nn as nn


def _target(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def frozen_parallel_outputs(
        model: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return shared representations and all three task-head predictions.

    The bilinear pair head is evaluated as ``(H W) H^T``.  This is equivalent
    to ``nn.Bilinear`` but avoids the two ``(B,K,K,D)`` expanded intermediates
    used by the production forward implementation.
    """
    target = _target(model)
    padding = ~mask
    hidden = target.encoder(
        target.init_net(x), src_key_padding_mask=padding)
    scores = target.pool_attn(hidden).masked_fill(
        padding.unsqueeze(-1), float("-inf"))
    attention = torch.softmax(scores, dim=1)
    pooled = (hidden * attention).sum(dim=1)

    jet_logits = target.jet_head(pooled)
    origin_logits = target.origin_head(hidden)
    origin_probs = torch.softmax(origin_logits, dim=-1)
    origin_probs = origin_probs * mask.unsqueeze(-1).to(origin_probs.dtype)

    weight = target.pair_head.weight[0]
    pair_logits = torch.matmul(torch.matmul(hidden, weight), hidden.transpose(1, 2))
    if target.pair_head.bias is not None:
        pair_logits = pair_logits + target.pair_head.bias[0]
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
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

