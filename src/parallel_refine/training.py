"""Training utilities for the experiment-owned Parallel loop."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.losses import compute_origin_class_weights, pair_vertex_loss


def choose_device(config) -> torch.device:
    if torch.cuda.is_available() and config.gpu_ids != [-1]:
        return torch.device(f"cuda:{config.gpu_ids[0]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def parallel_losses(output, batch, config, origin_criterion):
    jet = nn.functional.cross_entropy(output["jet_logits"], batch["y"])
    origin = output["jet_logits"].new_tensor(0.0)
    if config.lambda_origin:
        origin = origin_criterion(
            output["origin_logits"].reshape(-1, config.n_origin_classes),
            batch["origin"].reshape(-1))
    pair = output["jet_logits"].new_tensor(0.0)
    if config.lambda_pair:
        pair = pair_vertex_loss(
            output["pair_logits"], batch["truth_pair"], batch["mask"])
    total = (
        config.lambda_jet * jet
        + config.lambda_origin * origin
        + config.lambda_pair * pair)
    return {"total": total, "jet": jet, "origin": origin, "pair": pair}


def aggregate_losses(totals, losses, batch_size):
    for name, value in losses.items():
        totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size


@torch.no_grad()
def evaluate_loss(model, loader, device, loss_fn):
    model.eval()
    totals = {}
    count = 0
    jet_correct = 0
    for raw in loader:
        batch = move_batch(raw, device)
        output = model(batch["X"], batch["mask"])
        losses = loss_fn(output, batch)
        aggregate_losses(totals, losses, len(batch["y"]))
        jet_correct += int((
            output["jet_logits"].argmax(dim=-1) == batch["y"]).sum())
        count += len(batch["y"])
    result = {name: value / max(count, 1) for name, value in totals.items()}
    result["jet_accuracy"] = jet_correct / max(count, 1)
    return result


def origin_class_weights(arrays, n_classes: int, device):
    return compute_origin_class_weights(arrays, n_classes, device)


def save_history(history, output_dir, *, metadata=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {"epoch": epoch, **row} for epoch, row in enumerate(history, 1)
    ]
    payload = {
        "history_version": "parallel_refine_training_history_v1",
        "metadata": metadata or {},
        "epochs": rows,
    }
    (output / "training_history.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = list(dict.fromkeys(
        key for row in rows for key in row if key != "epoch"))
    with (output / "training_history.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *keys])
        writer.writeheader()
        writer.writerows(rows)
