"""Reusable training and validation primitives for the Parallel base model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.losses import classification_class_weights, pair_vertex_loss


def create_tensorboard_writer(log_dir: str | Path):
    """Create a TensorBoard writer, with an actionable dependency error."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging is enabled but unavailable. Install it with "
            "`pip install tensorboard`, or disable parallel.training.tensorboard."
        ) from error
    return SummaryWriter(str(log_dir))


def log_tensorboard_epoch(
        writer, *, epoch: int, train: dict, validation: dict,
        learning_rate: float, epoch_seconds: float) -> None:
    """Write the stable scalar schema for one Parallel training epoch."""
    for split, metrics in (("train", train), ("validation", validation)):
        for name in ("total", "jet", "origin", "pair"):
            writer.add_scalar(f"loss/{split}/{name}", metrics[name], epoch)
        writer.add_scalar(
            f"metrics/{split}/jet_accuracy", metrics["jet_accuracy"], epoch)
    writer.add_scalar("optimizer/learning_rate", learning_rate, epoch)
    writer.add_scalar("timing/epoch_seconds", epoch_seconds, epoch)
    writer.flush()


def log_tensorboard_scalars(writer, *, step: int, scalars: dict[str, float]) -> None:
    """Write an arbitrary, explicitly named collection of scalar values."""
    for tag, value in scalars.items():
        writer.add_scalar(tag, value, step)
    writer.flush()


def choose_device(config) -> torch.device:
    if torch.cuda.is_available() and config.gpu_ids != [-1]:
        return torch.device(f"cuda:{config.gpu_ids[0]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def parallel_losses(output, batch, config, jet_criterion, origin_criterion):
    jet = jet_criterion(output["jet_logits"], batch["y"])
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
        output = model(batch["X"], batch["jet_X"], batch["mask"])
        aggregate_losses(totals, loss_fn(output, batch), len(batch["y"]))
        jet_correct += int((output["jet_logits"].argmax(-1) == batch["y"]).sum())
        count += len(batch["y"])
    result = {name: value / max(count, 1) for name, value in totals.items()}
    result["jet_accuracy"] = jet_correct / max(count, 1)
    return result


def jet_class_weights(config, device):
    return classification_class_weights(
        config.jet_class_weights, config.n_jet_classes, device)


def origin_class_weights(config, device):
    return classification_class_weights(
        config.origin_class_weights, config.n_origin_classes, device)


def save_history_csv(history, path) -> None:
    """Persist epoch dictionaries as a rectangular CSV training history."""
    rows = list(history)
    keys = list(dict.fromkeys(
        key for row in rows for key in row if key != "epoch"))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *keys])
        writer.writeheader()
        writer.writerows(rows)


def save_history(history, output_dir, *, metadata=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{"epoch": epoch, **row} for epoch, row in enumerate(history, 1)]
    payload = {
        "history_version": "parallel_training_history_v1",
        "metadata": metadata or {},
        "epochs": rows,
    }
    (output / "training_history.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_history_csv(rows, output / "training_history.csv")
