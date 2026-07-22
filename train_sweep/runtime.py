"""Single-process, multi-model training with shared data and CUDA streams.

This module deliberately lives outside :mod:`src`: it composes the existing
SerialFlavour data, model, loss and plotting APIs without changing them.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import load_config
from src.data_fast import create_dataloaders
from src.losses import (
    compute_origin_class_weights,
    pair_vertex_loss,
    vertex_loss_fn,
)
from src.models import build_model, model_requires_pair_target
from src.plotting import (
    plot_c_discriminant_roc,
    plot_discriminant_roc,
    plot_gradient_diagnostics,
    plot_input_variables,
    plot_origin_confusion_matrix,
    plot_output_probabilities,
    plot_pair_vertexing,
    plot_refine_vtx_weight_history,
    plot_track_vertex_assignment,
    plot_training_summary,
    plot_vertex_fit,
    plot_vertex_loss_components,
    plot_vertex_metrics_history,
)
from src.training import (
    _compute_epoch_refine_vtx_stats,
    _compute_vertex_metrics,
    _measure_task_gradients,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


_HISTORY_KEYS = [
    "train_loss", "train_jet_loss", "train_origin_loss",
    "train_vertex_loss", "train_lxy_loss", "train_dz_loss",
    "val_loss", "val_jet_loss", "val_origin_loss",
    "val_vertex_loss", "val_lxy_loss", "val_dz_loss",
    "val_acc", "val_origin_acc",
    "train_refine_mean", "train_vtx_weight_mean",
    "val_b_refine_match_mean", "val_b_refine_other_mean",
    "val_b_vtx_weight_match_mean", "val_b_vtx_weight_other_mean",
    "val_c_refine_match_mean", "val_c_refine_other_mean",
    "val_c_vtx_weight_match_mean", "val_c_vtx_weight_other_mean",
    "grad_norm_shared_encoder_jet", "grad_norm_shared_encoder_origin",
    "grad_norm_shared_encoder_vertex", "grad_norm_vertex_encoder",
    "grad_norm_jet_encoder", "grad_norm_head_jet",
    "grad_norm_head_origin", "grad_norm_head_vtxw",
    "grad_cos_origin_vertex", "grad_cos_origin_jet",
    "grad_cos_vertex_jet",
    "val_b_lxy_mae", "val_c_lxy_mae", "val_b_dz_mae",
    "val_c_dz_mae", "val_b_lxy_pearson", "val_c_lxy_pearson",
    "val_b_dz_pearson", "val_c_dz_pearson",
]

_GRAD_KEYS = [key for key in _HISTORY_KEYS if key.startswith("grad_")]
_REFINE_KEYS = [
    "val_b_refine_match_mean", "val_b_refine_other_mean",
    "val_b_vtx_weight_match_mean", "val_b_vtx_weight_other_mean",
    "val_c_refine_match_mean", "val_c_refine_other_mean",
    "val_c_vtx_weight_match_mean", "val_c_vtx_weight_other_mean",
]
_VERTEX_METRIC_KEYS = [
    "val_b_lxy_mae", "val_c_lxy_mae", "val_b_dz_mae",
    "val_c_dz_mae", "val_b_lxy_pearson", "val_c_lxy_pearson",
    "val_b_dz_pearson", "val_c_dz_pearson",
]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@dataclass
class RunState:
    name: str
    config_path: str
    config: Any
    config_dict: dict[str, Any]
    output_dir: str
    model: nn.Module
    optimiser: torch.optim.Optimizer
    criterion_jet: nn.Module
    criterion_origin: nn.Module
    stream: Any = None
    writer: Any = None
    log_handle: Any = None
    history: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in _HISTORY_KEYS})
    best_jet_loss: float = float("inf")
    best_total_loss: float = float("inf")
    best_jet_epoch: int | None = None
    best_total_epoch: int | None = None
    pred_arrays: dict[str, np.ndarray] | None = None

    def log(self, message: str) -> None:
        """Write legacy-compatible text to file and tagged text to stdout."""
        print(f"[{self.name}] {message}")
        self.write_log(message + "\n")

    def write_log(self, message: str) -> None:
        """Write text only to this run's training log."""
        if self.log_handle is not None:
            self.log_handle.write(message)
            self.log_handle.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _data_signature(config) -> dict[str, Any]:
    """Return every config value that affects shared loader contents/order."""
    return {
        "train_file": os.path.abspath(os.path.expanduser(config.train_file)),
        "cache_dir": os.path.abspath(os.path.expanduser(config.cache_dir)),
        "n_train": config.n_train,
        "n_test": config.n_test,
        "batch_size": config.batch_size,
        "top_k": config.top_k,
        "track_fields": list(config.track_fields),
        "flavour_to_label": dict(config.flavour_to_label),
        "jet_class_names": list(config.jet_class_names),
        "vertex_leg_names": list(config.vertex_leg_names),
        "vertex_targets": _jsonable(config.vertex_targets),
        "use_pair_target": config.use_pair_target,
    }


def validate_sweep_configs(
        entries: list[tuple[str, Any, dict[str, Any]]],
        *, gpu_override: bool = False) -> None:
    """Reject combinations that cannot safely share one DataLoader/GPU."""
    if not entries:
        raise ValueError("At least one --config is required")

    names = [Path(path).stem for path, _, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"Config file stems must be unique: {names}")

    reference_path, reference, _ = entries[0]
    reference_signature = _data_signature(reference)
    reference_gpu_ids = list(reference.gpu_ids)
    if not gpu_override and len(reference_gpu_ids) > 1:
        raise ValueError("Sweep training supports one GPU only; DataParallel is unsupported")

    for path, config, _ in entries:
        if not gpu_override and list(config.gpu_ids) != reference_gpu_ids:
            raise ValueError(
                f"GPU selection differs between {reference_path} and {path}")
        signature = _data_signature(config)
        differing = [key for key in reference_signature
                     if signature[key] != reference_signature[key]]
        if differing:
            raise ValueError(
                f"Config {path} cannot share the DataLoader; differing keys: "
                f"{differing}")
        if model_requires_pair_target(config.model_type) and not config.use_pair_target:
            raise ValueError(
                f"Config {path}: model_type '{config.model_type}' requires "
                "use_pair_target=true")


def resolve_device(config) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_ids = list(config.gpu_ids)
    index = 0 if gpu_ids == [-1] else gpu_ids[0]
    return torch.device(f"cuda:{index}")


def normalise_gpu_request(gpu):
    """Return None, ``auto``, or a non-negative visible CUDA index."""
    if gpu is None:
        return None
    if isinstance(gpu, bool):
        raise ValueError("--gpu must be 'auto' or a non-negative integer")
    if isinstance(gpu, int):
        if gpu < 0:
            raise ValueError("--gpu must be 'auto' or a non-negative integer")
        return gpu
    if isinstance(gpu, str):
        value = gpu.strip().lower()
        if value == "auto":
            return value
        try:
            index = int(value)
        except ValueError as error:
            raise ValueError(
                "--gpu must be 'auto' or a non-negative integer") from error
        if index < 0:
            raise ValueError("--gpu must be 'auto' or a non-negative integer")
        return index
    raise ValueError("--gpu must be 'auto' or a non-negative integer")


def _query_gpu_inventory():
    """Query physical GPU memory without creating CUDA contexts."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        detail = getattr(error, "stderr", None)
        suffix = f": {detail.strip()}" if detail else ""
        raise RuntimeError(
            "Unable to query GPUs with nvidia-smi; use --gpu N to select "
            f"a visible GPU manually{suffix}") from error

    inventory = []
    try:
        for row in csv.reader(completed.stdout.splitlines()):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(f"unexpected nvidia-smi row: {row}")
            inventory.append({
                "physical_index": int(row[0].strip()),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "memory_total_mib": int(row[3].strip()),
                "memory_used_mib": int(row[4].strip()),
                "memory_free_mib": int(row[5].strip()),
            })
    except ValueError as error:
        raise RuntimeError(
            "Unable to parse nvidia-smi output; use --gpu N to select a "
            "visible GPU manually") from error
    if not inventory:
        raise RuntimeError(
            "nvidia-smi returned no GPUs; use --gpu N to select a visible "
            "GPU manually")
    return inventory


def _visible_gpu_inventory(inventory, visible_count, visible_devices=None):
    """Map nvidia-smi physical rows to PyTorch visible CUDA indices."""
    if visible_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    rows_by_index = {row["physical_index"]: row for row in inventory}

    if visible_devices is None or not visible_devices.strip():
        ordered = sorted(inventory, key=lambda row: row["physical_index"])
        return [
            {**row, "visible_index": index}
            for index, row in enumerate(ordered[:visible_count])
        ]

    tokens = [token.strip() for token in visible_devices.split(",")]
    mapped = []
    for visible_index, token in enumerate(tokens[:visible_count]):
        if not token or token == "-1":
            continue
        row = None
        try:
            row = rows_by_index.get(int(token))
        except ValueError:
            matches = [
                item for item in inventory
                if item["uuid"].startswith(token)
            ]
            if len(matches) == 1:
                row = matches[0]
        if row is None:
            raise RuntimeError(
                "Unable to map CUDA_VISIBLE_DEVICES entry "
                f"'{token}' to nvidia-smi output; use --gpu N manually")
        mapped.append({**row, "visible_index": visible_index})
    return mapped


def _selection_metadata(mode, request, device, inventory_row=None):
    visible_index = device.index if device.type == "cuda" else None
    metadata = {
        "mode": mode,
        "request": request,
        "device": str(device),
        "visible_index": visible_index,
        "physical_index": None,
        "uuid": None,
        "name": None,
        "memory_total_mib": None,
        "memory_used_mib": None,
        "memory_free_mib": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type != "cuda":
        return metadata

    row = inventory_row
    if row is None:
        try:
            inventory = _query_gpu_inventory()
            visible = _visible_gpu_inventory(
                inventory, torch.cuda.device_count())
            row = next(
                (item for item in visible
                 if item["visible_index"] == visible_index), None)
        except RuntimeError:
            row = None
    if row is not None:
        for key in (
                "physical_index", "uuid", "name", "memory_total_mib",
                "memory_used_mib", "memory_free_mib"):
            metadata[key] = row[key]
        return metadata

    metadata["name"] = torch.cuda.get_device_name(visible_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(visible_index)
    metadata["memory_free_mib"] = free_bytes // (1024 * 1024)
    metadata["memory_total_mib"] = total_bytes // (1024 * 1024)
    metadata["memory_used_mib"] = (
        metadata["memory_total_mib"] - metadata["memory_free_mib"])
    return metadata


def select_device(config, gpu=None):
    """Resolve a config, manual, or automatic single-device selection."""
    request = normalise_gpu_request(gpu)
    if request is None:
        device = resolve_device(config)
        return device, _selection_metadata(
            "config", list(config.gpu_ids), device)

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--gpu {request} was requested, but CUDA is unavailable")
    visible_count = torch.cuda.device_count()
    if visible_count < 1:
        raise RuntimeError("No CUDA GPUs are visible to this process")

    if request == "auto":
        inventory = _query_gpu_inventory()
        visible = _visible_gpu_inventory(inventory, visible_count)
        if not visible:
            raise RuntimeError(
                "No nvidia-smi GPUs match the CUDA-visible devices; use "
                "--gpu N manually")
        selected = max(
            visible,
            key=lambda row: (row["memory_free_mib"], -row["visible_index"]))
        index = selected["visible_index"]
        device = torch.device(f"cuda:{index}")
        return device, _selection_metadata(
            "auto", "auto", device, selected)

    if request >= visible_count:
        raise ValueError(
            f"--gpu {request} is out of range; this process sees "
            f"{visible_count} CUDA device(s), indexed 0..{visible_count - 1}")
    device = torch.device(f"cuda:{request}")
    return device, _selection_metadata("manual", request, device)


def _apply_gpu_override(entries, gpu_selection):
    """Apply one CLI-selected visible GPU to all in-memory configs."""
    effective_gpu_ids = [gpu_selection["visible_index"]]
    for _, config, config_dict in entries:
        config.gpu_ids = list(effective_gpu_ids)
        config_dict["gpu_ids"] = list(effective_gpu_ids)


def _chunks(items: list[RunState], size: int) -> Iterable[list[RunState]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _move_batch(batch, device: torch.device):
    non_blocking = device.type == "cuda"
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)


def _losses(run: RunState, batch, output):
    X_b, mask_b, y_b, origin_b, lxy_b, dz_b, vvalid_b, pair_b = batch
    jet_loss = run.criterion_jet(output["jet_logits"], y_b)
    if "origin_logits" in output:
        origin_loss = run.criterion_origin(
            output["origin_logits"].reshape(-1, run.config.n_origin_classes),
            origin_b.reshape(-1))
    else:
        origin_loss = output["jet_logits"].new_tensor(0.0)

    lxy_loss = output["jet_logits"].new_tensor(0.0)
    dz_loss = output["jet_logits"].new_tensor(0.0)
    if "lxy_pred" in output:
        vertex_loss, lxy_loss, dz_loss = vertex_loss_fn(
            output["lxy_pred"], output["dz_pred"], lxy_b, dz_b, vvalid_b,
            fit_lxy=run.config.fit_lxy, fit_dz=run.config.fit_dz,
            return_components=True)
        auxiliary_weight = run.config.lambda_vertex
    elif "pair_logits" in output:
        vertex_loss = pair_vertex_loss(output["pair_logits"], pair_b, mask_b)
        auxiliary_weight = run.config.lambda_pair
    else:
        vertex_loss = output["jet_logits"].new_tensor(0.0)
        auxiliary_weight = 0.0

    total_loss = (
        run.config.lambda_jet * jet_loss
        + run.config.lambda_origin * origin_loss
        + auxiliary_weight * vertex_loss)
    return total_loss, jet_loss, origin_loss, vertex_loss, lxy_loss, dz_loss


def _schedule_training_group(group: list[RunState], batch, device):
    pending = []
    if device.type == "cuda":
        default_stream = torch.cuda.current_stream(device)
        ready = torch.cuda.Event()
        ready.record(default_stream)
        for run in group:
            run.stream.wait_event(ready)
            with torch.cuda.stream(run.stream):
                for tensor in batch:
                    tensor.record_stream(run.stream)
                run.optimiser.zero_grad()
                output = run.model(batch[0], batch[1])
                losses = _losses(run, batch, output)
                losses[0].backward()
                run.optimiser.step()
                done = torch.cuda.Event()
                done.record(run.stream)
            pending.append((run, output, losses, done))
        for _, _, _, done in pending:
            done.synchronize()
    else:
        for run in group:
            run.optimiser.zero_grad()
            output = run.model(batch[0], batch[1])
            losses = _losses(run, batch, output)
            losses[0].backward()
            run.optimiser.step()
            pending.append((run, output, losses, None))
    return pending


def train_epoch_sweep(runs, dataloader, device, max_concurrent):
    for run in runs:
        run.model.train()
    totals = {
        run.name: {key: 0.0 for key in (
            "loss", "jet", "origin", "vertex", "lxy", "dz",
            "refine", "vtx_weight", "n")}
        for run in runs
    }

    for cpu_batch in dataloader:
        batch = _move_batch(cpu_batch, device)
        batch_size = len(batch[2])
        for group in _chunks(runs, max_concurrent):
            pending = _schedule_training_group(group, batch, device)
            for run, output, losses, _ in pending:
                values = totals[run.name]
                for key, tensor in zip(
                        ("loss", "jet", "origin", "vertex", "lxy", "dz"),
                        losses):
                    values[key] += tensor.detach().item() * batch_size
                if "refine" in output and "vtx_weight" in output:
                    valid = batch[1].unsqueeze(-1).expand_as(output["refine"])
                    values["refine"] += (
                        output["refine"][valid].float().mean().item() * batch_size)
                    values["vtx_weight"] += (
                        output["vtx_weight"][valid].float().mean().item()
                        * batch_size)
                values["n"] += batch_size
            del pending
        del batch

    metrics = {}
    for run in runs:
        values = totals[run.name]
        n_total = max(values.pop("n"), 1)
        metrics[run.name] = {
            key: value / n_total for key, value in values.items()
        }
    return metrics


def _new_validation_accumulator():
    return {
        "loss": 0.0, "jet": 0.0, "origin": 0.0, "vertex": 0.0,
        "lxy": 0.0, "dz": 0.0, "correct": 0, "origin_correct": 0,
        "origin_total": 0, "n": 0, "all_preds": [], "all_true": [],
        "all_probs": [], "origin_preds": [], "origin_true": [],
        "assignment_probs": [], "lxy_pred": [], "lxy_true": [],
        "dz_pred": [], "dz_true": [], "vtx_valid": [], "vtx_weight": [],
        "origin_full": [], "mask_full": [], "leg_origin_probs": [],
        "gate": [], "refine": [], "pair_logits": [], "pair_target": [],
        "pair_mask": [],
    }


def _schedule_validation_group(group, batch, device):
    pending = []
    if device.type == "cuda":
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
        for run in group:
            run.stream.wait_event(ready)
            with torch.cuda.stream(run.stream):
                for tensor in batch:
                    tensor.record_stream(run.stream)
                output = run.model(batch[0], batch[1])
                losses = _losses(run, batch, output)
                done = torch.cuda.Event()
                done.record(run.stream)
            pending.append((run, output, losses, done))
        for _, _, _, done in pending:
            done.synchronize()
    else:
        for run in group:
            output = run.model(batch[0], batch[1])
            losses = _losses(run, batch, output)
            pending.append((run, output, losses, None))
    return pending


def _accumulate_validation(acc, output, losses, batch):
    _, mask_b, y_b, origin_b, lxy_b, dz_b, vvalid_b, pair_b = batch
    n = len(y_b)
    for key, tensor in zip(("loss", "jet", "origin", "vertex", "lxy", "dz"), losses):
        acc[key] += tensor.detach().item() * n
    preds = output["jet_logits"].argmax(dim=1)
    acc["correct"] += (preds == y_b).sum().item()
    acc["all_preds"].append(preds.cpu())
    acc["all_true"].append(y_b.cpu())
    acc["all_probs"].append(torch.softmax(output["jet_logits"], dim=1).cpu())

    if "origin_logits" in output:
        origin_preds = output["origin_logits"].argmax(dim=-1)
        origin_mask = origin_b >= 0
        acc["origin_correct"] += (
            (origin_preds == origin_b) & origin_mask).sum().item()
        acc["origin_total"] += origin_mask.sum().item()
        acc["origin_preds"].append(origin_preds[origin_mask].cpu())
        acc["origin_true"].append(origin_b[origin_mask].cpu())
    if "assignment_probs" in output:
        acc["assignment_probs"].append(output["assignment_probs"].cpu())
    if "lxy_pred" in output:
        acc["lxy_pred"].append(output["lxy_pred"].cpu())
        acc["lxy_true"].append(lxy_b.cpu())
        acc["dz_pred"].append(output["dz_pred"].cpu())
        acc["dz_true"].append(dz_b.cpu())
        acc["vtx_valid"].append(vvalid_b.cpu())
    if "vtx_weight" in output:
        acc["vtx_weight"].append(output["vtx_weight"].cpu())
        acc["origin_full"].append(origin_b.cpu())
        acc["mask_full"].append(mask_b.cpu())
    if "leg_origin_probs" in output:
        acc["leg_origin_probs"].append(output["leg_origin_probs"].cpu())
        acc["gate"].append(output["gate"].cpu())
        acc["refine"].append(output["refine"].cpu())
    if "pair_logits" in output:
        acc["pair_logits"].append(output["pair_logits"].cpu())
        acc["pair_target"].append(pair_b.cpu())
        acc["pair_mask"].append(mask_b.cpu())
    acc["n"] += n


def _finish_validation(acc):
    n = max(acc["n"], 1)
    metrics = {key: acc[key] / n for key in (
        "loss", "jet", "origin", "vertex", "lxy", "dz")}
    metrics["acc"] = acc["correct"] / n
    metrics["origin_acc"] = (
        acc["origin_correct"] / acc["origin_total"]
        if acc["origin_total"] else float("nan"))
    arrays = {
        "all_preds": torch.cat(acc["all_preds"]).numpy(),
        "all_true": torch.cat(acc["all_true"]).numpy(),
        "all_probs": torch.cat(acc["all_probs"]).numpy(),
    }
    for key in (
        "origin_preds", "origin_true", "assignment_probs", "lxy_pred",
        "lxy_true", "dz_pred", "dz_true", "vtx_valid", "vtx_weight",
        "origin_full", "mask_full", "leg_origin_probs", "gate", "refine",
        "pair_logits", "pair_target", "pair_mask",
    ):
        if acc[key]:
            value = torch.cat(acc[key]).numpy()
            if key in {"vtx_valid", "mask_full", "pair_mask"}:
                value = value.astype(bool)
            arrays[key] = value
    return metrics, arrays


@torch.no_grad()
def validate_epoch_sweep(runs, dataloader, device, max_concurrent):
    for run in runs:
        run.model.eval()
    accumulators = {run.name: _new_validation_accumulator() for run in runs}
    for cpu_batch in dataloader:
        batch = _move_batch(cpu_batch, device)
        for group in _chunks(runs, max_concurrent):
            pending = _schedule_validation_group(group, batch, device)
            for run, output, losses, _ in pending:
                _accumulate_validation(
                    accumulators[run.name], output, losses, batch)
            del pending
        del batch
    return {
        run.name: _finish_validation(accumulators[run.name]) for run in runs
    }


def _append_epoch(run, train_metrics, val_metrics, pred_arrays, grad_stats):
    mapping = {
        "train_loss": train_metrics["loss"],
        "train_jet_loss": train_metrics["jet"],
        "train_origin_loss": train_metrics["origin"],
        "train_vertex_loss": train_metrics["vertex"],
        "train_lxy_loss": train_metrics["lxy"],
        "train_dz_loss": train_metrics["dz"],
        "train_refine_mean": train_metrics["refine"],
        "train_vtx_weight_mean": train_metrics["vtx_weight"],
        "val_loss": val_metrics["loss"],
        "val_jet_loss": val_metrics["jet"],
        "val_origin_loss": val_metrics["origin"],
        "val_vertex_loss": val_metrics["vertex"],
        "val_lxy_loss": val_metrics["lxy"],
        "val_dz_loss": val_metrics["dz"],
        "val_acc": val_metrics["acc"],
        "val_origin_acc": val_metrics["origin_acc"],
    }
    for key, value in mapping.items():
        run.history[key].append(float(value))
    for key in _GRAD_KEYS:
        run.history[key].append(float(grad_stats.get(key, float("nan"))))

    refine_stats = (
        _compute_epoch_refine_vtx_stats(pred_arrays, run.config)
        if "refine" in pred_arrays else {})
    for key in _REFINE_KEYS:
        run.history[key].append(float(refine_stats.get(key, 0.0)))
    vertex_stats = (
        _compute_vertex_metrics(pred_arrays, run.config)
        if "lxy_pred" in pred_arrays else {})
    for key in _VERTEX_METRIC_KEYS:
        run.history[key].append(float(vertex_stats.get(key, 0.0)))


def _save_epoch(run: RunState, epoch: int, val_metrics):
    if val_metrics["jet"] < run.best_jet_loss:
        run.best_jet_loss = val_metrics["jet"]
        run.best_jet_epoch = epoch
        torch.save(run.model.state_dict(), os.path.join(run.output_dir, "best_jet.pt"))
        run.log(
            f"Saved best_jet.pt (epoch={epoch}, "
            f"val_jet_loss={val_metrics['jet']:.6f})")
    if val_metrics["loss"] < run.best_total_loss:
        run.best_total_loss = val_metrics["loss"]
        run.best_total_epoch = epoch
        torch.save(run.model.state_dict(), os.path.join(run.output_dir, "best_total.pt"))
        run.log(
            f"Saved best_total.pt (epoch={epoch}, "
            f"val_loss={val_metrics['loss']:.6f})")
    if epoch % run.config.checkpoint_interval == 0:
        torch.save(run.model.state_dict(), os.path.join(
            run.output_dir, f"epoch_{epoch}.pt"))
        run.log(f"Saved epoch_{epoch}.pt")
    torch.save(run.model.state_dict(), os.path.join(run.output_dir, "last.pt"))


def _log_startup(run: RunState, gpu_ids, data_log):
    """Reproduce the legacy pre-data/model portion of the training log."""
    run.log(f"Device: {list(gpu_ids)}  |  DataParallel: False")
    run.log(f"Config saved to {os.path.join(run.output_dir, 'config.json')}")
    run.log("Loading data...")
    run.write_log(data_log)
    if data_log and not data_log.endswith("\n"):
        run.write_log("\n")


def _log_model_setup(run: RunState, device, train_size, test_size):
    """Reproduce the legacy post-data model and optimiser log section."""
    config = run.config
    run.log(f"Model type: {config.model_type}")
    if config.model_type == "staged_origin_vertex_jet":
        run.log(f"  Vertex fit coordinates: {config.vertex_fit_coords}")
        run.log(f"  Stage-3 extra inputs: {config.stage3_extra_inputs}")
        run.log(
            f"  Stage-3 tagging fields ({len(config.tagging_fields)}): "
            f"{config.tagging_fields}")
        run.log(
            "  Calibrate vertex fit (learnable per-leg scale): "
            f"{config.calibrate_vertex_fit}")

    run.log("Origin class weights:")
    weights = run.criterion_origin.weight.detach().cpu()
    for index, (name, weight) in enumerate(zip(
            config.origin_class_names, weights)):
        run.log(f"  {index:2d}  {name:<18s}  {weight:.3f}")

    n_params = sum(
        parameter.numel() for parameter in run.model.parameters()
        if parameter.requires_grad)
    run.log(f"Parameters: {n_params:,}")
    run.log(
        f"Device: {device}  |  Train: {train_size:,}  |  Test: {test_size:,}")
    run.write_log("\n")
    if config.tensorboard_log_dir and SummaryWriter is None:
        run.log("TensorBoard is unavailable; continuing without SummaryWriter")


def _log_epoch(
        run, epoch, train_metrics, val_metrics, pred_arrays, grad_stats,
        elapsed_seconds):
    """Emit legacy detail for boundary epochs and a compact middle summary."""
    config = run.config
    detailed = epoch <= 5 or epoch > config.epochs - 5
    if not detailed:
        run.log(
            f"Epoch {epoch:02d}/{config.epochs}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  "
            f"origin_acc={val_metrics['origin_acc']:.4f}  "
            f"epoch_seconds={elapsed_seconds:.2f}")
        return

    calibration = ""
    if (config.calibrate_vertex_fit
            and hasattr(run.model, "calibration_scales")):
        parts = []
        for coord, values in run.model.calibration_scales().items():
            per_leg = ", ".join(
                f"{name}={value:.3f}"
                for name, value in zip(config.vertex_leg_names, values))
            parts.append(f"{coord}_scale=[{per_leg}]")
        if parts:
            calibration = "  " + "  ".join(parts)

    run.log(
        f"Epoch {epoch:02d}/{config.epochs}  "
        f"loss={train_metrics['loss']:.4f} (jet={train_metrics['jet']:.4f} "
        f"origin={train_metrics['origin']:.4f} "
        f"vtx={train_metrics['vertex']:.4f}"
        f"(Lxy={train_metrics['lxy']:.4f},dz={train_metrics['dz']:.4f}))  "
        f"val_loss={val_metrics['loss']:.4f} (jet={val_metrics['jet']:.4f} "
        f"origin={val_metrics['origin']:.4f} "
        f"vtx={val_metrics['vertex']:.4f}"
        f"(Lxy={val_metrics['lxy']:.4f},dz={val_metrics['dz']:.4f}))  "
        f"val_acc={val_metrics['acc']:.4f}  "
        f"origin_acc={val_metrics['origin_acc']:.4f}{calibration}")

    if "refine" in pred_arrays:
        history = run.history
        run.log(
            f"         refine train={train_metrics['refine']:.4f}  "
            f"val b(m={history['val_b_refine_match_mean'][-1]:.4f} "
            f"o={history['val_b_refine_other_mean'][-1]:.4f})  "
            f"c(m={history['val_c_refine_match_mean'][-1]:.4f} "
            f"o={history['val_c_refine_other_mean'][-1]:.4f})  "
            f"vtx_w b(m={history['val_b_vtx_weight_match_mean'][-1]:.4f} "
            f"o={history['val_b_vtx_weight_other_mean'][-1]:.4f})  "
            f"c(m={history['val_c_vtx_weight_match_mean'][-1]:.4f} "
            f"o={history['val_c_vtx_weight_other_mean'][-1]:.4f})")

    if "lxy_pred" in pred_arrays:
        parts = []
        for leg_name in config.vertex_leg_names:
            short_name = leg_name.replace("_vertex", "")
            values = []
            if config.fit_lxy:
                values.append(
                    f"Lxy MAE={run.history[f'val_{short_name}_lxy_mae'][-1]:.4f} "
                    f"r={run.history[f'val_{short_name}_lxy_pearson'][-1]:.3f}")
            if config.fit_dz:
                values.append(
                    f"dz MAE={run.history[f'val_{short_name}_dz_mae'][-1]:.4f} "
                    f"r={run.history[f'val_{short_name}_dz_pearson'][-1]:.3f}")
            if values:
                parts.append(f"{short_name}: " + "  ".join(values))
        if parts:
            run.log("    vtx metrics  " + " | ".join(parts))

    if grad_stats:
        if "grad_norm_shared_encoder_origin" in grad_stats:
            run.log(
                "    grad  shared: "
                f"jet={grad_stats.get('grad_norm_shared_encoder_jet', 0):.4f}  "
                f"origin={grad_stats['grad_norm_shared_encoder_origin']:.4f}  "
                f"vertex={grad_stats.get('grad_norm_shared_encoder_vertex', 0):.4f}")
        else:
            run.log(
                "    grad  assignment: "
                f"jet={grad_stats.get('grad_norm_shared_encoder_jet', 0):.4f}  "
                f"vertex={grad_stats.get('grad_norm_shared_encoder_vertex', 0):.4f}")
        if "grad_norm_vertex_encoder" in grad_stats:
            run.log(
                f"          vertex_enc={grad_stats['grad_norm_vertex_encoder']:.4f}  "
                f"jet_enc={grad_stats.get('grad_norm_jet_encoder', 0):.4f}")
        if "grad_norm_head_origin" in grad_stats:
            run.log(
                f"          heads: jet={grad_stats.get('grad_norm_head_jet', 0):.4f}  "
                f"origin={grad_stats['grad_norm_head_origin']:.4f}  "
                f"vtxw={grad_stats.get('grad_norm_head_vtxw', 0):.4f}")
            run.log(
                f"          cos: orig-vtx={grad_stats.get('grad_cos_origin_vertex', 0):.3f}  "
                f"orig-jet={grad_stats.get('grad_cos_origin_jet', 0):.3f}  "
                f"vtx-jet={grad_stats.get('grad_cos_vertex_jet', 0):.3f}")
        else:
            run.log(
                f"          heads: jet={grad_stats.get('grad_norm_head_jet', 0):.4f}  "
                f"assignment={grad_stats.get('grad_norm_head_vtxw', 0):.4f}")
            run.log(
                f"          cos: vertex-jet="
                f"{grad_stats.get('grad_cos_vertex_jet', 0):.3f}")


def _log_checkpoint_summary(run: RunState):
    run.log("Checkpoint summary:")
    run.log(
        f"  best_jet.pt: epoch={run.best_jet_epoch}, "
        f"val_jet_loss={run.best_jet_loss:.6f}")
    run.log(
        f"  best_total.pt: epoch={run.best_total_epoch}, "
        f"val_loss={run.best_total_loss:.6f}")
    run.log(f"  last.pt: epoch={run.config.epochs}")
    run.log(
        f"  periodic checkpoints: every "
        f"{run.config.checkpoint_interval} epoch(s)")


def _write_history(run: RunState):
    json_path = os.path.join(run.output_dir, "training_history.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(run.history), handle, indent=2, allow_nan=True)
    csv_path = os.path.join(run.output_dir, "training_history.csv")
    epochs = len(run.history["train_loss"])
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        keys = list(run.history)
        writer.writerow(["epoch", *keys])
        for index in range(epochs):
            writer.writerow([index + 1, *[run.history[key][index] for key in keys]])
    grad_path = os.path.join(run.output_dir, "gradient_diagnostics.csv")
    with open(grad_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *_GRAD_KEYS])
        for index in range(epochs):
            writer.writerow([index + 1, *[
                run.history[key][index] for key in _GRAD_KEYS]])
    if epochs:
        run.log("Exported gradient_diagnostics.csv")


def _run_evaluation(run: RunState):
    arrays = run.pred_arrays
    config = run.config
    plot_dir = run.output_dir
    print("\nJet classification report:")
    print(classification_report(
        arrays["all_true"], arrays["all_preds"],
        target_names=config.jet_class_names, zero_division=0))
    print(confusion_matrix(arrays["all_true"], arrays["all_preds"]))
    if "origin_preds" in arrays:
        print("\nTrack-origin classification report:")
        print(classification_report(
            arrays["origin_true"], arrays["origin_preds"],
            target_names=config.origin_class_names,
            labels=list(range(config.n_origin_classes)), zero_division=0))
    if "assignment_probs" in arrays:
        assignment_true = np.full(
            arrays["origin_full"].shape, config.n_vertex_legs, dtype=np.int64)
        for leg, leg_name in enumerate(config.vertex_leg_names):
            origin_names = config.vertex_legs[leg_name]
            if isinstance(origin_names, str):
                origin_names = [origin_names]
            origin_ids = [config.origin_class_names.index(name)
                          for name in origin_names]
            assignment_true[np.isin(arrays["origin_full"], origin_ids)] = leg
        valid = arrays["mask_full"] & (arrays["origin_full"] >= 0)
        assignment_pred = arrays["assignment_probs"].argmax(axis=-1)
        assignment_names = [*config.vertex_leg_names, "other"]
        print("\nVertex-assignment classification report:")
        print(classification_report(
            assignment_true[valid], assignment_pred[valid],
            target_names=assignment_names,
            labels=list(range(config.n_vertex_legs + 1)), zero_division=0))
        print(confusion_matrix(
            assignment_true[valid], assignment_pred[valid],
            labels=list(range(config.n_vertex_legs + 1))))

    plot_training_summary(
        run.history, arrays["all_true"], arrays["all_preds"],
        config.jet_class_names, plot_dir, len(run.history["train_loss"]),
        model_type=config.model_type)
    plot_refine_vtx_weight_history(run.history, plot_dir)
    plot_gradient_diagnostics(run.history, plot_dir)
    plot_vertex_metrics_history(run.history, plot_dir)
    plot_vertex_loss_components(run.history, plot_dir)
    plot_output_probabilities(
        arrays["all_probs"], arrays["all_true"], config.jet_class_names,
        config.colours, plot_dir)
    if "origin_preds" in arrays:
        plot_origin_confusion_matrix(
            arrays["origin_true"], arrays["origin_preds"],
            config.origin_class_names, plot_dir)
    plot_discriminant_roc(
        arrays["all_probs"], arrays["all_true"], config.jet_class_names,
        config.disc_bkg_weights, config.colours, plot_dir)
    plot_c_discriminant_roc(
        arrays["all_probs"], arrays["all_true"], config.jet_class_names,
        config.c_disc_bkg_weights, config.colours, plot_dir)
    if "lxy_pred" in arrays:
        plot_vertex_fit(
            arrays["lxy_pred"], arrays["lxy_true"], arrays["dz_pred"],
            arrays["dz_true"], arrays["vtx_valid"], arrays["all_true"],
            config, plot_dir)
    if "vtx_weight" in arrays:
        plot_track_vertex_assignment(
            arrays["vtx_weight"], arrays["origin_full"], arrays["mask_full"],
            arrays["all_true"], config.origin_class_names,
            config.vertex_leg_names, config.vertex_legs, config.n_vertex_legs,
            config.leg_owner_cls, config.jet_class_names, config.colours,
            plot_dir, leg_origin_probs=arrays.get("leg_origin_probs"),
            gate=arrays.get("gate"), refine=arrays.get("refine"))
    if "pair_logits" in arrays:
        plot_pair_vertexing(
            arrays["pair_logits"], arrays["pair_target"], arrays["pair_mask"],
            config.jet_class_names, arrays["all_true"], config.colours,
            plot_dir)
    print(f"\nAll outputs saved to {plot_dir}")


def _write_manifest(manifest, runs):
    for run in runs:
        path = os.path.join(run.output_dir, "sweep_manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(manifest), handle, indent=2, allow_nan=True)


def _resolve_output_dirs(entries, timestamp, *, require_absent=False):
    resolved_outputs = []
    for path, config, config_dict in entries:
        output_dir = os.path.abspath(
            f"{config.plot_dir.rstrip('/\\')}_{timestamp}") + os.sep
        resolved_outputs.append((path, output_dir))
    normalised = [os.path.normcase(path) for _, path in resolved_outputs]
    if len(set(normalised)) != len(normalised):
        raise ValueError("Resolved training output directories must be unique")
    if require_absent:
        existing = [path for _, path in resolved_outputs if os.path.exists(path)]
        if existing:
            raise FileExistsError(
                f"Sweep output directories already exist: {existing}")
    return resolved_outputs


def _build_runs(entries, train_data, device, timestamp, seed):
    runs = []
    resolved_outputs = _resolve_output_dirs(entries, timestamp)

    for (path, config, config_dict), (_, output_dir) in zip(
            entries, resolved_outputs):
        name = Path(path).stem
        config.plot_dir = output_dir
        config.num_workers = 0
        config_dict = dict(config_dict)
        config_dict["train_plot_dir"] = output_dir
        config_dict["num_workers"] = 0
        os.makedirs(output_dir, exist_ok=False)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config_dict, handle, indent=4)

        torch.manual_seed(seed)
        model = build_model(config).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion_jet = nn.CrossEntropyLoss()
        origin_weights = compute_origin_class_weights(
            train_data, config.n_origin_classes, device)
        criterion_origin = nn.CrossEntropyLoss(
            ignore_index=-1, weight=origin_weights)
        stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        tb_dir = None
        if config.tensorboard_log_dir:
            tb_dir = os.path.join(
                config.tensorboard_log_dir.rstrip("/"),
                os.path.basename(output_dir.rstrip("/\\")))
        writer = (
            SummaryWriter(tb_dir)
            if tb_dir and SummaryWriter is not None else None)
        log_handle = open(
            os.path.join(output_dir, "training_log.md"), "w", encoding="utf-8")
        runs.append(RunState(
            name=name, config_path=os.path.abspath(path), config=config,
            config_dict=config_dict, output_dir=output_dir, model=model,
            optimiser=optimiser, criterion_jet=criterion_jet,
            criterion_origin=criterion_origin, stream=stream, writer=writer,
            log_handle=log_handle))

    return runs


def run_sweep(config_paths, max_concurrent=None, seed=42, gpu=None):
    """Train all configs in one process and return their completed states."""
    entries = []
    for path in config_paths:
        config, config_dict = load_config(path)
        entries.append((path, config, config_dict))
    gpu_request = normalise_gpu_request(gpu)
    validate_sweep_configs(entries, gpu_override=gpu_request is not None)
    count = len(entries)
    if max_concurrent is None:
        max_concurrent = count
    if not 1 <= max_concurrent <= count:
        raise ValueError(f"max_concurrent must lie in [1, {count}]")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device, gpu_selection = select_device(entries[0][1], gpu_request)
    if gpu_request is not None:
        _apply_gpu_override(entries, gpu_selection)
    execution_mode = "cuda_streams" if device.type == "cuda" else "cpu_sequential"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _resolve_output_dirs(entries, timestamp, require_absent=True)

    # Mutate only in-memory Config objects. Existing JSON and src files remain untouched.
    for _, config, config_dict in entries:
        config.num_workers = 0
        config_dict["num_workers"] = 0
    base_config = entries[0][1]
    data_log_buffer = io.StringIO()
    with redirect_stdout(_Tee(sys.stdout, data_log_buffer)):
        train_loader, val_loader, train_data, _, y_train, y_test = (
            create_dataloaders(base_config, str(device)))
    data_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_loader.dataset, batch_size=base_config.batch_size, shuffle=True,
        pin_memory=device.type == "cuda", num_workers=0,
        generator=data_generator)
    val_loader = DataLoader(
        val_loader.dataset, batch_size=base_config.batch_size, shuffle=False,
        pin_memory=device.type == "cuda", num_workers=0)

    runs = _build_runs(entries, train_data, device, timestamp, seed)
    manifest = {
        "status": "running", "timestamp": timestamp,
        "ordered_configs": [os.path.abspath(path) for path in config_paths],
        "runs": {run.name: run.output_dir for run in runs},
        "seed": seed, "device": str(device), "execution_mode": execution_mode,
        "gpu_selection": gpu_selection,
        "max_concurrent": max_concurrent, "num_workers": 0,
        "epoch_times_seconds": [],
    }
    _write_manifest(manifest, runs)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        logged_gpu_ids = (
            [gpu_selection["visible_index"]]
            if device.type == "cuda" else [0])
        for run in runs:
            _log_startup(
                run, logged_gpu_ids, data_log_buffer.getvalue())
            with redirect_stdout(_Tee(sys.stdout, run.log_handle)):
                plot_input_variables(
                    train_data["X"], train_data["mask"], train_data["y"],
                    run.config.track_fields, run.config.top_k,
                    run.config.jet_class_names, run.config.colours,
                    run.output_dir)
            _log_model_setup(
                run, device, len(y_train), len(y_test))

        diagnostic_cpu = next(iter(val_loader))
        diagnostic_batch = _move_batch(diagnostic_cpu, device)
        max_epochs = max(run.config.epochs for run in runs)
        torch.manual_seed(seed + 1)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 1)

        for epoch in range(1, max_epochs + 1):
            active = [run for run in runs if epoch <= run.config.epochs]
            epoch_started = time.perf_counter()
            train_results = train_epoch_sweep(
                active, train_loader, device, max_concurrent)

            grad_results = {}
            for run in active:
                if device.type == "cuda":
                    torch.cuda.current_stream(device).wait_stream(run.stream)
                grad_results[run.name] = _measure_task_gradients(
                    run.model, *diagnostic_batch, run.criterion_jet,
                    run.criterion_origin, run.config.n_origin_classes,
                    run.config.fit_lxy, run.config.fit_dz)

            validation_results = validate_epoch_sweep(
                active, val_loader, device, max_concurrent)
            elapsed = time.perf_counter() - epoch_started
            manifest["epoch_times_seconds"].append(elapsed)

            for run in active:
                val_metrics, arrays = validation_results[run.name]
                _append_epoch(
                    run, train_results[run.name], val_metrics, arrays,
                    grad_results[run.name])
                run.pred_arrays = arrays
                _save_epoch(run, epoch, val_metrics)
                if run.writer is not None:
                    for key, values in run.history.items():
                        value = values[-1]
                        if value == value:
                            run.writer.add_scalar(key, value, epoch)
                    run.writer.flush()
                _log_epoch(
                    run, epoch, train_results[run.name], val_metrics, arrays,
                    grad_results[run.name], elapsed)
            _write_manifest(manifest, runs)

        for run in runs:
            _log_checkpoint_summary(run)
            _write_history(run)
            with redirect_stdout(_Tee(sys.stdout, run.log_handle)):
                _run_evaluation(run)

        manifest["status"] = "complete"
        manifest["total_seconds"] = time.perf_counter() - started
        if device.type == "cuda":
            manifest["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        _write_manifest(manifest, runs)
        return runs
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        manifest["total_seconds"] = time.perf_counter() - started
        _write_manifest(manifest, runs)
        raise
    finally:
        for run in runs:
            run.close()
