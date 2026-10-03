"""A task-neutral supervised loop with validation-only model selection.

loss(model, batch) returns (scalar_loss, number_of_samples).
score(model, loader, device) returns the chosen validation metric.
Domain adapters own inputs, objectives, prediction units, and metric semantics.
"""

import csv
import math
from pathlib import Path
import time

import torch

from pipeline.io import write_json
from pipeline.runtime import to_device


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit(model, train_loader, val_loader, loss, score, directory, settings, metadata, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
    )
    mode = settings.get("selection_mode", "min")
    if mode not in ("min", "max"):
        raise ValueError("selection_mode must be min or max")
    patience = int(settings.get("early_stopping_patience", 0) or 0)
    min_delta = float(settings.get("min_delta", 0.0) or 0.0)
    scheduler = None
    scheduler_settings = settings.get("scheduler")
    resolved_scheduler = None
    if scheduler_settings is not None:
        if scheduler_settings.get("type") != "reduce_on_plateau":
            raise ValueError("scheduler.type must be reduce_on_plateau")
        factor = float(scheduler_settings.get("factor", 0.5))
        scheduler_patience = scheduler_settings.get("patience", 8)
        min_lr_ratio = float(scheduler_settings.get("min_lr_ratio", 0.01))
        threshold = float(scheduler_settings.get("threshold", min_delta))
        if not 0.0 < factor < 1.0 or not 0.0 < min_lr_ratio <= 1.0:
            raise ValueError("scheduler factor and min_lr_ratio must be in (0, 1) and (0, 1]")
        if isinstance(scheduler_patience, bool) or not isinstance(scheduler_patience, int) or scheduler_patience < 0:
            raise ValueError("scheduler patience must be a non-negative integer")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("scheduler threshold must be finite and non-negative")
        if patience and scheduler_patience >= patience:
            raise ValueError("scheduler patience must be shorter than early stopping patience")
        resolved_scheduler = {
            "type": "reduce_on_plateau", "mode": mode, "factor": factor,
            "patience": scheduler_patience, "threshold": threshold, "threshold_mode": "abs",
            "min_lr_ratio": min_lr_ratio, "min_lr": float(settings["learning_rate"]) * min_lr_ratio,
            "cooldown": 0, "eps": 1e-8,
        }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=scheduler_patience,
            threshold=threshold, threshold_mode="abs", min_lr=resolved_scheduler["min_lr"],
            cooldown=0, eps=1e-8,
        )
    best = math.inf if mode == "min" else -math.inf
    best_epoch = 0
    history = []
    stopped = False
    started = time.perf_counter()
    initial_validation = None
    if settings.get("include_initial_checkpoint", False):
        model.eval()
        with torch.inference_mode():
            initial_validation = float(score(model, val_loader, device))
        if not math.isfinite(initial_validation):
            raise FloatingPointError("non-finite initial validation metric")
        best = initial_validation
        if scheduler is not None:
            scheduler.step(initial_validation)
        save_checkpoint(directory / "best.pt", {
            "state_dict": model.state_dict(), "metadata": metadata, "epoch": 0,
            "validation_metric": best, "selection_mode": mode,
        })
    for epoch in range(1, settings["epochs"] + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            objective, samples = loss(model, batch)
            if not torch.isfinite(objective):
                raise FloatingPointError("non-finite training objective")
            objective.backward()
            if settings.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["clip_grad"]))
            optimizer.step()
            total += float(objective.detach()) * samples
            count += samples
        if not count:
            raise ValueError("empty training loader")
        model.eval()
        with torch.inference_mode():
            validation = float(score(model, val_loader, device))
        if not math.isfinite(validation):
            raise FloatingPointError("non-finite validation metric")
        improved = validation < best - min_delta if mode == "min" else validation > best + min_delta
        if improved:
            best = validation
            best_epoch = epoch
            save_checkpoint(directory / "best.pt", {
                "state_dict": model.state_dict(), "metadata": metadata,
                "epoch": epoch, "validation_metric": validation,
                "selection_mode": mode,
            })
        if scheduler is not None:
            scheduler.step(validation)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {"epoch": epoch, "train_loss": total / count, "validation_metric": validation,
                  "learning_rate": learning_rate, "next_learning_rate": next_learning_rate,
                  "lr_reduced": next_learning_rate < learning_rate,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        print(f"{directory.name} epoch {epoch}/{settings['epochs']}: loss={record['train_loss']:.5f}, val={validation:.5f}, lr={learning_rate:.3g}, next_lr={next_learning_rate:.3g}", flush=True)
        if patience and epoch - best_epoch >= patience:
            stopped = True
            print(f"{directory.name} early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break
    write_json(directory / "history.json", history)
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(directory / "training_manifest.json", {
        **metadata, "training": settings, "best_validation_metric": best,
        "best_epoch": best_epoch, "epochs_run": len(history), "early_stopped": stopped,
        "initial_validation_metric": initial_validation, "scheduler": resolved_scheduler,
        "lr_reductions": sum(record["lr_reduced"] for record in history),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "seconds": time.perf_counter() - started,
    })
    return [directory / name for name in ("best.pt", "history.json", "history.csv", "training_manifest.json")]
