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
        save_checkpoint(directory / "best.pt", {
            "state_dict": model.state_dict(), "metadata": metadata, "epoch": 0,
            "validation_metric": best, "selection_mode": mode,
        })
    for epoch in range(1, settings["epochs"] + 1):
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
        record = {"epoch": epoch, "train_loss": total / count, "validation_metric": validation,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        print(f"{directory.name} epoch {epoch}/{settings['epochs']}: loss={record['train_loss']:.5f}, val={validation:.5f}", flush=True)
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
        "initial_validation_metric": initial_validation,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "seconds": time.perf_counter() - started,
    })
    return [directory / name for name in ("best.pt", "history.json", "history.csv", "training_manifest.json")]
