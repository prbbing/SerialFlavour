"""Validation-driven learning-rate schedules shared by all jet trainers."""

from __future__ import annotations

import math

import torch


def resolve_lr_scheduler(options, *, initial_lr, default_patience):
    """Validate options; absent configuration preserves fixed-LR training."""
    if options is None:
        return {"enabled": False}
    if not isinstance(options, dict):
        raise ValueError("lr_scheduler must be an object")
    allowed = {"enabled", "type", "factor", "patience", "min_lr",
               "threshold", "threshold_mode", "cooldown"}
    if set(options) - allowed:
        raise ValueError(f"unknown lr_scheduler fields: {sorted(set(options) - allowed)}")
    config = {
        "enabled": True, "type": "reduce_on_plateau", "factor": 0.5,
        "patience": default_patience, "min_lr": 1e-5,
        "threshold": 1e-4, "threshold_mode": "rel", "cooldown": 0,
        **options,
    }
    if not isinstance(config["enabled"], bool):
        raise ValueError("lr_scheduler.enabled must be a boolean")
    if config["type"] != "reduce_on_plateau":
        raise ValueError("lr_scheduler.type must be 'reduce_on_plateau'")
    for key in ("factor", "min_lr", "threshold"):
        value = config[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"lr_scheduler.{key} must be finite and numeric")
    if not 0 < config["factor"] < 1:
        raise ValueError("lr_scheduler.factor must be in (0, 1)")
    if config["min_lr"] < 0 or (config["enabled"] and config["min_lr"] > initial_lr):
        raise ValueError("lr_scheduler.min_lr must be non-negative and at most initial LR")
    if config["threshold"] < 0:
        raise ValueError("lr_scheduler.threshold must be non-negative")
    if config["threshold_mode"] not in ("rel", "abs"):
        raise ValueError("lr_scheduler.threshold_mode must be 'rel' or 'abs'")
    for key in ("patience", "cooldown"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"lr_scheduler.{key} must be a non-negative integer")
    return config


def create_lr_scheduler(optimiser, options, *, default_patience):
    config = resolve_lr_scheduler(
        options, initial_lr=min(group["lr"] for group in optimiser.param_groups),
        default_patience=default_patience)
    if not config["enabled"]:
        return None, config
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", **{
            key: value for key, value in config.items()
            if key not in {"enabled", "type"}})
    return scheduler, config


def step_lr_scheduler(optimiser, scheduler, validation_loss):
    """Step after validation; distinguish this epoch's LR from the next one."""
    before = [group["lr"] for group in optimiser.param_groups]
    if scheduler is not None:
        scheduler.step(validation_loss)
    after = [group["lr"] for group in optimiser.param_groups]
    return {
        "lr": before[0], "lr_next": after[0],
        "lr_reduced": any(new < old for old, new in zip(before, after)),
    }
