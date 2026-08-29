#!/usr/bin/env python3
"""Train configured Parallel seeds on A and validate on A-val."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.config import seed_everything

from src.parallel_refine.config import (
    active_parallel_config, load_study_config, materialize_parallel_config,
    write_json_atomic)
from src.parallel_refine.data import create_loader
from src.parallel_refine.models import build_parallel
from src.parallel_refine.plotting import plot_training_history
from src.parallel_refine.splits import load_split_bundle
from src.parallel_refine.training import (
    aggregate_losses, choose_device, evaluate_loss, move_batch,
    origin_class_weights, parallel_losses, save_history)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--skip-complete", action="store_true",
        help="Skip a seed when its configured checkpoint already exists.")
    args = parser.parse_args(argv)
    study = load_study_config(args.config)
    selected = study.selected_seeds(args.seed)

    for run in selected:
        checkpoint = study.checkpoint(run)
        if checkpoint.exists():
            if args.skip_complete:
                print(f"skip seed={run.seed}: {checkpoint}")
                continue
            raise FileExistsError(
                f"configured checkpoint already exists: {checkpoint}; "
                "use --skip-complete or choose another output_name")
        output = study.parallel_directory(run)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"refusing to mix with partial output directory: {output}")
        resolved = materialize_parallel_config(study, run, stage="parallel")
        print(
            f"train Parallel seed={run.seed} output_name={run.output_name} "
            f"config={resolved}")
        config = active_parallel_config(study, run, stage="parallel")
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "config.json", config._raw)
        bundle = load_split_bundle(config.split_dir, config=config)
        write_json_atomic(output / "split_manifest.json", bundle.summary)

        device = choose_device(config)
        seed_everything(
            config.seed, [device.index] if device.type == "cuda" else ())
        train_loader, train_arrays = create_loader(
            config, "a_train", progress=True)
        val_loader, _ = create_loader(
            config, "a_val", shuffle=False, progress=True)
        model = build_parallel(config).to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay)
        origin_criterion = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=origin_class_weights(
                train_arrays, config.n_origin_classes, device))
        loss_fn = lambda model_output, batch: parallel_losses(
            model_output, batch, config, origin_criterion)
        history = []
        best_jet = float("inf")
        best_total = float("inf")
        best_jet_epoch = None
        best_total_epoch = None
        history_metadata = {
            "stage": "parallel",
            "experiment_name": study.experiment_name,
            "parallel_output_name": run.output_name,
            "seed": run.seed,
            "train_split": "a_train",
            "validation_split": "a_val",
            "checkpoint_policy": {
                "best_jet.pt": "minimum validation jet cross-entropy",
                "best_total.pt": "minimum validation total loss",
            },
        }
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            totals = {}
            count = 0
            jet_correct = 0
            for raw in train_loader:
                batch = move_batch(raw, device)
                optimiser.zero_grad(set_to_none=True)
                model_output = model(batch["X"], batch["mask"])
                losses = loss_fn(model_output, batch)
                losses["total"].backward()
                optimiser.step()
                aggregate_losses(totals, losses, len(batch["y"]))
                jet_correct += int((
                    model_output["jet_logits"].argmax(-1) == batch["y"]).sum())
                count += len(batch["y"])
            train = {
                name: value / max(count, 1) for name, value in totals.items()
            }
            train["jet_accuracy"] = jet_correct / max(count, 1)
            validation = evaluate_loss(model, val_loader, device, loss_fn)
            saved_best_jet = validation["jet"] < best_jet
            saved_best_total = validation["total"] < best_total
            if saved_best_jet:
                best_jet = validation["jet"]
                best_jet_epoch = epoch
                torch.save(model.state_dict(), output / "best_jet.pt")
            if saved_best_total:
                best_total = validation["total"]
                best_total_epoch = epoch
                torch.save(model.state_dict(), output / "best_total.pt")
            if epoch % config.checkpoint_interval == 0:
                torch.save(model.state_dict(), output / f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), output / "last.pt")
            history.append({
                "lr": optimiser.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_samples": count,
                **{f"train_{key}": value for key, value in train.items()},
                **{f"val_{key}": value for key, value in validation.items()},
                "best_jet_loss_so_far": best_jet,
                "best_jet_epoch_so_far": best_jet_epoch,
                "best_total_loss_so_far": best_total,
                "best_total_epoch_so_far": best_total_epoch,
                "saved_best_jet": saved_best_jet,
                "saved_best_total": saved_best_total,
            })
            save_history(history, output, metadata=history_metadata)
            print(
                f"seed={run.seed} epoch={epoch} "
                f"train={train['total']:.6f} "
                f"val_jet={validation['jet']:.6f}")
        plot_training_history(history, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
