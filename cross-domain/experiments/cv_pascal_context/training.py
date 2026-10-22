"""A-only author-model training: Adam/poly schedule and A_val mIoU selection."""
import math
import time

import torch
from torch.nn import functional as F
from pipeline.fit import save_checkpoint
from pipeline.io import read_json, write_json
from pipeline.runtime import configure, seed_all, to_device
from experiments.cv_pascal_context.data import make_loader, data_identity
from experiments.cv_pascal_context.model import build_model, TASK_OUTPUTS
from experiments.cv_pascal_context.source import author_module, verify_source


def upstream_dir(context, variant, seed):
    return context.output_dir / "upstream" / variant / f"seed{seed}"


def semantic_loss(logits, target):
    valid = target != 255
    if not valid.any():
        return logits.sum()*0
    if ((target[valid] < 0) | (target[valid] >= 21)).any():
        raise ValueError("semantic label outside 21-class schema")
    return F.cross_entropy(logits, target, ignore_index=255)


def task_loss(logits, target, task):
    if task in ("semseg", "human_parts"):
        valid = target != 255
        if not valid.any():
            return logits.sum()*0
        return F.cross_entropy(logits, target, ignore_index=255)
    losses = author_module("losses.loss_functions")
    if task == "normals":
        # Exact author's normalized elementwise L1; voids exist only in A truth.
        normal = F.normalize(logits, dim=1, eps=1e-12)
        mask = target != 255
        return (normal[mask]-target[mask]).abs().mean() if mask.any() else logits.sum()*0
    return losses.BalancedCrossEntropyLoss(pos_weight=.95 if task == "edge" else None)(logits, target)


def upstream_objective(output, batch, variant, weights):
    tasks = list(TASK_OUTPUTS) if variant == "multi_task" else ["semseg"]
    loss = sum(weights[t]*task_loss(output[t], batch[t], t) for t in tasks)
    if variant != "single_task":
        for scale in range(4):
            deep = output["deep_supervision"][f"scale_{scale}"]
            for task in tasks:
                resized = F.interpolate(deep[task], size=batch["semseg"].shape[-2:], mode="bilinear", align_corners=False)
                loss = loss + weights[task]*task_loss(resized, batch[task], task)
    return loss


def train(context):
    from experiments.cv_pascal_context.evaluate import metrics
    settings = context.config["upstream"]
    device = configure(context.config["runtime"])
    artifacts = []
    for variant in settings["variants"]:
        if context.filters.get("variant") not in (None, variant):
            continue
        for seed in settings["seeds"]:
            if context.filters.get("seed") not in (None, seed):
                continue
            seed_all(seed)
            model, initialization = build_model(context, variant, load_pretrained=True)
            metadata = {"identity": context.identity, "data_identity": data_identity(context), "variant": variant, "seed": seed, **initialization,
                        "training_split": "a_train", "selection_split": "a_val", "selection_metric": "dataset_miou_21", "optimizer": "Adam", "schedule": "poly exponent 0.9", "loss_weights": settings["loss_weights"],
                        "architecture": "author SingleTaskModel HRNet-W18" if variant == "single_task" else "unscaled author MTINet HRNet-W18 (all five branches)",
                        "supervision": "all five tasks with four-scale deep supervision" if variant == "multi_task" else "semseg only; full MTI graph" if variant == "main_only_matched" else "semseg only", "augmentation": context.config["data"]["augmentation"]}
            directory = upstream_dir(context, variant, seed)
            directory.mkdir(parents=True, exist_ok=True)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
            training = make_loader(context, "a_train", seed, settings["batch_size"], True, variant == "multi_task")
            validation = make_loader(context, "a_val", seed, settings["batch_size"])
            history, best, best_epoch, started = [], -math.inf, 0, time.perf_counter()
            for epoch in range(settings["epochs"]+1):
                total, count = 0., 0
                if epoch:
                    model.train()
                    lr = settings["learning_rate"]*(1-(epoch-1)/settings["epochs"])**.9
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    for batch in training:
                        batch = to_device(batch, device)
                        optimizer.zero_grad(set_to_none=True)
                        objective = upstream_objective(model(batch["image"]), batch, variant, settings["loss_weights"])
                        if not torch.isfinite(objective):
                            raise FloatingPointError("non-finite A objective")
                        objective.backward()
                        if settings.get("clip_grad"):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad"])
                        optimizer.step()
                        total += float(objective.detach())*len(batch["image"])
                        count += len(batch["image"])
                    if count != len(training.dataset):
                        raise ValueError("A training coverage incomplete")
                value = metrics(model, validation, device)["miou"]
                if not math.isfinite(value):
                    raise FloatingPointError("nonfinite A_val metric")
                if value > best:
                    best, best_epoch = value, epoch
                    save_checkpoint(directory / "best.pt", {"state_dict": model.state_dict(), "metadata": metadata, "epoch": epoch, "validation_metric": best})
                history.append({"epoch": epoch, "train_loss": total/count if count else None, "validation_miou": value, "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter()-started})
                print(f"{variant}/seed{seed} epoch={epoch} A_val_mIoU={value:.6f}", flush=True)
                patience = settings.get("early_stopping_patience", 0)
                if epoch and patience and epoch-best_epoch >= patience:
                    break
            artifacts.extend([directory / "best.pt", write_json(directory / "history.json", history), write_json(directory / "training_manifest.json", {**metadata, "best_epoch": best_epoch, "best_validation_miou": best, "epochs_run": len(history)-1, "seconds": time.perf_counter()-started, "training": settings, "parameters": sum(p.numel() for p in model.parameters())})])
            del model, optimizer
    return artifacts


def load_upstream(context, variant, seed, device, frozen=True):
    verify_source()
    checkpoint = torch.load(upstream_dir(context, variant, seed) / "best.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    expected = {"identity": context.identity, "data_identity": data_identity(context), "variant": variant, "seed": seed, "source": verify_source()}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("upstream checkpoint provenance mismatch")
    if metadata["pretrained"]["sha256"] != context.config["model"]["pretrained"]["sha256"]:
        raise ValueError("upstream pretrained source changed")
    model, _ = build_model(context, variant, load_pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    if frozen:
        model.requires_grad_(False)
    return model, metadata
