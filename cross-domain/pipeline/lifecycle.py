"""One frozen-upstream lifecycle at a time, with durable receipts and safe eviction.

Experiment packages own streaming evaluation and cache layout. This module owns
serial scheduling, artifact checks, GPU allocation and receipt-aware resume.
No tensor framework, model, device or output directory is touched by dry-run.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from pipeline.context import load_context
from pipeline.io import artifact_record, artifacts_match, read_json, sha256_file, write_json
from pipeline.worker import marker_path


def gpu_mapping(indices):
    if not indices or len(set(indices)) != len(indices) or any(not str(i).isdigit() for i in indices):
        raise ValueError("distinct nonnegative logical GPU indices required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(i) for i in indices]
    allocated = [i.strip() for i in visible.split(",") if i.strip()]
    if not allocated or "-1" in allocated or any(int(i) >= len(allocated) for i in indices):
        raise ValueError("GPU indices must address inherited CUDA_VISIBLE_DEVICES allocation")
    return [allocated[int(i)] for i in indices]


def seed_order(context):
    return [(v, s) for v in context.config["upstream"]["variants"] for s in context.config["upstream"]["seeds"]]


def seed_units(context, variant, seed):
    recipes = context.module("data").applicable_recipes(context, variant)
    return [f"upstream:{variant}:{seed}", f"cache:{variant}:{seed}", *[
        f"refine:{variant}:{seed}:{recipe}:{ds}" for recipe in recipes
        for ds in context.config["refiner"]["seeds"]]]


def unit_complete(context, unit):
    try:
        record = read_json(marker_path(context, unit))
        return (record.get("status") == "complete" and record.get("identity") == context.identity
                and artifacts_match(record.get("artifacts", [])))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def verify_seed_ready(context, variant, seed):
    if (variant, seed) not in seed_order(context):
        raise ValueError("upstream is outside the locked lifecycle plan")
    missing = [u for u in ["prepare", *seed_units(context, variant, seed)] if not unit_complete(context, u)]
    if missing:
        raise ValueError(f"streaming Y evaluation blocked by incomplete/mismatched units: {missing}")


def receipt_path(context, variant, seed):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variant) or type(seed) is not int or seed < 0:
        raise ValueError("unsafe lifecycle identity")
    return context.output_dir / "lifecycle" / variant / f"seed{seed}" / "receipt.json"


def _unique_paths(paths):
    return sorted({str(Path(p).resolve()) for p in paths})


def _safe_cleanup(context, variant, seed, roots, records):
    anchor = context.output_dir.resolve()
    bases = [anchor / kind / variant / f"seed{seed}" for kind in ("cache", "scratch")]
    checked = []
    for value in roots:
        path = Path(value)
        resolved = path.resolve()
        if (not path.is_absolute() or path.is_symlink() or not resolved.is_relative_to(anchor)
                or not any(resolved.is_relative_to(base) for base in bases)):
            raise ValueError(f"cache cleanup root outside this upstream namespace: {value}")
        checked.append(resolved)
    seen = set()
    for row in records:
        path = Path(row["path"])
        resolved = path.resolve()
        if (not path.is_absolute() or path.is_symlink() or str(resolved) in seen
                or not any(resolved.is_relative_to(r) for r in checked)
                or type(row.get("bytes")) is not int or row["bytes"] < 0
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", ""))):
            raise ValueError(f"unsafe/duplicate cache cleanup record: {row.get('path')}")
        seen.add(str(resolved))
    return checked


def load_receipt(context, variant, seed):
    record = read_json(receipt_path(context, variant, seed))
    required = {"schema": 1, "status": "complete", "identity": context.identity,
                "variant": variant, "upstream_seed": seed}
    if any(record.get(k) != v for k, v in required.items()):
        raise ValueError("stale or incomplete lifecycle receipt; preserve evidence and use a new experiment")
    expected = _unique_paths(context.module("workflow").seed_artifacts(context, variant, seed))
    actual = [row["path"] for row in record["artifacts"]]
    if len(actual) != len(set(actual)) or sorted(actual) != expected or not artifacts_match(record["artifacts"]):
        raise ValueError("durable lifecycle artifact missing or checksum changed")
    _safe_cleanup(context, variant, seed, record["cleanup_roots"], record["cleanup_records"])
    if set(actual) & {r["path"] for r in record["cleanup_records"]}:
        raise ValueError("durable evidence cannot be a disposable cache")
    return record


def valid_receipt(context, variant, seed):
    try:
        load_receipt(context, variant, seed)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def verify_receipts(context):
    return [load_receipt(context, variant, seed) for variant, seed in seed_order(context)]


def evaluate_one(context, variant, seed):
    """Internal subprocess operation; seal only a complete predetermined readout grid."""
    backend = context.module("workflow")
    backend.validate(context)
    target = receipt_path(context, variant, seed)
    if target.exists():
        load_receipt(context, variant, seed)
        return target
    verify_seed_ready(context, variant, seed)
    produced = _unique_paths(backend.evaluate_seed(context, variant, seed))
    permanent = _unique_paths(backend.seed_artifacts(context, variant, seed))
    if not permanent or not set(produced) <= set(permanent):
        raise ValueError("stream evaluation outputs must all appear in durable seed_artifacts")
    cleanup = backend.cleanup_records(context, variant, seed)
    roots = _unique_paths(backend.cleanup_roots(context, variant, seed))
    _safe_cleanup(context, variant, seed, roots, cleanup)
    if set(permanent) & {r["path"] for r in cleanup}:
        raise ValueError("refusing to retire durable seed evidence")
    write_json(target, {"schema": 1, "status": "complete", "identity": context.identity,
        "variant": variant, "upstream_seed": seed, "artifacts": [artifact_record(p) for p in permanent],
        "cleanup_roots": roots, "cleanup_records": cleanup,
        "selection": "fixed A_val/B_val choices; intermediate Y results never select later seeds/configs"})
    load_receipt(context, variant, seed)
    return target


def retire_cache(context, variant, seed):
    """Delete only recorded reproducible files, after verifying permanent evidence.

    Never recursively remove directories. Unknown files and every checkpoint,
    normalization, input dataset and lineage manifest remain untouched.
    """
    record = load_receipt(context, variant, seed)
    roots = _safe_cleanup(context, variant, seed, record["cleanup_roots"], record["cleanup_records"])
    removed, absent = [], []
    for row in record["cleanup_records"]:
        path = Path(row["path"])
        if not path.exists():
            absent.append(str(path))
            continue
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"refusing to remove modified or unexpected cache: {path}")
        path.unlink()
        removed.append(str(path))
        parent = path.parent
        while any(parent.is_relative_to(r) for r in roots):
            try:
                parent.rmdir()
            except OSError:
                break
            if parent in roots:
                break
            parent = parent.parent
    ledger = receipt_path(context, variant, seed).with_name("cache_retirement.json")
    return write_json(ledger, {"identity": context.identity, "receipt_sha256": sha256_file(receipt_path(context, variant, seed)),
        "intentional_eviction": True, "removed": removed, "already_absent": absent,
        "retained": "checkpoints, normalization, metrics, receipts and lineage manifests"})


def _run_identity(context):
    path = context.output_dir / "run_manifest.json"
    if path.exists() and read_json(path).get("identity") != context.identity:
        raise ValueError("experiment identity changed; choose a new experiment name")


@contextmanager
def exclusive_experiment(context):
    directory = context.resolve(context.config.get("log_root", "cross-domain/logs")) / context.config["dataset"] / context.config["experiment"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lifecycle.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _unit_subprocess(context, unit, gpu, python, retries):
    marker = marker_path(context, unit)
    if unit_complete(context, unit):
        return
    if marker.exists():
        previous = read_json(marker)
        if previous.get("identity") != context.identity or previous.get("status") == "complete":
            raise ValueError(f"stale or corrupt completed unit {unit}; no automatic overwrite")
    marker.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1")
    command = [python, str(context.domain_root / "scripts/run_unit.py"), "--config", str(context.config_path), "--unit", unit]
    for attempt in range(retries + 1):
        with marker.with_suffix(f".attempt{attempt}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=context.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and unit_complete(context, unit):
            return
    raise RuntimeError(f"unit failed: {unit}; cache retained for investigation/retry")


def _evaluation_subprocess(context, variant, seed, gpu, python):
    target = receipt_path(context, variant, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1")
    command = [python, str(context.domain_root / "scripts/run_lifecycle.py"), "--config", str(context.config_path),
               "--evaluate-seed", variant, str(seed)]
    with target.with_name("evaluate.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=context.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not valid_receipt(context, variant, seed):
        raise RuntimeError(f"streaming evaluation failed for {variant}/{seed}; cache retained")


def check_output_space(context):
    """Fail before preparation when estimated output demand cannot fit.

    Estimates exclude original inputs/weights; each backend states its allowance.
    Existing regenerable bytes count toward the peak so interrupted seeds can
    resume without demanding a second copy of their cache.
    """
    workflow = context.config.get("workflow", {})
    cache = context.config.get("cache", {})
    if not workflow.get("check_free_space", cache.get("check_free_space", True)):
        return
    if all(valid_receipt(context, v, s) for v, s in seed_order(context)):
        return
    plan = context.module("workflow").plan(context)
    keys = ("peak_output_bytes", "planned_output_bytes", "peak_output_tensor_bytes_estimate",
            "peak_regenerable_cache_bytes_estimate")
    estimate = next((int(plan[k]) for k in keys if k in plan), None)
    if estimate is None or estimate < 0:
        raise ValueError("lifecycle backend must provide a nonnegative peak output estimate")
    reserve = int(workflow.get("free_space_reserve_bytes", cache.get("free_space_reserve_bytes", 0)))
    if reserve < 0:
        raise ValueError("free-space reserve must be nonnegative")
    present = 0
    for kind in ("cache", "scratch"):
        root = context.output_dir / kind
        if root.is_symlink():
            raise ValueError("resource cache/scratch root must not be a symbolic link")
        if root.exists():
            present += sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    directory = context.output_dir.resolve()
    while not directory.exists():
        directory = directory.parent
    required = max(0, estimate - present) + reserve
    free = shutil.disk_usage(directory).free
    if free < required:
        raise OSError(f"insufficient output disk space: {free:,} free bytes, {required:,} additional estimated bytes required "
                      "including reserve; inputs/weights require separate space")


def run(context, gpus, retries=1, python=None, unit_runner=None, evaluation_runner=None):
    if not gpus or retries < 0:
        raise ValueError("at least one assigned GPU and nonnegative retries required")
    backend = context.module("workflow")
    backend.validate(context)
    if context.config.get("workflow", {}).get("mode") != "seed_lifecycle":
        raise ValueError("resource runner requires workflow.mode=seed_lifecycle")
    if context.config["workflow"].get("max_active_seeds", 1) != 1:
        raise ValueError("this runner processes exactly one upstream seed at a time")
    _run_identity(context)
    unit_runner = unit_runner or _unit_subprocess
    evaluation_runner = evaluation_runner or _evaluation_subprocess
    python = python or sys.executable
    with exclusive_experiment(context):
        _run_identity(context)
        check_output_space(context)
        unit_runner(context, "prepare", gpus[0], python, retries)
        verify_inputs = getattr(backend, "verify_inputs", None)
        if verify_inputs is not None:
            verify_inputs(context)
        for index, (variant, seed) in enumerate(seed_order(context)):
            if receipt_path(context, variant, seed).exists():
                load_receipt(context, variant, seed)
            else:
                for unit in seed_units(context, variant, seed):
                    unit_runner(context, unit, gpus[index % len(gpus)], python, retries)
                evaluation_runner(context, variant, seed, gpus[index % len(gpus)], python)
                load_receipt(context, variant, seed)
            if context.config["workflow"].get("release_cache", True):
                retire_cache(context, variant, seed)
            print(f"sealed {variant}/seed{seed}; durable evidence verified", flush=True)
        verify_receipts(context)
        artifacts = list(backend.aggregate(context))
        write_json(context.output_dir / "lifecycle" / "complete.json", {"identity": context.identity,
            "status": "complete", "seed_receipts": [artifact_record(receipt_path(context, v, s)) for v, s in seed_order(context)],
            "artifacts": [artifact_record(p) for p in artifacts], "max_active_seeds": 1})
        return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-seed", nargs=2, metavar=("VARIANT", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    context = load_context(args.config, root)
    backend = context.module("workflow")
    backend.validate(context)
    if args.retries < 0:
        parser.error("retries must be nonnegative")
    if args.dry_run:
        print(json.dumps({**backend.plan(context), "gpu_mapping": gpu_mapping(args.gpus),
            "max_active_seeds": 1,
            "free_space_reserve_bytes": context.config.get("workflow", {}).get("free_space_reserve_bytes", context.config.get("cache", {}).get("free_space_reserve_bytes", 0)),
            "execution": "read-only; no tensor import, CUDA, outputs or subprocess"}, indent=2))
        return
    if args.evaluate_seed:
        variant, seed = args.evaluate_seed
        evaluate_one(context, variant, int(seed))
        return
    run(context, gpu_mapping(args.gpus), args.retries, args.python)


if __name__ == "__main__":
    main()
