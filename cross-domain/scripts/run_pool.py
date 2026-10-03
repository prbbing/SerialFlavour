"""Single-node multi-GPU work queue over (variant, seed) units, then evaluate.

Phase 1: run the ``prepare`` unit once.
Phase 2: dispatch one ``run_seed`` process per free GPU until all seed units finish.
Phase 3: run ``evaluate`` then ``analyze``.

Completed seeds (marker ``status == complete``) are skipped, so the pool
resumes. A seed is retried by ``run_seed``; a still-failing seed is skipped and
the pool reports a non-zero exit after finishing the rest.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT))

from pipeline.context import load_context


def log_dir(context, log_root):
    return log_root / context.config["dataset"] / context.config["experiment"]


def run_unit_subprocess(python, config, unit, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen([python, str(Path(__file__).with_name("run_unit.py")),
                                    "--config", config, "--unit", unit],
                                   stdout=handle, stderr=subprocess.STDOUT, cwd=str(DOMAIN_ROOT.parent), env=env)
        return process.wait()


def seed_complete(context, log_root, variant, seed):
    marker = log_dir(context, log_root) / "seeds" / f"{variant}__seed{seed}.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text())["status"] == "complete"
    except Exception:  # noqa: BLE001 - corrupt marker is treated as incomplete
        return False


def marker_complete(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text())["status"] == "complete"
    except Exception:  # noqa: BLE001 - corrupt marker is treated as incomplete
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    context = load_context(args.config, DOMAIN_ROOT)
    log_root = context.resolve(context.config.get("log_root", "cross-domain/logs"))
    units_dir = log_dir(context, log_root)
    config = str(Path(args.config).resolve())
    scheduler_log = units_dir / "scheduler.log"

    def note(message):
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        scheduler_log.parent.mkdir(parents=True, exist_ok=True)
        with scheduler_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if not marker_complete(units_dir / "units" / "prepare.json"):
        note("prepare: start")
        rc = run_unit_subprocess(args.python, config, "prepare", units_dir / "prepare.log")
        if rc != 0:
            raise SystemExit("prepare failed")
        note("prepare: done")

    queue = [(variant, seed) for variant in context.config["upstream"]["variants"]
             for seed in context.config["upstream"]["seeds"] if not seed_complete(context, log_root, variant, seed)]
    note(f"queue: {len(queue)} seeds, gpus={args.gpus}")
    running, failed = {}, []
    while queue or running:
        for gpu in args.gpus:
            if gpu in running or not queue:
                continue
            variant, seed = queue.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = units_dir / "seed" / f"{variant}__seed{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen([args.python, str(Path(__file__).with_name("run_seed.py")),
                                        "--config", config, "--variant", variant, "--seed", str(seed),
                                        "--gpu", str(gpu), "--retries", str(args.retries)],
                                       stdout=handle, stderr=subprocess.STDOUT, cwd=str(DOMAIN_ROOT.parent), env=env)
            running[gpu] = (variant, seed, process, handle)
            note(f"seed {variant}:{seed} -> gpu {gpu}")
        time.sleep(5)
        for gpu, (variant, seed, process, handle) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            del running[gpu]
            note(f"seed {variant}:{seed} gpu {gpu} rc={rc}")
            if rc != 0:
                failed.append(f"{variant}:{seed}")
    for unit in ("evaluate", "analyze"):
        note(f"{unit}: start")
        rc = run_unit_subprocess(args.python, config, unit, units_dir / f"{unit}.log")
        note(f"{unit}: done rc={rc}")
        if rc != 0:
            failed.append(unit)
    if failed:
        raise SystemExit(f"pool finished with failures: {failed}")
    note("pool complete")


if __name__ == "__main__":
    main()
