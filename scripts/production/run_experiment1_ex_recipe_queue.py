#!/usr/bin/env python3
"""Dependency-aware, one-job-per-GPU scheduler for Experiment 1 EX.

Each seed advances through Parallel, B-cache, one recipe at a time, Y-cache,
evaluation, and pruning.  A GPU is released after every unit, so an idle GPU
immediately receives a dependency-ready unit.

Once a seed finishes Parallel, its refine chain is prioritised: units after
Parallel are drawn from a separate queue ahead of not-yet-started Parallel
units, so a seed does not wait for the whole Parallel wave.  ``--max-refine``
bounds how many GPUs may run refine chains at once (leaving the rest on
Parallel training); when the Parallel queue drains, refine may use every GPU.

Each unit is launched with ``--retries`` so the seed runner retries a failing
stage once.  If a unit still fails, that Parallel seed is skipped (its serial
continuation is dropped) while every other seed keeps running; the scheduler
exits non-zero after draining so the caller can report the skipped seeds.

At startup the scheduler adopts any external stage-parallel unit it finds
(a seed runner already training a Parallel model outside this process): its
GPU is reserved out of the pool and it is polled until its ``last.pt`` lands
or its process disappears.  This lets a restart resume beside in-flight work
without double-dispatching a seed or oversubscribing a GPU.

As soon as every seed of a configuration has been pruned, that configuration's
cross-seed rejection aggregate is launched (CPU only) and its
``<log-base>/aggregate/<label>.done`` marker written, so each config is
aggregated without waiting for the others.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.config import (
    FEATURE_RECIPES, GRAPH_RECIPES, load_study_config)


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def terminate_group(process, grace: int = 10) -> None:
    """Terminate a worker and every child it spawned (its own session group)."""
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def read_cmdline(pid, proc_root=Path("/proc")):
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def parse_stage_parallel_cmdline(cmdline):
    """Return ``(config, seed, gpu)`` for an external stage-parallel runner."""
    if not cmdline or not any(
            "run_experiment1_ex_seed.py" in part for part in cmdline):
        return None

    def value(flag):
        if flag in cmdline:
            index = cmdline.index(flag)
            if index + 1 < len(cmdline):
                return cmdline[index + 1]
        return None

    if value("--stage") != "parallel":
        return None
    config, seed, gpu = value("--config"), value("--seed"), value("--gpus")
    if config is None or seed is None or gpu is None:
        return None
    return config, seed, gpu


def pid_state(pid, proc_root=Path("/proc")):
    """Return the process state letter, ``"Z"`` for a zombie, or ``None``."""
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = raw.rfind(")")
    if end < 0:
        return None
    fields = raw[end + 1:].split()
    return fields[0] if fields else None


def discover_external_parallels(config_names, pool, proc_root=Path("/proc")):
    """Map ``(study_name, seed) -> (gpu, pid)`` for live external units.

    ``config_names`` maps a resolved config path to its study name; only gpus
    inside ``pool`` are considered.
    """
    found = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        parsed = parse_stage_parallel_cmdline(
            read_cmdline(entry.name, proc_root))
        if parsed is None:
            continue
        config, seed, gpu = parsed
        name = config_names.get(str(Path(config).resolve()))
        if name is None or str(gpu) not in pool:
            continue
        found[(name, int(seed))] = (str(gpu), int(entry.name))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--log-base", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--max-refine", type=int, default=None,
        help=("Cap how many GPUs may run a seed's post-Parallel refine chain "
              "(b/recipe/y/eval/prune) at once, so at least the remaining GPUs "
              "keep training Parallel models.  Defaults to len(gpus)."))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    max_refine = args.max_refine if args.max_refine is not None else len(args.gpus)
    max_refine = max(1, min(max_refine, len(args.gpus)))

    log_base = Path(args.log_base)
    scheduler_log = log_base / "scheduler.log"
    failed_log = log_base / "failed_jobs.log"
    units_dir = log_base / "units"
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    states: dict[tuple[str, int], dict] = {}
    completed: set[tuple[str, int]] = set()
    config_names: dict[str, str] = {}
    studies: dict[str, object] = {}
    study_total: dict[str, int] = {}
    study_done: dict[str, int] = {}
    aggregate_dir = log_base / "aggregate"
    aggregate_failed: list[str] = []
    ready_parallel: deque[tuple[str, int, str, str | None]] = deque()
    ready_refine: deque[tuple[str, int, str, str | None]] = deque()
    for config_path in args.config:
        study = load_study_config(config_path)
        config_names[str(Path(study.path).resolve())] = study.study_name
        studies[study.study_name] = study
        study_total[study.study_name] = len(study.seeds)
        study_done.setdefault(study.study_name, 0)
        configured = list(study.refiners["recipes"])
        recipes = [r for r in configured
                   if r in FEATURE_RECIPES or r in GRAPH_RECIPES]
        dropped = [r for r in configured if r not in recipes]
        if dropped:
            log(scheduler_log,
                f"ignore recipe(s) not trainable for {study.study_name}: {dropped}")
        b_train = int(study.data["sizes"]["b_train"])
        label = study.study_name.removeprefix("experiment1_ex_p122k_")
        for run in study.seeds:
            key = (study.study_name, run.seed)
            states[key] = {"study": study, "run": run, "recipes": recipes,
                           "b_train": b_train, "next": 0, "label": label}
            marker = state_dir / f"{label}.seed{run.seed}.done"
            if marker.is_file() and not args.force:
                log(scheduler_log, f"skip completed {label}_seed{run.seed}")
                completed.add(key)
                study_done[study.study_name] += 1
            else:
                ready_parallel.append((study.study_name, run.seed, "parallel", None))

    pool = [str(gpu) for gpu in args.gpus]
    external = {
        key: value for key, value in
        discover_external_parallels(config_names, pool).items()
        if key in states and key not in completed}
    if external:
        adopted = set(external)
        ready_parallel = deque(
            item for item in ready_parallel if (item[0], item[1]) not in adopted)
        for (name, seed), (gpu, pid) in sorted(external.items()):
            log(scheduler_log,
                f"adopt external {states[(name, seed)]['label']}_seed{seed} "
                f"gpu={gpu} pid={pid}")
    reserved = {gpu for gpu, _ in external.values()}

    running: dict[int, tuple[str, int, str, str | None, str, object]] = {}
    free_gpus = deque(gpu for gpu in pool if gpu not in reserved)
    skipped: dict[tuple[str, int], tuple[str, str | None]] = {}
    aggregate_running: dict[int, tuple[str, object, object]] = {}

    def start_aggregate(name: str) -> None:
        """Launch the cross-seed rejection aggregate for one finished config."""
        label = studies[name].study_name.removeprefix("experiment1_ex_p122k_")
        marker = aggregate_dir / f"{label}.done"
        if marker.is_file():
            return
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        handle = (aggregate_dir / f"{label}.log").open("w", encoding="utf-8")
        command = [
            args.python, "scripts/evaluate.py", "--config",
            str(studies[name].path), "--model", "parallel_dnn",
            "--aggregate-parallel-seeds"]
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT)
        aggregate_running[process.pid] = (name, process, handle)
        log(scheduler_log, f"aggregate start {label} pid={process.pid}")

    def poll_aggregate() -> None:
        for pid, (name, process, handle) in list(aggregate_running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            del aggregate_running[pid]
            label = studies[name].study_name.removeprefix("experiment1_ex_p122k_")
            if rc:
                aggregate_failed.append(label)
                log(failed_log, f"FAILED aggregate {label} rc={rc}")
            else:
                (aggregate_dir / f"{label}.done").touch()
                log(scheduler_log, f"aggregate done {label}")

    for name in studies:
        if study_done.get(name, 0) == study_total[name]:
            start_aggregate(name)

    def enqueue_next(name: str, seed: int, stage: str, recipe: str | None):
        state = states[(name, seed)]
        if stage == "parallel":
            if state["b_train"] > 0:
                ready_refine.append((name, seed, "b", None))
            else:
                ready_refine.append((name, seed, "y", None))
        elif stage == "b":
            if state["recipes"]:
                ready_refine.append((name, seed, "recipe", state["recipes"][0]))
            else:
                ready_refine.append((name, seed, "y", None))
        elif stage == "recipe":
            state["next"] += 1
            if state["next"] < len(state["recipes"]):
                ready_refine.append(
                    (name, seed, "recipe", state["recipes"][state["next"]]))
            else:
                ready_refine.append((name, seed, "y", None))
        elif stage == "y":
            ready_refine.append((name, seed, "eval", None))
        elif stage == "eval":
            ready_refine.append((name, seed, "prune", None))

    def launch(gpu: str, item: tuple[str, int, str, str | None]):
        name, seed, stage, recipe = item
        state = states[(name, seed)]
        unit_dir = units_dir / name / f"seed{seed}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        label = stage if recipe is None else f"recipe_{recipe}"
        unit_log = unit_dir / f"{label}.log"
        command = [
            args.python, "scripts/production/run_experiment1_ex_seed.py", "--config",
            str(state["study"].path), "--seed", str(seed), "--gpus", gpu,
            "--log-dir", str(unit_dir), "--retries", str(args.retries),
            "--patience", str(args.patience), "--stage", stage,
        ]
        if recipe is not None:
            command += ["--recipe", recipe]
        if args.force:
            command.append("--force")
        handle = unit_log.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        running[process.pid] = (*item, gpu, process, handle)
        log(scheduler_log, f"launch gpu={gpu} job={name}_seed{seed}_{label} pid={process.pid}")

    def next_ready() -> tuple[str, int, str, str | None] | None:
        refine_running = sum(
            1 for data in running.values() if data[2] != "parallel")
        if ready_refine and refine_running < max_refine:
            return ready_refine.popleft()
        if ready_parallel:
            return ready_parallel.popleft()
        if ready_refine:
            return ready_refine.popleft()
        return None

    def poll_external() -> None:
        for key in list(external):
            name, seed = key
            gpu, pid = external[key]
            state = states[key]
            checkpoint = state["study"].parallel_directory(state["run"]) / "last.pt"
            if checkpoint.is_file():
                log(scheduler_log,
                    f"external done job={name}_seed{seed}_parallel gpu={gpu}")
                del external[key]
                free_gpus.append(gpu)
                enqueue_next(name, seed, "parallel", None)
                continue
            if (parse_stage_parallel_cmdline(read_cmdline(pid)) is None
                    or pid_state(pid) in (None, "Z")):
                log(scheduler_log,
                    f"external lost job={name}_seed{seed}_parallel gpu={gpu}; "
                    f"re-queueing")
                del external[key]
                free_gpus.append(gpu)
                ready_parallel.append((name, seed, "parallel", None))

    def pump() -> None:
        while (ready_parallel or ready_refine or running or external
               or aggregate_running):
            while free_gpus:
                item = next_ready()
                if item is None:
                    break
                launch(free_gpus.popleft(), item)
            poll_external()
            poll_aggregate()
            for pid, data in list(running.items()):
                name, seed, stage, recipe, gpu, process, handle = data
                rc = process.poll()
                if rc is None:
                    continue
                handle.close()
                del running[pid]
                free_gpus.append(gpu)
                label = stage if recipe is None else f"recipe_{recipe}"
                if rc:
                    skipped[(name, seed)] = (stage, recipe)
                    log(failed_log,
                        f"FAILED job={name}_seed{seed}_{label} gpu={gpu} rc={rc}; "
                        f"skipping this seed")
                else:
                    log(scheduler_log, f"done job={name}_seed{seed}_{label} gpu={gpu}")
                    enqueue_next(name, seed, stage, recipe)
                    if stage == "prune":
                        state = states[(name, seed)]
                        (state_dir / f"{state['label']}.seed{seed}.done").touch()
                        study_done[name] = study_done.get(name, 0) + 1
                        if study_done[name] == study_total[name]:
                            start_aggregate(name)
            time.sleep(0.25)

    def stop(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        pump()
    except KeyboardInterrupt:
        for data in list(running.values()):
            terminate_group(data[5])
        log(scheduler_log, "interrupted: terminated running jobs")
        return 130

    if skipped:
        summary = ", ".join(f"{name}_seed{seed}" for name, seed in skipped)
        log(scheduler_log,
            f"SKIPPED {len(skipped)} seed(s) after retries exhausted: {summary}")
    if aggregate_failed:
        log(scheduler_log,
            f"AGGREGATE FAILED for: {', '.join(aggregate_failed)}")
    if skipped or aggregate_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
