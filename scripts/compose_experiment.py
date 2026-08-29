#!/usr/bin/env python3
"""Create a lightweight experiment config that references component configs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.config import (
    COMPONENT_KEYS, load_study_config, write_json_atomic)


def _read_component(kind: str, path: Path):
    values = json.loads(path.read_text(encoding="utf-8"))
    if set(values) != COMPONENT_KEYS[kind]:
        raise ValueError(
            f"{kind} component must contain exactly "
            f"{sorted(COMPONENT_KEYS[kind])}, got {sorted(values)}")
    return values


def compose(experiment_name, output_root, components):
    return {
        "config_kind": "experiment",
        "experiment": {
            "name": experiment_name,
            "output_root": output_root,
        },
        "components": dict(components),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--parallel-config", required=True)
    parser.add_argument("--refiner-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite experiment config: {output}; use --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    references = {}
    for kind, argument in {
            "data": args.data_config,
            "parallel": args.parallel_config,
            "refiner": args.refiner_config}.items():
        component = Path(argument).resolve()
        _read_component(kind, component)
        references[kind] = Path(os.path.relpath(
            component, output.parent.resolve())).as_posix()
    payload = compose(args.experiment_name, args.output_root, references)

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=".experiment-check-",
            dir=output.parent, delete=False, encoding="utf-8") as handle:
        candidate = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        load_study_config(candidate)
    finally:
        candidate.unlink(missing_ok=True)

    write_json_atomic(output, payload)
    print(f"wrote lightweight experiment config: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
