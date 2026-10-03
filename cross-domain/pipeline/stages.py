"""Stage dependency checks and artifact-backed completion records."""

import importlib
import time
from datetime import datetime, timezone

from pipeline.io import artifact_record, artifacts_match, read_json, write_json


STAGES = ("download", "prepare", "train", "cache", "refine", "evaluate", "analyze")


def execute(context, name):
    state_path = context.output_dir / "stage_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    identity = context.identity
    for dependency in STAGES[:STAGES.index(name)]:
        record = state.get(dependency, {})
        if record.get("status") != "complete" or record.get("identity") != identity or not artifacts_match(record.get("artifacts", [])):
            raise RuntimeError(f"{name} requires valid completed {dependency}; run --stage all or restore the dependency")
    previous = state.get(name, {})
    if previous.get("status") == "complete" and previous.get("identity") == identity and artifacts_match(previous.get("artifacts", [])):
        print(f"[{name}] complete artifacts verified; skip", flush=True)
        return
    # Downstream selections cannot survive a rebuilt prerequisite.
    for downstream in STAGES[STAGES.index(name) + 1:]:
        state.pop(downstream, None)
    started = time.perf_counter()
    state[name] = {"status": "running", "identity": identity,
                   "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(state_path, state)
    print(f"[{name}] start", flush=True)
    try:
        paths = importlib.import_module(f"pipeline.{name}").run(context)
        artifacts = [artifact_record(path) for path in paths]
        if not artifacts:
            raise RuntimeError("a completed stage must publish at least one artifact")
        state[name].update(status="complete", seconds=time.perf_counter() - started, artifacts=artifacts)
    except Exception as error:
        state[name].update(status="failed", seconds=time.perf_counter() - started,
                           error=f"{type(error).__name__}: {error}")
        write_json(state_path, state)
        raise
    write_json(state_path, state)
    print(f"[{name}] complete ({state[name]['seconds']:.2f}s)", flush=True)
