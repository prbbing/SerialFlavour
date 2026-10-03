"""Execute one work unit (dataset-agnostic) without touching global stage state.

Usage:
    python cross-domain/scripts/run_unit.py --config <config> --unit <unit-id>

Each unit writes a marker JSON under ``<log_root>/<dataset>/<experiment>/units/``
with status, timings, and artifact checksums. Filters derived from the unit id
restrict the domain loops but do not enter ``context.identity``.
"""

import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT / "src"))

from pipeline.context import load_context
from pipeline.io import artifact_record, write_json
from pipeline.runtime import configure
from pipeline.units import unit_filters
from pipeline.worker import execute, marker_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    context = load_context(args.config, DOMAIN_ROOT)
    context.filters = unit_filters(args.unit)
    configure(context.config.get("runtime", {}))
    context.initialize()
    path = marker_path(context, args.unit)
    started = time.perf_counter()
    try:
        artifacts = execute(context, args.unit)
        records = [artifact_record(item) for item in artifacts]
        write_json(path, {"unit": args.unit, "filters": context.filters, "status": "complete",
                          "identity": context.identity, "seconds": time.perf_counter() - started,
                          "finished_utc": datetime.now(timezone.utc).isoformat(), "artifacts": records})
        print(f"[unit] {args.unit} complete ({len(records)} artifacts)", flush=True)
    except Exception as error:
        write_json(path, {"unit": args.unit, "filters": context.filters, "status": "failed",
                          "identity": context.identity, "seconds": time.perf_counter() - started,
                          "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
