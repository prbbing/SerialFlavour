#!/usr/bin/env python3
"""Export upstream rejection scans from existing locked-Y prediction files."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.rejection_scan import build_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-directory", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    pending = []
    for study in args.study_directory:
        for seed in args.seed or range(1, 6):
            directory = study / "evaluation" / f"parallel_seed{seed}" / "parallel"
            destination = directory / "rejection_scan.json"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite: {destination}")
            payload = build_scan(directory)
            encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            pending.append((destination, encoded))
    for destination, encoded in pending:
        temporary = destination.with_name(".rejection_scan.json.tmp")
        with temporary.open("x", encoding="utf-8") as output:
            output.write(encoded)
        temporary.replace(destination)
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if len(saved["curves"]) != 284:
            raise ValueError(f"Export verification failed: {destination}")
        print(f"EXPORTED {destination} rows=284 bytes={destination.stat().st_size}", flush=True)
    print(f"COMPLETE files={len(pending)} working_point_checks={len(pending) * 4}")


if __name__ == "__main__":
    main()
