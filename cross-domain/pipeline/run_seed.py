"""One (variant, upstream seed) full lifecycle, pinned to a GPU (jet-style).

Runs upstream training -> frozen cache -> every refiner (recipe x downstream
seed) for a single variant/seed, then writes a seed marker. Existing artifacts
are skipped so a re-run resumes. CUDA_VISIBLE_DEVICES is set before importing
torch so the process binds to one physical GPU.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from pipeline.context import load_context
    from pipeline.io import write_json
    from pipeline.run_unit import execute
    from pipeline.runtime import configure
    from pipeline.units import unit_filters

    context = load_context(args.config, DOMAIN_ROOT)
    configure(context.config.get("runtime", {}))
    context.initialize()

    def run(unit):
        for attempt in range(args.retries + 1):
            context.filters = unit_filters(unit)
            try:
                execute(context, unit)
                print(f"[seed {args.variant}:{args.seed}] {unit} done", flush=True)
                return True
            except Exception as error:  # noqa: BLE001 - reported and retried
                print(f"[seed {args.variant}:{args.seed}] {unit} attempt {attempt + 1} failed: {error}", flush=True)
        return False

    started = time.perf_counter()
    ok = True
    ok &= run(f"upstream:{args.variant}:{args.seed}")
    ok &= run(f"cache:{args.variant}:{args.seed}")
    recipes = context.module("data").applicable_recipes(context, args.variant)
    for recipe in recipes:
        for downstream_seed in context.config["refiner"]["seeds"]:
            ok &= run(f"refine:{args.variant}:{args.seed}:{recipe}:{downstream_seed}")
    log_root = context.resolve(context.config.get("log_root", "logs")) / context.config["dataset"] / context.config["experiment"]
    marker = log_root / "seeds" / f"{args.variant}__seed{args.seed}.json"
    write_json(marker, {"variant": args.variant, "seed": args.seed, "status": "complete" if ok else "failed",
                        "seconds": time.perf_counter() - started,
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
    if not ok:
        raise SystemExit(f"seed {args.variant}:{args.seed} failed")
    print(f"[seed {args.variant}:{args.seed}] complete ({time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
