"""Run modular cross-domain stages from any working directory."""

import argparse
from pathlib import Path
import sys

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT))

from pipeline.context import load_context
from pipeline.runtime import configure
from pipeline.stages import STAGES, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args()
    context = load_context(args.config, DOMAIN_ROOT)
    configure(context.config.get("runtime", {}))
    context.initialize()
    for stage in STAGES if args.stage == "all" else (args.stage,):
        execute(context, stage)


if __name__ == "__main__":
    main()
