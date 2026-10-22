"""One-upstream-at-a-time PASCAL resource lifecycle entry point."""
from pathlib import Path
import sys
DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(DOMAIN_ROOT))
from experiments.cv_pascal_context.workflow import validate as validate_context, unit_complete, seed_units


def main():
    from pipeline.lifecycle import main as run
    run()


if __name__ == "__main__":
    main()
