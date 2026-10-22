"""Legacy seed-only entry is retired; use the full receipt-aware lifecycle."""
from pathlib import Path
import sys

def main():
    if any(flag in sys.argv for flag in ("--variant", "--seed", "--gpu")):
        raise SystemExit("Seed-only scheduling is retired. Use scripts/run_cluster.py --config CONFIG --gpus GPU to execute the fixed full lifecycle with receipts and safe cleanup.")
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from pipeline.lifecycle import main as run
    run()

if __name__ == "__main__":
    main()
