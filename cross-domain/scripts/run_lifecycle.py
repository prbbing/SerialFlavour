"""Run a package-declared resource-friendly frozen-upstream lifecycle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.lifecycle import main
if __name__ == "__main__":
    main()
