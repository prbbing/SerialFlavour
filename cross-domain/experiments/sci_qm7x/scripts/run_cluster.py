"""QM7-X wrapper for the shared resource-friendly seed lifecycle."""
from pathlib import Path
import sys

DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(DOMAIN_ROOT))
from pipeline.lifecycle import main, gpu_mapping
from experiments.sci_qm7x.workflow import plan

if __name__ == '__main__':
    main()
