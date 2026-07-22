"""Command-line entry point for single-process multi-model training."""
import argparse

from .runtime import normalise_gpu_request, run_sweep


def _parse_gpu(value):
    try:
        return normalise_gpu_request(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train multiple SerialFlavour configs in one process, "
                    "using independent CUDA streams on one GPU.")
    parser.add_argument(
        "--config", action="append", required=True,
        help="JSON config path. Repeat once per model, in deterministic order.")
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Maximum models submitted concurrently per batch (default: all).")
    parser.add_argument(
        "--gpu", type=_parse_gpu, default=None, metavar="{auto,N}",
        help="Override config gpu_ids with a visible CUDA index, or auto-select "
             "the GPU with the most free memory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run_sweep(
        args.config, max_concurrent=args.max_concurrent, seed=args.seed,
        gpu=args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
