# Experiment 2: fixed Transformer architecture, A-train data scaling

This matrix measures how the frozen-feature DNN gain changes as the upstream Parallel Transformer is trained on more A-train data. At every point, the Transformer-only result and Transformer + DNN result are evaluated from the same trained Parallel checkpoints on the same locked Y-test split.

| Parallel config | Exact parameters | d_model | heads | layers | d_ffn |
|---|---:|---:|---:|---:|---:|
| p056k | 56,381 | 32 | 2 | 6 | 64 |
| p122k | 122,077 | 48 | 4 | 6 | 96 |

| A-train config | A-train | A-val | B-train | B-val | Y-test |
|---|---:|---:|---:|---:|---:|
| a600k | 600,000 | 100,000 | 200,000 | 100,000 | 500,000 |
| a800k | 800,000 | 100,000 | 200,000 | 100,000 | 500,000 |
| a1m | 1,000,000 | 100,000 | 200,000 | 100,000 | 500,000 |
| a1500k | 1,500,000 | 100,000 | 200,000 | 100,000 | 500,000 |
| a2m | 2,000,000 | 100,000 | 200,000 | 100,000 | 500,000 |
| a3m | 3,000,000 | 100,000 | 200,000 | 100,000 | 500,000 |

The 12 experiment JSON files are the Cartesian product of the two fixed Transformer sizes and the six A-train scales. Every experiment uses the DNN component with hidden dimensions input -> 128 -> 64 -> 32 -> output. B-train/B-val remain a fixed downstream-training and selection resource; Y-test is locked for final evaluation only.

The existing 1M 56k run is represented explicitly by `experiment2_p056k_a1m.json`. It uses the already-existing `data_1m_b200k_y500k.json` split specification and records `parallel_refine_a1m_6layers` as its prior-result reference. Adding this matrix entry does not by itself reuse or alias that prior output directory: queue scripts should omit this entry when the existing result is accepted as complete.

Each new A scale has its own split and processed-cache directory because normalization is fitted from that scale's A-train split.

## Shared-split preparation

Before training, run the standalone preparation script once from the repository root:

```bash
python scripts/prepare_experiment2_shared_splits.py
```

The script generates the A=3M master split with the existing splitter, then derives nested A-train subsets for all six scales while copying the exact A-val, B-train, B-val, and Y-test indices. It writes the standard `indices.npz` and `split_manifest.json` files consumed by the existing training pipeline. Use `--force` only when intentionally regenerating the anchor and all derived bundles.

After the shared split is ready, the matrix can be run in the background with:

```bash
bash scripts/run_experiment2_matrix_queue.sh start
bash scripts/run_experiment2_matrix_queue.sh status
```

The queue uses its own logs/PID/completion directory under `logs/parallel_refine/experiment2_matrix_queue`. It contains the 11 outstanding entries; `experiment2_p056k_a1m.json` remains in this matrix but is omitted because the prior `parallel_refine_a1m_6layers` result is recorded separately. Use `run` for foreground debugging or `prepare` to run only the shared-split preparation.

For one configuration with explicit, manually staged commands, edit the `CONFIG` line in `scripts/run_experiments_plain_experiment2.sh` and run it from the repository root. The script prepares or verifies the shared split, then runs the five Parallel seeds, B caches, all five DNN recipes, Y caches, and final evaluation with explicit GPU assignments.
