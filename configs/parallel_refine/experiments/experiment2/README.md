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

## Running one configuration

The one-off shared-split materializer and matrix queue have been retired. Select an experiment JSON with the repository's single plain runner, either by editing its `CONFIG` line or by overriding it for one invocation:

```bash
PARALLEL_REFINE_CONFIG=configs/parallel_refine/experiments/experiment2/experiment2_p122k_a1m.json \
  bash scripts/run_parallel_refine_experiment.sh
```

The runner executes data preparation, the five Parallel seeds, B caches, configured DNN or graph recipes, Y caches, evaluation, and cross-Parallel-seed aggregation. Completed training units are skipped through their existing completion checks.

Existing Experiment 2 results may depend on previously materialized split bundles whose A-val, B-train, B-val, and Y-test indices were shared across A-train scales. Keep those `indices.npz` and `split_manifest.json` artifacts when reproducing the archived results. If the split directories are absent, the generic runner can create config-sized splits, but that does not by itself reconstruct the retired cross-configuration shared-split procedure.
