# SerialFlavour

## Project Purpose

SerialFlavour is a configurable PyTorch project for studying transformer-based jet-flavour tagging on ATLAS open data. The parallel model follows the multi-task design of the ATLAS GN2 algorithm described in [“Transforming jet flavour tagging at ATLAS”](https://arxiv.org/abs/2505.19689), using one shared transformer encoder with independent heads for jet-flavour classification, track-origin classification, and track-pair vertex compatibility.

The project investigates a staged alternative in which track-origin predictions guide differentiable vertex reconstruction before jet-flavour classification. The staged and parallel designs are compared in terms of overall performance and how they construct and process jet-level representations, without assuming that either design is superior.

The project supports controlled architecture ablations, reproducible data splits and training seeds, cached HDF5 preprocessing, single-model training, checkpoint evaluation, and memory-efficient multi-model sweeps.

## Project Architecture

```text
SerialFlavour/
├── train_origin_vertex_jet.py   Single-model training and evaluation entry point
├── train_sweep/                 Single-process, multi-model CUDA-stream training
├── configs/                     JSON experiment and ablation configurations
├── src/
│   ├── config.py                Defaults, validation, seed utilities, derived settings
│   ├── data_fast.py             Accelerated HDF5 loading, cache handling, DataLoaders
│   ├── data.py                  Reference data implementation
│   ├── models/                  Model implementations and registry
│   ├── losses.py                Jet, origin, vertex, and pair losses
│   ├── training.py              Training, validation, checkpoint, and history logic
│   └── plotting.py              Evaluation and diagnostic plots
├── tests/                       Regression and model-contract tests
├── reference/                   Historical reference implementations
├── results/                     Saved reports and selected experiment outputs
└── local/                       Local scripts, notes, and machine-specific workflows
```


The parallel model uses one shared transformer encoder for three parallel tasks: b/c/light jet-flavour classification, per-track origin classification, and same-vertex track-pair classification.

The staged model uses separate transformer stages for three linked tasks: Stage 1 predicts per-track origin classes, Stage 2 uses the origin predictions to assign tracks and perform differentiable secondary-vertex fits, and Stage 3 predicts b/c/light jet flavour from track and optional vertex information.

| Model family                 | Purpose                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `parallel_origin_vertex_jet` | Shared encoder with parallel jet-flavour, track-origin, and track-pair tasks |
| `staged_origin_vertex_jet`   | Sequential track-origin, secondary-vertex, and jet-flavour tasks             |

The remaining model variants and ablation architectures are experimental and are still being evaluated.


## Basic Usage

Train one model:

```bash
python train_origin_vertex_jet.py --config configs/staged.json
```

Evaluate a checkpoint:

```bash
python train_origin_vertex_jet.py \
  --config path/to/run/config.json \
  --eval-only \
  --weights path/to/run/best_jet.pt
```

Train several compatible configurations in one process:

```bash
python -m train_sweep \
  --config configs/model_a.json \
  --config configs/model_b.json \
  --max-concurrent 2 \
  --gpu auto \
  --seed 42
```

See [`train_sweep/README.md`](train_sweep/README.md) for sweep constraints and failure handling.

## Configuration

All supported configuration keys and defaults are defined in `src/config.py`. A JSON file passed through `--config` overrides only the required values.

Important fields include `model_type`, `train_file`, `train_cache_dir`, `n_train`, `n_test`, `top_k`, `batch_size`, `epochs`, `lr`, `gpu_ids`, `use_pair_target`, `seed`, and `data_seed`.

- `seed` controls model initialization, training randomness, DataLoader shuffling, and worker seeds.
- `data_seed` controls the train/test split and is included in the track-cache identity.
- `use_pair_target` defaults to `false`; the parallel model requires `true`, while staged models avoid retaining the dense `(N, K, K)` target.

Training writes an effective `config.json`, `best_jet.pt`, `best_total.pt`, `last.pt`, periodic checkpoints, logs, histories, TensorBoard data, and plots to a timestamped output directory.

## Using datasets

The development dataset is [`mc-flavtag-ttbar-small.h5`](D:/hep_analysis/gn2_study/opendata_tt/mc-flavtag-ttbar-small.h5), a 3.06 GB (2.85 GiB) gzip-compressed compound-HDF5 \(t\bar{t}\) Monte Carlo sample. Set `train_file` to its location on the target machine.

| Item | Shape / value | Contents |
| --- | ---: | --- |
| `/eventwise` | `1,365,435` | Jet count per event and primary-vertex displacements |
| `/jets` | `5,619,475` | Jet kinematics, truth labels and matching, GN2v01, and DL1dv01 scores |
| `/tracks` | `5,619,475 × 40` | Kinematics, impact parameters and uncertainties, hit counts, truth origin/vertex, and `valid` padding mask |
| `/truth_hadrons` | `5,619,475 × 5` | PDG IDs, kinematics, decay vertices, and \(L_{xy}\) |
| Event summary | `4.12` jets/event | `eventwise.nJets` sums to `5,619,475`, matching `/jets` |
| Uniform 131k-jet sample | `52.3 GeV`; `7.6` | Median jet \(p_T\); mean valid tracks per jet |

| Truth flavour | Jets | Share | Used by default |
| --- | ---: | ---: | --- |
| Light | `2,805,185` | 49.92% | Yes |
| b | `2,115,856` | 37.65% | Yes |
| c | `493,817` | 8.79% | Yes |
| Tau | `204,617` | 3.64% | No |

`HadronConeExclTruthLabelID` defines the flavour labels, and 98.37% of jets are successfully matched to a truth jet.

## Known findings (2026-07-17)

- **Dz formula corrected.** The original `"old_dz"` method (`dz = Σ(w·z0st/σ²)/Σ(w/σ²)`) ignored the Lxy projection, inflating the dz calibration scale to 14–37×. The corrected `"wls_3d"` achieves b-vertex dz Pearson r = 0.92 (up from 0.51) and is now the default.
- **Refine has limited effect.** The multiplicative refine weight tends toward small values; only `staged_3dwls` shows a weak positive signal/background ratio (4×). The gate (from Stage 1 origin probabilities) drives most of the vertex-weighting signal.
- **No-refine model competitive.** Removing encoder2 and matching parameters (3 encoder layers) yields the best jet accuracy (0.786) among all variants, consistent with GN2's finding that one auxiliary task can be dropped with limited impact.

Detailed tables and figures in `results/update_2026-7-17.pdf`.

## Ongoing Research Problems

1. We are studying whether each stage encoder and its input features can be simplified without reducing overall model performance.
2. We are studying whether track-pair and track-origin supervision contain the same task overlap in the staged architecture that has been observed for GN2.
