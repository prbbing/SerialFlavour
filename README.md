# SerialFlavour

Modular transformer-based jet flavour tagging on ATLAS open data. One entry point, JSON config files,
multi-model comparison.

## Project structure

```
SerialFlavour/
├── train_origin_vertex_jet.py      CLI entry point
├── configs/                        JSON experiment configs
├── src/
│   ├── config.py                   _DEFAULTS + Config class
│   ├── data.py                     HDF5 loading, caching, PyTorch Dataset
│   ├── losses.py                   Vertex-fit, pair-vertex, class-weighted losses
│   ├── training.py                 Training & validation loops
│   ├── plotting.py                 Evaluation and diagnostic plots
│   └── models/                     Model implementations + registry
│       ├── staged_origin_vertex_jet.py           Staged (multiplicative refine)
│       ├── staged_origin_vertex_jet_fix_refine.py Staged (additive refine)
│       ├── staged_origin_vertex_jet_no_refine.py  Staged (no refine)
│       └── parallel_origin_vertex_jet.py          Parallel (GN2-inspired)
├── reference/                      Original single-file reference scripts
└── local/                          Notebooks, notes, scratch (gitignored)
```

## Quick start

```bash
python train_origin_vertex_jet.py --config configs/staged.json
python train_origin_vertex_jet.py --config configs/parallel.json
```

Dense pair-supervision targets are controlled by `use_pair_target`, which
defaults to `false`. Parallel model configs must set it to `true`; staged
models leave it disabled and avoid materialising the dense `(N, K, K)` array.

Training writes checkpoints to a timestamped output directory: `best_jet.pt`, `best_total.pt`, `last.pt`, plus
periodic `epoch_N.pt` snapshots (interval configurable via `checkpoint_interval`, default 20).

Evaluate a checkpoint:

```bash
python train_origin_vertex_jet.py --config path/to/run/config.json \
    --eval-only --weights path/to/run/best_jet.pt
```

Outputs land in `eval_<pt_name>/`. Override with `--output-dir`.

## Available models

### `staged_origin_vertex_jet` — Staged pipeline

Three transformer encoders connected sequentially through differentiable intermediates:

```
x → encoder1 → origin_logits ─┐
       soft_probs              ↓
                               encoder2 → vtx_weight → Lxy, dz
                                                         │
                               encoder3 → CLS → jet_logits
```

- **Stage 1** — Track-origin classification (8 classes, per-track).
- **Stage 2** — Origin-gated vertex fit. `vtx_weight` selects tracks per leg (b/c vertex). A closed-form
  weighted fit produces Lxy and dz predictions.
- **Stage 3** — Jet-flavour classification (b / c / light) with vertex tokens prepended to the sequence.

Three staged variants share this architecture with different Stage 2 weighting:

| model_type                               | Stage 2 weighting                  | Params |
|------------------------------------------|------------------------------------|--------|
| `staged_origin_vertex_jet`               | `refine * gate` (multiplicative)   | ~55 k  |
| `staged_origin_vertex_jet_fix_refine`    | `clamp(gate + Δw, 0, 1)` per coord | ~56 k  |
| `staged_origin_vertex_jet_no_refine`     | encoder2 removed; `gate` only       | ~38 k  |

#### Vertex fitting methods

Selectable via `vertex_fit_method` (default `"wls_3d"`):

| Method       | Description                                                   |
|-------------|---------------------------------------------------------------|
| `"wls_3d"`  | Joint 3D WLS solve for (X,Y,Z); Lxy = √(X²+Y²). **Recommended.** |
| `"two_step"`| WLS Lxy + flight-phi, then two-step Z from Lxy geometry.       |
| `"old_dz"`  | WLS Lxy, dz = Σ(w·z0st/σ²)/Σ(w/σ²). Historical, not recommended. |

### `parallel_origin_vertex_jet` — Parallel pipeline (GN2-inspired)

Single shared encoder with three parallel heads, no sequential dependency:

```
x → init_net → encoder ─┬── pool_attn → jet_head → jet_logits
                         ├── origin_head → origin_logits
                         └── Bilinear → pair_logits
```

~55 k params. Outputs: `jet_logits`, `origin_logits`, `pair_logits`.

### Model summary

|               | staged               | fix_refine          | no_refine           | parallel               |
|--------------|----------------------|---------------------|---------------------|------------------------|
| Encoders     | 3 (one per stage)    | 3                   | 2 (no encoder2)      | 1 shared               |
| Vertex task  | closed-form WLS      | closed-form WLS     | closed-form WLS     | track-pair BCE         |
| Params       | ~55 k                | ~56 k               | ~38 k               | ~55 k                   |

## Configuration

All parameters live in `src/config.py:_DEFAULTS`. A JSON file via `--config` overrides any subset.
Key fields: `model_type`, `d_model`, `n_heads`, `n_layers`, `d_ffn`, `dropout`, `epochs`, `lr`,
`batch_size`, `n_train`, `n_test`, `train_file`, `gpu_ids`.

Reproducibility uses two independent config values (both default to `42`):

- `seed` controls model initialisation, training RNG and DataLoader shuffling.
- `data_seed` controls the train/test sample selection and is included in each track-cache ID.

Vertex-fit options shared by all staged models:

- `vertex_fit_method` — `"wls_3d"` (default), `"two_step"`, `"old_dz"`
- `vertex_fit_reg` — Tikhonov regularisation for 3D WLS (default `1e-6`)
- `vertex_fit_coords` — subset of `["Lxy", "dz"]`
- `calibrate_vertex_fit` — per-leg learned calibration (default `true`)
- `stage3_extra_inputs` — subset of `["origin_probs", "vtx_weight"]`
- `delta_w_amp` — additive delta-weight amplitude for fix_refine (default `0.5`)

## Extending

Create a module under `src/models/`, register a builder in `src/models/__init__.py`, add a JSON config
with `"model_type"`. New output keys need corresponding loss/metrics/plots. No changes to the entry point
or config system required.

## Known findings (2026-07-17)

- **Dz formula corrected.** The original `"old_dz"` method (`dz = Σ(w·z0st/σ²)/Σ(w/σ²)`) ignored the
  Lxy projection, inflating the dz calibration scale to 14–37×. The corrected `"wls_3d"` achieves
  b-vertex dz Pearson r = 0.92 (up from 0.51) and is now the default.
- **Refine has limited effect.** The multiplicative refine weight tends toward small values; only
  `staged_3dwls` shows a weak positive signal/background ratio (4×). The gate (from Stage 1 origin
  probabilities) drives most of the vertex-weighting signal.
- **No-refine model competitive.** Removing encoder2 and matching parameters (3 encoder layers) yields
  the best jet accuracy (0.786) among all variants, consistent with GN2's finding that one auxiliary
  task can be dropped with limited impact.

Detailed tables and figures in `results/update_2026-7-17.pdf`.
