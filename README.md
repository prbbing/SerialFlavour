# SerialFlavour

Modular transformer-based jet flavour tagging with multi-model support. Train and compare different neural architectures on the ATLAS open dataset, all through a single entry point and JSON config files.

## Project structure

```
SerialFlavour/
├── train_origin_vertex_jet.py          Single entry point for all models
├── configs/
│   ├── staged.json                     Config for staged 3-encoder model
│   └── parallel.json                   Config for parallel 3-task model (GN2-inspired)
├── src/
│   ├── config.py                       _DEFAULTS dictionary, Config class, load_config()
│   ├── data.py                         HDF5 I/O, track pre-processing, caching, PyTorch Dataset
│   ├── losses.py                       Loss functions (vertex-fit, pair-vertex, class weights)
│   ├── training.py                     Training & validation loops with model-agnostic dispatch
│   ├── plotting.py                     Evaluation plots for all models
│   └── models/
│       ├── __init__.py                 Model registry + build_model() factory
│       ├── staged_origin_vertex_jet.py Staged 3-encoder architecture
│       └── parallel_origin_vertex_jet.py Shared-encoder + 3 parallel heads (GN2-inspired)
├── reference/                          Original single-file scripts (for reference)
└── local/                              Scratch space: notebooks, notes (gitignored)
```

## Quick start

```bash
python train_origin_vertex_jet.py --config configs/staged.json
python train_origin_vertex_jet.py --config configs/parallel.json
```

Training runs for 100 epochs by default and keeps three checkpoints in the timestamped output directory: `best_jet.pt` (lowest validation jet loss), `best_total.pt` (lowest validation total loss), and `last.pt` (latest completed epoch, normally epoch 100). The configurable `checkpoint_interval` setting also writes periodic checkpoints such as `epoch_20.pt`, `epoch_40.pt`, and so on; its default value is 20.

Evaluate any saved checkpoint with:

```bash
python train_origin_vertex_jet.py --config path/to/run/config.json \
    --eval-only --weights path/to/run/best_jet.pt
```

Evaluation outputs are written beside the checkpoint under `eval_<pt_name>/` (for example, `eval_best_jet/`). Use `--output-dir` to override the parent directory.

## Available models

### `staged_origin_vertex_jet` — Staged 3-encoder pipeline

Three dedicated transformer encoders connected in sequence through differentiable intermediate quantities. Gradients flow end-to-end through all three stages.

- **Stage 1** — Track-origin prediction (8 classes, per-track, no CLS token).
- **Stage 2** — Origin-gated differentiable secondary-vertex fit. Stage 1's soft origin probabilities gate track selection for each vertex leg (b-vertex, c-vertex). A closed-form weighted least-squares fit produces Lxy and dz predictions.
- **Stage 3** — Jet-flavour classification (b / c / light). Fitted vertex coordinates are embedded as vertex tokens, prepended with a CLS token before the track sequence, and fed through encoder 3.

```
x → encoder1 → origin_logits ──┐
       soft_probs               ↓ (gate)
                               encoder2 → vtx_weight → Lxy, dz
                                                         │
                               encoder3 → CLS → jet_logits
```

~55 k params. Outputs: `jet_logits`, `origin_logits`, `vtx_weight`, `lxy_pred`, `dz_pred`.

### `parallel_origin_vertex_jet` — Parallel 3-task transformer (GN2-inspired)

A single shared transformer encoder drives three parallel task heads, inspired by the ATLAS GN2 architecture. No sequential dependency between tasks.

- **Per-track init network** — 2-layer MLP with ReLU.
- **Shared TransformerEncoder** — Multi-layer self-attention over all tracks.
- **Attention pooling** — Learned weighted sum of track embeddings produces a global jet representation.
- **Three parallel heads** — jet-flavour classification (pooled → b/c/light), track-origin classification (per-track → 8 classes), and track-pair vertexing (pairwise Bilinear → same vertex or not).

```
x → init_net → encoder ─┬── pool_attn → jet_head → jet_logits
                         ├── origin_head → origin_logits
                         └── Bilinear → pair_logits
```

~55 k params. Outputs: `jet_logits`, `origin_logits`, `pair_logits`.

### Model summary

|                   | staged                            | parallel                                |
| ----------------- | --------------------------------- | --------------------------------------- |
| Encoders          | 3 independent (one per stage)     | 1 shared                                |
| Task organisation | Sequential (Stage 1 → 2 → 3)      | Parallel (3 heads from shared backbone) |
| Third task        | Closed-form Lxy/dz vertex fit     | Track-pair vertexing (binary BCE)       |
| Default config    | d=32, heads=4, layers=2, d_ffn=64 | d=32, heads=4, layers=6, d_ffn=64       |
| Parameters        | ~55 k                             | ~55 k                                   |

## Configuration

All parameters are defined in `src/config.py:_DEFAULTS`. A JSON file passed via `--config` overrides any subset; the rest inherit defaults. Key fields include `model_type`, `d_model`, `n_heads`, `n_layers`, `d_ffn`, `dropout`, `epochs`, `lr`, `batch_size`, `n_train`, `n_test`, `train_file`, `gpu_ids`, `train_plot_dir`, and `model_name`. See `configs/staged.json` and `configs/parallel.json` for complete examples. Both models share the same track cache — data is processed once.

Training outputs evaluation plots and classification reports under `train_plot_dir`; each model type generates its own diagnostic figures (vertex-fit comparisons for staged, pair-vertexing ROC for parallel).

## Extending

To add a new architecture, create a module under `src/models/`, register a builder in `src/models/__init__.py`, and provide a JSON config with `"model_type"`. If the model introduces new output keys, add the corresponding loss/metrics/plots following the existing annotation convention. No changes are needed to the training entry point or config system.


## Issue
Currently most value of vtx_weight in staged model is 0, though the training is going well. Detailed analysis shows that this because the `refine = torch.sigmoid(self.vertex_weight_head(h2))` becomes 0 in the training, which includes information from vertex encoder. This issue is under addressing.