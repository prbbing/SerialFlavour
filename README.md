# SerialFlavour

SerialFlavour studies GN2-like multi-task jet-flavour tagging on ATLAS Open Data.
The retained upstream model uses one shared Transformer encoder with jet-flavour, track-origin, and track-pair vertex-compatibility heads.

## Project Architecture

```text
SerialFlavour/
├── src/
│   ├── config.py                Shared Parallel defaults and seed utilities
│   ├── data.py                  Shared HDF5 and atomic-cache helpers
│   ├── losses.py                Origin weighting and pair BCE
│   ├── parallel_model.py         Parallel Transformer implementation
│   ├── training.py               Reusable Parallel training and validation loop
│   └── parallel_refine/         A/B/Y split, cache, refiner, evaluation, and plots
├── configs/parallel_refine/     Component and experiment configurations
├── local/parallel_refine.md     Parallel Refine research contract (local notes)
├── scripts/                     Python preparation, training, and evaluation entry points
├── local/scripts/               Local multi-GPU shell runners
└── tests/                       Parallel regression tests
```

## Workflow

`parallel_refine` is a two-stage experiment with event-disjoint A/B/Y data.
A-train/A-val train and select the upstream Parallel checkpoint.
B-train/B-val fit and select frozen-feature DNN readouts.
Y-test is reserved for final locked evaluation only.

Run the configured stages from the repository root:

```bash
python scripts/prepare_data.py --config configs/parallel_refine/experiments/default.json
python scripts/train_parallel.py --config configs/parallel_refine/experiments/default.json
python scripts/generate_cache.py --config configs/parallel_refine/experiments/default.json
python scripts/train_dnn.py --config configs/parallel_refine/experiments/default.json
python scripts/evaluate.py --config configs/parallel_refine/experiments/default.json
```

See [`local/parallel_refine.md`](local/parallel_refine.md) for the experiment contract, feature recipes, and runner commands.

## Using datasets

The development dataset is [`mc-flavtag-ttbar-small.h5`](D:/hep_analysis/gn2_study/opendata_tt/mc-flavtag-ttbar-small.h5), a 3.06 GB (2.85 GiB) gzip-compressed compound-HDF5 \(t\bar{t}\) Monte Carlo sample.

| Item | Shape / value | Contents |
| --- | ---: | --- |
| `/jets` | `5,619,475` | Jet kinematics and flavour truth labels |
| `/tracks` | `5,619,475 × 40` | Track features, origin/vertex truth, and `valid` mask |
| `/truth_hadrons` | `5,619,475 × 5` | Hadron truth records |
| Event summary | `4.12` jets/event | `eventwise.nJets` sums to `5,619,475` |

`HadronConeExclTruthLabelID` maps b, c, and light jets to the three-class target; tau jets are excluded by default.
