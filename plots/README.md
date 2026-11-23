# SerialFlavour paper figures

`figure/` contains the seven PNGs used in the paper, including the workflow diagram
(Figure 1). The workflow diagram is distributed as a PNG.

The six experimental figures can be redrawn from the prepared JSON files in
`data/prepared/`. No raw datasets, model checkpoints, training, or evaluation are
required. The prepared catalog identifies which input file each figure uses and records its provenance.

## Reproduce Figures 2–7

Use Python 3.11 or later. Run these commands from the `plots` directory. The
`--plot-only` flag uses the bundled inputs, and the separate output directory
preserves the paper PNGs in `figure/`.

```sh
python -m pip install -r ../requirements.txt
python scripts/figures/rejection_3m_own_f1_f2_f3.py --plot-only --output-dir rendered
python scripts/figures/overlay_3m_width_scan_six_working_points.py --plot-only --output-dir rendered
python scripts/figures/upstream_f3_sorted.py --plot-only --output-dir rendered
python scripts/figures/overlay_3m_width_scan_training_reuse_light.py --plot-only --output-dir rendered
python scripts/figures/discriminant_vs_vertices.py --plot-only --output-dir rendered
python scripts/figures/overlay_1m_5m_six_working_points.py --plot-only --output-dir rendered
```

The commands are listed in paper figure order. Each command writes a PNG and a
render report containing the input and output checksums.
