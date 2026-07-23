# Train Sweep

## Problem This Solves

The original batch-experiment workflow starts one independent Python process for every configuration. Each process loads and retains its own training data, DataLoader, model, and runtime state, so multiple experiments can quickly exhaust system memory and consume large amounts of swap even when each model uses only a small fraction of the available GPU memory.

`train_sweep` places multiple models in one Python process so that they share one DataLoader, one CPU dataset, and one GPU copy of each batch. In a CUDA environment, the forward pass, backward pass, and optimizer step for each model are submitted to an independent CUDA stream, reducing duplicated system-memory usage while giving the GPU an opportunity to execute work from multiple models concurrently.

This approach primarily reduces duplicated dataset memory and Python-process overhead. Every model still has independent weights, gradients, Adam state, and activations, so the maximum concurrency remains limited by GPU memory. Actual speedup from CUDA streams also depends on the resource usage of the individual kernels.

## Usage

```bash
python -m train_sweep \
  --config configs/model_a.json \
  --config configs/model_b.json \
  --config configs/model_c.json \
  --max-concurrent 3 \
  --gpu auto \
  --seed 42
```

Main options:

- `--config`: Repeat this option once for each model. The supplied order is recorded in the manifest and determines the reproducible execution order.
- `--max-concurrent`: Maximum number of models submitted concurrently for each batch. The default is all models.
- `--gpu N`: Use GPU `N`, where `N` is the PyTorch-visible index and therefore follows `CUDA_VISIBLE_DEVICES`.
- `--gpu auto`: Use `nvidia-smi` to select the visible GPU with the most free memory.
- No `--gpu`: Use `gpu_ids` from the configurations. The runtime falls back to sequential CPU execution when CUDA is unavailable.
- `--seed`: Control model initialization and sweep training randomness. The default is `42`.

Run the predefined five-model sweep with:

```bash
bash train_sweep/run_train_sweep.sh --gpu auto
```

The shell script forwards additional arguments unchanged, so options such as `--gpu 1` or a lower `--max-concurrent` value can be supplied directly.

## Configuration Requirements

All configurations must be able to share the same dataset and batches. This requires matching training files, cache directories, `n_train`, `n_test`, `data_seed`, `top_k`, `batch_size`, field order, class mapping, vertex targets, and `use_pair_target`. Sweep training always uses `num_workers=0`, supports only one GPU, and does not support DataParallel.

`seed` and `data_seed` have different purposes:

- `seed` controls model initialization and training randomness. The command-line `--seed` value takes priority and is saved in the effective `config.json` for every run.
- `data_seed` comes from the configuration and controls the data split and cache identity. Configurations with different `data_seed` values cannot share one sweep.

## Outputs and Failure Handling

Each model retains an independent `train_plot_dir_<timestamp>/` containing checkpoints, training logs, TensorBoard data, CSV files, history, and plots. Every output directory also contains the same `sweep_manifest.json`, which records configuration order, GPU selection, seed, concurrency, timing, and run status.

The weights, Adam state, and activations of concurrent models occupy GPU memory at the same time. A CUDA OOM or asynchronous CUDA error stops the entire sweep without an automatic retry; rerun with a lower `--max-concurrent` value. Whether CUDA streams provide a real speedup should be evaluated using wall-clock time and profiler results.
