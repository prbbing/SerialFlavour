"""Evaluate the complete locked readout matrix on Y, once choices are fixed."""

import numpy as np

from experiments.qm9.data import array_hash, load_arrays, split_indices
from experiments.qm9_native_graph.identity import bind_sources
from experiments.qm9_native_graph.plan import all_configs
from experiments.qm9_native_graph.refine import load_cache, predict_refiner, refiner_dir
from pipeline.io import sha256_file, write_json
from pipeline.runtime import configure


def evaluate(context):
    binding = bind_sources(context)
    items = all_configs(context.config)
    # Fail before opening Y if the full planned grid is not ready.
    if any(not (refiner_dir(context, item) / "best.pt").is_file() for item in items):
        raise ValueError("incomplete readout grid; final Y evaluation is locked")
    device = configure(context.config["runtime"])
    prepared = load_arrays(context)
    y_ids = prepared["ids"][split_indices(context, "y_test")]
    rows, artifacts = [], [binding]
    directory = context.output_dir / "evaluation"
    for variant in context.config["upstream"]["variants"]:
        for us in context.config["upstream"]["seeds"]:
            arrays = load_cache(context, variant, us, "y_test")
            if not np.array_equal(arrays["ids"], y_ids):
                raise ValueError("Y IDs/order differ from the locked prepared split")
            native = float(np.abs(arrays["main_prediction"] - arrays["target"]).mean())
            rows.append({"variant": variant, "upstream_seed": us, "recipe": "native", "mae_eV": native,
                         "b_train_size": None, "downstream_seed": None, "selected_epoch": None})
            for item in items:
                if item["variant"] != variant or item["upstream_seed"] != us:
                    continue
                prediction, epoch = predict_refiner(context, item, arrays, device)
                prediction_path = (directory / variant / f"seed{us}" / f"b{item['b_train_size']}"
                                   / item["recipe"] / f"seed{item['downstream_seed']}.npz")
                from experiments.qm9.data import save_arrays
                artifacts.append(save_arrays(prediction_path, ids=y_ids, target=arrays["target"], prediction_eV=prediction))
                rows.append({**item, "mae_eV": float(np.abs(prediction - arrays["target"]).mean()),
                             "selected_epoch": int(epoch), "prediction_sha256": sha256_file(prediction_path),
                             "checkpoint_sha256": sha256_file(refiner_dir(context, item) / "best.pt")})
    path = directory / "metrics.json"
    write_json(path, {"identity": context.identity, "y_ids_sha256": array_hash(y_ids), "rows": rows,
                      "selection": "B_val gap MAE; Y used only for locked final scoring", "unit": "eV"})
    return [*artifacts, path]
