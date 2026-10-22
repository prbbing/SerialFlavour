"""Paired comparisons: downstream means within upstream, then sample SD."""

import numpy as np

from experiments.qm9_native_graph.identity import bind_sources
from experiments.qm9_native_graph.plan import all_configs
from pipeline.io import read_json, write_json

CONTRASTS = {
    "geometry_to_bond": ("geometry", "bond"),
    "charge_increment": ("bond", "bond_charge"),
    "connectivity": ("uniform", "existence"),
    "bond_type": ("existence", "bond_charge"),
    "edge_alignment": ("shuffle", "bond_charge"),
    "geometry_to_hidden": ("geometry", "aux_hidden"),
    "native_to_geometry": ("native", "geometry"),
    "native_to_bond_charge": ("native", "bond_charge"),
}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_upstream": len(values), "n_positive": int((values > 0).sum())}


def analyze(context):
    binding = bind_sources(context)
    metrics = read_json(context.output_dir / "evaluation" / "metrics.json")
    if metrics["identity"] != context.identity:
        raise ValueError("evaluation identity mismatch")
    keys = ("variant", "upstream_seed", "b_train_size", "recipe", "downstream_seed")
    expected = {tuple(item[key] for key in keys) for item in all_configs(context.config)}
    expected |= {(v, u, None, "native", None) for v in context.config["upstream"]["variants"]
                 for u in context.config["upstream"]["seeds"]}
    lookup = {tuple(row[key] for key in keys): row["mae_eV"] for row in metrics["rows"]}
    if len(lookup) != len(metrics["rows"]) or set(lookup) != expected or not all(np.isfinite(list(lookup.values()))):
        raise ValueError("evaluation grid is incomplete, duplicated, unexpected, or nonfinite")
    methods, paired = [], []
    for variant in context.config["upstream"]["variants"]:
        for budget in context.config["nested_b"]["budgets"]:
            recipes = {row["recipe"] for row in metrics["rows"] if row["variant"] == variant
                       and row["b_train_size"] == budget} | {"native"}
            grouped = {}
            for recipe in sorted(recipes):
                grouped[recipe] = {}
                for us in context.config["upstream"]["seeds"]:
                    # Native is one measurement, never extra downstream replicas.
                    if recipe == "native":
                        grouped[recipe][us] = lookup[(variant, us, None, "native", None)]
                    else:
                        grouped[recipe][us] = float(np.mean([lookup[(variant, us, budget, recipe, ds)]
                                                           for ds in context.config["refiner"]["seeds"]]))
                methods.append({"variant": variant, "b_train_size": budget, "recipe": recipe,
                                **summary(list(grouped[recipe].values()))})
            for name, (first, second) in CONTRASTS.items():
                if first not in grouped or second not in grouped:
                    continue
                values = [grouped[first][us] - grouped[second][us] for us in context.config["upstream"]["seeds"]]
                paired.append({"variant": variant, "b_train_size": budget, "contrast": name,
                               "first": first, "second": second, **summary(values),
                               "per_upstream": dict(zip(map(str, context.config["upstream"]["seeds"]), values))})
    path = context.output_dir / "analysis" / "summary.json"
    write_json(path, {"identity": context.identity, "methods": methods, "paired_deltas": paired,
        "unit": "eV", "positive_delta": "MAE(first)-MAE(second); positive favors second",
        "statistical_unit": "upstream seed; average nested downstream seeds first; sample SD ddof=1",
        "limits": "fixed A/Y and nested B; budgets are paired dependent settings, not independent repetitions"})
    return [binding, path]
