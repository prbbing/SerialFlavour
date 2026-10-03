"""Complete-grid, upstream-grouped comparisons of frozen QM9 readouts."""

from collections import defaultdict
import numpy as np

from data.qm9 import applicable_recipes
from evaluate.qm9 import write_csv
from pipeline.io import read_json, write_json


def summarize_values(values):
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("empty or non-finite analysis values")
    sd = float(values.std(ddof=1)) if len(values) > 1 else None
    return {"mean": float(values.mean()), "sample_sd": sd,
            "standard_error": None if sd is None else sd / np.sqrt(len(values)),
            "n_upstream_seeds": len(values)}


def analyze(context):
    metrics = read_json(context.output_dir / "evaluation" / "metrics.json")
    if metrics["identity"] != context.identity:
        raise ValueError("evaluation identity mismatch")
    upstream_seeds = context.config["upstream"]["seeds"]
    downstream_seeds = context.config["refiner"]["seeds"]
    if len(set(upstream_seeds)) != len(upstream_seeds) or len(set(downstream_seeds)) != len(downstream_seeds):
        raise ValueError("duplicate configured seeds")
    expected = {}
    native_methods = {"ST-native", "MT-native"}
    for variant in context.config["upstream"]["variants"]:
        prefix = "ST" if variant == "single_task" else "MT"
        expected[f"{prefix}-native"] = {(u, None) for u in upstream_seeds}
        for recipe in applicable_recipes(context, variant):
            expected[f"{prefix}-{recipe}"] = {(u, d) for u in upstream_seeds for d in downstream_seeds}
    lookup, observed = {}, defaultdict(set)
    for row in metrics["main"]:
        method, u, d = row["method"], row["upstream_seed"], row["downstream_seed"]
        key = (method, u, d)
        if key in lookup:
            raise ValueError(f"duplicate evaluation row: {key}")
        if not np.isfinite(row["mae"]):
            raise ValueError("non-finite evaluation MAE")
        lookup[key] = row["mae"]
        observed[method].add((u, d))
    if dict(observed) != expected:
        raise ValueError("incomplete or unexpected method/seed evaluation grid")

    per_method, methods = {}, []
    for method in expected:
        per_method[method] = {}
        for u in upstream_seeds:
            replicas = [lookup[(method, u, d)] for d in ([None] if method in native_methods else downstream_seeds)]
            per_method[method][u] = float(np.mean(replicas))
        methods.append({"method": method, **summarize_values(per_method[method].values()),
                        "n_runs": len(expected[method])})

    candidates = {
        "pool_to_local_gain_mae": ("MT-R0", "MT-R3"),
        "predicted_local_gain_mae": ("MT-R3", "MT-R4"),
        "auxiliary_readout_gain_mae": ("MT-R0", "MT-R2"),
        "auxiliary_supervision_gain_mae": ("ST-R0", "MT-R0"),
        "native_auxiliary_supervision_gain_mae": ("ST-native", "MT-native"),
        "random_readout_gain_mae": ("MT-native", "MT-R0"),
        "st_native_initialized_readout_gain_mae": ("ST-native", "ST-R0-native"),
        "mt_native_initialized_readout_gain_mae": ("MT-native", "MT-R0-native"),
        "native_initialized_auxiliary_readout_gain_mae": ("MT-R0-native", "MT-R2-native"),
        "predicted_graph_gain_given_H_mae": ("MT-R3-graph", "MT-R4-nocharge"),
        "charge_gain_given_graph_mae": ("MT-R4-nocharge", "MT-R4"),
        "connectivity_gain_mae": ("MT-R4-uniform", "MT-R4-existence"),
        "bond_order_gain_mae": ("MT-R4-existence", "MT-R4"),
        "graph_assignment_gain_mae": ("MT-R4-shuffle", "MT-R4"),
    }
    contrasts = {name: pair for name, pair in candidates.items() if all(method in expected for method in pair)}
    paired = []
    for u in upstream_seeds:
        for d in downstream_seeds:
            record = {"upstream_seed": u, "downstream_seed": d}
            for name, (first, second) in contrasts.items():
                first_d = None if first in native_methods else d
                second_d = None if second in native_methods else d
                record[name] = lookup[(first, u, first_d)] - lookup[(second, u, second_d)]
            paired.append(record)
    per_upstream = [{"upstream_seed": u, **{name: per_method[first][u] - per_method[second][u]
                                          for name, (first, second) in contrasts.items()}}
                    for u in upstream_seeds]
    deltas = {}
    for name in contrasts:
        values = [row[name] for row in per_upstream]
        deltas[name] = {**summarize_values(values),
                        "n_improved_upstream_seeds": int(np.count_nonzero(np.asarray(values) > 0))}

    directory = context.output_dir / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / "summary.json"
    write_json(summary, {"identity": context.identity, "methods": methods, "paired_deltas": deltas,
        "contrasts": {name: list(pair) for name, pair in contrasts.items()},
        "aggregation": "average downstream seeds within upstream seed, then mean/sample SD across upstream seeds",
        "positive_delta": "MAE(first) - MAE(second); positive means the second method has lower MAE",
        "statistical_unit": "upstream initialization seed on one fixed dataset/split; downstream replicas are nested",
        "native_pairing": "native values are repeated only to align raw downstream pairs, never counted as extra upstream seeds",
        "interpretation": "R0 vs R3 compares readout architectures, not information loss; use matched graph ablations for auxiliary attribution"})
    markdown = directory / "summary.md"
    lines = [f"# QM9 {context.config['tasks']['main']} 冻结读出结果", "",
             "增量 = MAE(第一个方法) − MAE(第二个方法)；正值表示第二个方法更好。", "",
             "| 方法 | MAE 均值 | 上游种子数 | 上游样本 SD | 运行数 |", "|---|---:|---:|---:|---:|"]
    for row in methods:
        sd = f"{row['sample_sd']:.6f}" if row["sample_sd"] is not None else "未估计"
        lines.append(f"| {row['method']} | {row['mean']:.6f} | {row['n_upstream_seeds']} | {sd} | {row['n_runs']} |")
    lines.extend(["", "## 配对增量", ""])
    for name, value in deltas.items():
        lines.append(f"- `{name}`（{contrasts[name][0]} − {contrasts[name][1]}）：{value['mean']:.6f}；第二个方法改善 {value['n_improved_upstream_seeds']}/{value['n_upstream_seeds']} 个上游种子。")
    lines.extend(["", f"本结果包含 {len(upstream_seeds)} 个上游初始化种子。下游 seed 先在上游 seed 内平均，不能把 Cartesian replicas 当作独立上游重复。",
                  "样本 SD/标准误只覆盖固定划分下的初始化变化，不覆盖分子抽样、数据划分或调参不确定性。",
                  "R0/R3/R4 架构、输入与容量不同；辅助归因优先看同容量的连接、电荷、键级和 shuffle 消融。", ""])
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return [summary, markdown, write_csv(directory / "methods.csv", methods),
            write_csv(directory / "paired_deltas.csv", paired),
            write_csv(directory / "per_upstream_deltas.csv", per_upstream)]
