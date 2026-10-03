"""Paired MAE deltas for the gap experiment."""

from collections import defaultdict
import numpy as np

from evaluate.qm9 import write_csv
from pipeline.io import read_json, write_json


def summarize_values(values):
    values = np.array(list(values), dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_upstream_seeds": len(values)}


def analyze(context):
    metrics = read_json(context.output_dir / "evaluation" / "metrics.json")
    if metrics["identity"] != context.identity:
        raise ValueError("evaluation identity mismatch")
    rows = metrics["main"]
    groups, lookup = defaultdict(lambda: defaultdict(list)), {}
    for row in rows:
        groups[row["method"]][row["upstream_seed"]].append(row["mae"])
        lookup[(row["method"], row["upstream_seed"], row["downstream_seed"])] = row["mae"]
    methods = [{"method": method, **summarize_values([np.mean(values) for values in upstream.values()])}
               for method, upstream in groups.items()]
    contrasts = {
        "pool_to_local_gain_mae": ("MT-R0", "MT-R3"),
        "predicted_local_gain_mae": ("MT-R3", "MT-R4"),
        "auxiliary_readout_gain_mae": ("MT-R0", "MT-R2"),
        "auxiliary_supervision_gain_mae": ("ST-R0", "MT-R0"),
    }
    paired = []
    for upstream_seed in context.config["upstream"]["seeds"]:
        for downstream_seed in context.config["refiner"]["seeds"]:
            record = {"upstream_seed": upstream_seed, "downstream_seed": downstream_seed}
            for name, (method_a, method_b) in contrasts.items():
                key_a = (method_a, upstream_seed, None) if method_a.endswith("native") else (method_a, upstream_seed, downstream_seed)
                key_b = (method_b, upstream_seed, None) if method_b.endswith("native") else (method_b, upstream_seed, downstream_seed)
                record[name] = lookup[key_a] - lookup[key_b] if key_a in lookup and key_b in lookup else None
            paired.append(record)
    deltas = {}
    for name in contrasts:
        per_upstream = defaultdict(list)
        for row in paired:
            if row[name] is not None:
                per_upstream[row["upstream_seed"]].append(row[name])
        if per_upstream:
            deltas[name] = summarize_values([np.mean(values) for values in per_upstream.values()])
    directory = context.output_dir / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / "summary.json"
    write_json(summary, {"identity": context.identity, "methods": methods, "paired_deltas": deltas,
        "contrasts": {name: list(pair) for name, pair in contrasts.items()},
        "aggregation": "average downstream seeds within upstream seed, then mean/sample SD across upstream seeds",
        "positive_delta": "MAE reduction in physical units for the first minus the second method",
        "interpretation": "engineering smoke test; R0(g) vs R3(H) measures pooling loss, R3 vs R4 the incremental value of predicted local structure"})
    markdown = directory / "summary.md"
    lines = [f"# QM9 {context.config['tasks']['main']} 本地闭环结果", "",
             "MAE 单位见 metrics；正值表示第一个方法 MAE 更低。", "",
             "| 方法 | MAE 均值 | 上游种子数 | 上游样本 SD |", "|---|---:|---:|---:|"]
    for row in methods:
        sd = f"{row['sample_sd']:.6f}" if row["sample_sd"] is not None else "未估计"
        lines.append(f"| {row['method']} | {row['mean']:.6f} | {row['n_upstream_seeds']} | {sd} |")
    lines.extend(["", "## 配对增量", ""])
    for name, value in deltas.items():
        lines.append(f"- `{name}`（{contrasts[name][0]} − {contrasts[name][1]}）：{value['mean']:.6f}")
    lines.extend(["", "这是单种子工程测试；不据此判断统计显著性。R0/R3/R4 需在容量匹配与输入消融下解释。", ""])
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return [summary, markdown, write_csv(directory / "methods.csv", methods), write_csv(directory / "paired_deltas.csv", paired)]
