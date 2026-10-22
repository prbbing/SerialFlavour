"""Strict complete-matrix aggregation: downstream means, then upstream SD."""
import csv
import statistics
from pipeline.io import read_json, write_json, sha256_file
from experiments.cv_pascal_context.data import applicable_recipes, data_identity
from experiments.cv_pascal_context.evaluate import expected_keys
from experiments.cv_pascal_context.source import verify_source


def summarize(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values)>1 else None, "n_upstream": len(values), "upstream_values": values}


def analyze(context):
    path = context.output_dir / "evaluation.json"
    evaluation = read_json(path)
    if evaluation["identity"] != context.identity or evaluation["data_identity"] != data_identity(context) or evaluation["source"] != verify_source() or not evaluation["matrix_complete"]:
        raise ValueError("stale or incomplete evaluation")
    rows = evaluation["results"]
    keyed = {(r["variant"], r["upstream_seed"], r["recipe"], r["downstream_seed"]): r for r in rows}
    if len(keyed) != len(rows) or set(keyed) != expected_keys(context):
        raise ValueError("analysis refuses missing/duplicate matrix cells")
    upstream = context.config["upstream"]["seeds"]
    downstream = context.config["refiner"]["seeds"]
    summaries, contrasts, means = [], [], {}
    for variant in context.config["upstream"]["variants"]:
        for recipe in ["native", *applicable_recipes(context, variant)]:
            values = []
            for us in upstream:
                seeds = [None] if recipe == "native" else downstream
                value = statistics.mean(keyed[(variant, us, recipe, ds)]["miou"] for ds in seeds)
                means[(variant, recipe, us)] = value
                values.append(value)
            summaries.append({"variant": variant, "recipe": recipe, "metric": "miou", **summarize(values)})
            if recipe != "native":
                native_delta = [means[(variant, recipe, us)]-means[(variant, "native", us)] for us in upstream]
                contrasts.append({"contrast": f"{variant}/{recipe} - {variant}/native", **summarize(native_delta)})
                if recipe != "embedding_matched":
                    # Match both upstream and downstream seeds before averaging.
                    paired = [[keyed[(variant, us, recipe, ds)]["miou"]-keyed[(variant, us, "embedding_matched", ds)]["miou"] for ds in downstream] for us in upstream]
                    contrasts.append({"contrast": f"{variant}/{recipe} - {variant}/embedding_matched", "downstream_paired_deltas": paired, **summarize([statistics.mean(p) for p in paired])})
    for left, right in (("multi_task", "main_only_matched"), ("multi_task", "single_task")):
        if left not in context.config["upstream"]["variants"] or right not in context.config["upstream"]["variants"]:
            continue
        for recipe in ("native", "embedding_matched"):
            contrasts.append({"contrast": f"{left}/{recipe} - {right}/{recipe}", "architecture_match": right == "main_only_matched", **summarize([means[(left, recipe, us)]-means[(right, recipe, us)] for us in upstream])})
    result = {"identity": context.identity, "data_identity": data_identity(context), "source": verify_source(), "evaluation_sha256": sha256_file(path), "summaries": summaries, "paired_contrasts": contrasts, "upstream_seed_order": upstream,
              "aggregation": f"average {len(downstream)} downstream seeds inside each upstream seed; mean and sample SD(ddof=1) over {len(upstream)} upstream seeds; native counted once per upstream", "claims": "descriptive paired differences; no causal or significance claim; author ST architecture differs from MTI; matched main-only controls auxiliary supervision within the identical graph"}
    summary_path = write_json(context.output_dir / "summary.json", result)
    table = context.output_dir / "summary.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "recipe", "metric", "mean", "sample_sd", "n_upstream"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in summaries)
    return [summary_path, table]
