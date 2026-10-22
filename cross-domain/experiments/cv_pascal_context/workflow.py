"""Resource lifecycle backend. Import/plan are stdlib-only and read-only."""
import math
from pathlib import Path
from pipeline.io import read_json, write_json, sha256_file, artifacts_match
from experiments.cv_pascal_context.data import applicable_recipes, data_identity, partition_counts
from experiments.cv_pascal_context.source import verify_source


def validate(context):
    verify_source()
    if context.config["dataset"] != "cv_pascal_context":
        raise ValueError("requires cv_pascal_context config")
    settings = context.config["data"]
    distinct_counts = partition_counts(settings["counts"], settings.get("shared_validation", False), settings.get("unused_train_count", 0))
    if sum(distinct_counts.values()) != settings["expected_train"]:
        raise ValueError("unique A/B/validation/unused counts must cover official train exactly")
    variants = context.config["upstream"]["variants"]
    if not variants or len(set(variants)) != len(variants) or any(v not in ("single_task", "main_only_matched", "multi_task") for v in variants):
        raise ValueError("unknown or duplicate upstream variants")
    for section in ("upstream", "refiner"):
        seeds = context.config[section]["seeds"]
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise ValueError("full PASCAL configuration requires 5 distinct upstream x 5 downstream seeds")
    if context.config["refiner"]["recipes"] != ["embedding_matched", "aux_prediction", "aux_hidden", "aux_shuffle"]:
        raise ValueError("full PASCAL readout matrix changed")
    if context.config.get("workflow", {}).get("mode") == "seed_lifecycle":
        if context.config["cache"].get("splits") != ["b_train", "b_val"]:
            raise ValueError("resource lifecycle requires B-only cache")
        if context.config["workflow"].get("max_active_seeds", 1) != 1:
            raise ValueError("PASCAL lifecycle supports one active upstream")
    run = context.output_dir / "run_manifest.json"
    if run.exists() and read_json(run)["identity"] != context.identity:
        raise ValueError("code/config identity changed; choose a new experiment name")


def plan(context):
    validate(context)
    config = context.config
    height, width = config["data"]["image_size"]
    if height % 32 or width % 32:
        raise ValueError("author four-scale MTI propagation requires dimensions divisible by 32")
    scales = [(height//(4*2**i))*(width//(4*2**i)) for i in range(4)]
    channel_area = sum(c*a for c,a in zip((18,36,72,144),scales))
    area = height*width
    splits = config["cache"].get("splits", ["b_train","b_val","y_test"])
    images = sum(config["data"]["expected_val"] if s == "y_test" else config["data"]["counts"][s] for s in splits)
    variants = config["upstream"]["variants"]
    seeds = len(config["upstream"]["seeds"])
    per_seed = {}
    spool = {}
    for variant in variants:
        sample = 4*(2*channel_area + 21*scales[0]) + area + 16
        if variant != "single_task":
            sample += 4*(4*channel_area + 24*sum(scales))
        per_seed[variant] = math.ceil(sample*images*1.05)
        spool[variant] = math.ceil(4*24*sum(scales)*config["data"]["expected_val"]*1.05) if "aux_shuffle" in applicable_recipes(context,variant) else 0
    refiners = seeds*sum(len(applicable_recipes(context,v))*len(config["refiner"]["seeds"]) for v in variants)
    rows = seeds*len(variants)+refiners
    predictions = rows*config["data"]["expected_val"]*area if config["evaluation"]["save_predictions"] else 0
    total = seeds*sum(per_seed.values())
    resource = config.get("workflow",{}).get("mode") == "seed_lifecycle"
    release = resource and config["workflow"].get("release_cache", True)
    peak_cache = max(per_seed.values()) if release else total
    peak_spool = (max(spool.values()) if release else seeds*sum(spool.values())) if resource else 0
    return {"upstream_units":seeds*len(variants), "cache_units":seeds*len(variants), "refine_units":refiners, "y_rows":rows,
            "unit_count":3+2*seeds*len(variants)+refiners, "max_active_upstreams":1 if resource else None,
            "shared_validation":config["data"].get("shared_validation",False),
            "split_counts":config["data"]["counts"],
            "unique_official_train_images":config["data"]["expected_train"],
            "unique_active_train_images":sum(partition_counts(config["data"]["counts"],config["data"].get("shared_validation",False)).values()),
            "unused_train_images":config["data"].get("unused_train_count",0),
            "cache_splits":splits, "cache_bytes_per_upstream":per_seed, "spool_bytes_per_upstream":spool,
            "release_cache":release, "total_spool_bytes":seeds*sum(spool.values()) if resource else 0,
            "full_cache_bytes":total, "total_cache_bytes":total, "peak_cache_bytes":peak_cache,
            "peak_spool_bytes":peak_spool, "prediction_uint8_uncompressed_bytes":predictions,
            "peak_output_bytes":peak_cache+peak_spool+predictions, "planned_output_bytes":peak_cache+peak_spool+predictions,
            "cache_precision":"float32 features; uint8 target restored to long on load; no feature quantization",
            "allowance":"5% serialization overhead; durable checkpoints/logs and original inputs/weights excluded; total cache is lifetime writes, peak assumes receipt cleanup between seeds"}


def seed_units(context, variant, seed):
    return [f"upstream:{variant}:{seed}", f"cache:{variant}:{seed}",
            *[f"refine:{variant}:{seed}:{r}:{ds}" for r in applicable_recipes(context,variant) for ds in context.config["refiner"]["seeds"]]]


def unit_complete(context, unit):
    from pipeline.worker import marker_path
    path = marker_path(context,unit)
    if not path.is_file():
        return False
    try:
        marker = read_json(path)
        return marker["identity"] == context.identity and marker["status"] == "complete" and artifacts_match(marker["artifacts"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def evaluate_seed(context, variant, seed):
    from experiments.cv_pascal_context.stream_evaluate import evaluate_seed as execute
    return execute(context,variant,seed)


def seed_evaluation_path(context, variant, seed):
    return context.output_dir / "seed_evaluation" / variant / f"seed{seed}.json"


def seed_artifacts(context, variant, seed):
    directory = context.output_dir
    # Explicit stable global lineage only. Synthetic fixtures may intentionally
    # omit acquisition/runtime files; once sealed, receipt hashes detect removal.
    stable = [directory/"run_manifest.json", directory/"resolved_config.json",
              directory/"prepared"/"manifest.json", directory/"prepared"/"acquisition.json",
              directory/"prepared"/"source.json"]
    paths = [path for path in stable if path.is_file()]
    paths.extend(directory/"upstream"/variant/f"seed{seed}"/name for name in ("best.pt","history.json","training_manifest.json"))
    paths.append(directory/"cache"/variant/f"seed{seed}"/"manifest.json")
    for recipe in applicable_recipes(context,variant):
        for ds in context.config["refiner"]["seeds"]:
            paths.extend(directory/"refine"/variant/f"seed{seed}"/recipe/f"seed{ds}"/name for name in ("best.pt","history.json","history.csv","training_manifest.json"))
    evaluation = seed_evaluation_path(context,variant,seed)
    value = read_json(evaluation)
    paths.append(evaluation)
    paths.extend(Path(path) for path in value.get("prediction_artifacts",[]))
    if value.get("spool_manifest"):
        paths.append(Path(value["spool_manifest"]))
    return paths


def cleanup_roots(context, variant, seed):
    return [context.output_dir/"cache"/variant/f"seed{seed}", context.output_dir/"scratch"/variant/f"seed{seed}"]


def cleanup_records(context, variant, seed):
    cache, scratch = cleanup_roots(context,variant,seed)
    manifest = read_json(cache/"manifest.json")
    records = []
    for rows in manifest["records"].values():
        for row in rows:
            records.append({"path":str((cache/row["path"]).resolve()),"bytes":row["bytes"],"sha256":row["sha256"]})
    evaluation = read_json(seed_evaluation_path(context,variant,seed))
    if evaluation.get("spool_manifest"):
        for row in read_json(evaluation["spool_manifest"])["records"]:
            records.append({"path":row["path"],"bytes":row["bytes"],"sha256":row["sha256"]})
    return records


def aggregate(context):
    # Shared receipt validation remains the sole gate; cleaned caches are never loaded.
    from pipeline.lifecycle import valid_receipt
    from experiments.cv_pascal_context.evaluate import expected_keys
    rows = []
    for variant in context.config["upstream"]["variants"]:
        for seed in context.config["upstream"]["seeds"]:
            if not valid_receipt(context,variant,seed):
                raise ValueError("full matrix aggregation requires every valid sealed seed receipt")
            value = read_json(seed_evaluation_path(context,variant,seed))
            if value["identity"] != context.identity or value["data_identity"] != data_identity(context) or value["source"] != verify_source():
                raise ValueError("stale streamed seed evaluation")
            rows.extend(value["results"])
    keys = [(r["variant"],r["upstream_seed"],r["recipe"],r["downstream_seed"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys(context):
        raise ValueError("missing or duplicate final matrix cells")
    evaluation = write_json(context.output_dir/"evaluation.json",{"identity":context.identity,"data_identity":data_identity(context),"source":verify_source(),
           "split":"y_test","selection":"fixed full matrix; A_val/B_val selection only; all seed receipts sealed before aggregation","matrix_complete":True,"results":rows})
    from experiments.cv_pascal_context.analysis import analyze
    return [evaluation,*analyze(context)]
