"""Reuse legacy QM9 preparation in a new namespace; lock one nested B pool."""

import numpy as np

from experiments.qm9 import data as legacy
from experiments.qm9_native_graph.identity import bind_sources, verify_sources
from experiments.qm9_native_graph.plan import validate_config
from pipeline.io import read_json, sha256_file, write_json


def applicable_recipes(context, variant):
    from experiments.qm9_native_graph.plan import applicable_recipes as recipes
    return recipes(context.config, variant)


def download(context):
    validate_config(context.config)
    binding = bind_sources(context)
    return [binding, *legacy.download(context)]


def prepare(context):
    validate_config(context.config)
    binding = bind_sources(context)
    artifacts = legacy.prepare(context)
    arrays = legacy.load_arrays(context)
    pool = legacy.split_indices(context, "b_train")
    # Canonical ID sort before a single fixed permutation makes the order explicit.
    sorted_pool = pool[np.argsort(arrays["ids"][pool], kind="stable")]
    order = np.random.default_rng(context.config["nested_b"]["order_seed"]).permutation(len(pool))
    positions = sorted_pool[order]
    pool_position = {int(index): i for i, index in enumerate(pool)}
    rows = [pool_position[int(index)] for index in positions]
    path = context.output_dir / "data" / "nested_b.json"
    write_json(path, {"identity": context.identity, "order_seed": context.config["nested_b"]["order_seed"],
        "ordering": "sort B_pool by molecule id, then one seeded permutation; budgets are prefixes",
        "legacy_source_sha256": verify_sources(context),
        "dataset_sha256": sha256_file(legacy.processed_dir(context) / "dataset.npz"),
        "splits_sha256": sha256_file(legacy.processed_dir(context) / "splits.npz"),
        "pool_ids": arrays["ids"][pool].tolist(), "ordered_pool_rows": rows,
        "ordered_ids": arrays["ids"][positions].tolist(),
        "budgets": context.config["nested_b"]["budgets"]})
    return [binding, *artifacts, path]


def nested_rows(context, budget, cache_ids):
    verify_sources(context)
    manifest = read_json(context.output_dir / "data" / "nested_b.json")
    expected = {"identity": context.identity,
                "legacy_source_sha256": verify_sources(context),
                "dataset_sha256": sha256_file(legacy.processed_dir(context) / "dataset.npz"),
                "splits_sha256": sha256_file(legacy.processed_dir(context) / "splits.npz")}
    if any(manifest.get(k) != v for k, v in expected.items()) or manifest["pool_ids"] != cache_ids.tolist():
        raise ValueError("nested B pool identity/order mismatch")
    if budget not in manifest["budgets"]:
        raise ValueError("unknown B budget")
    rows = np.asarray(manifest["ordered_pool_rows"], dtype=np.int64)
    if sorted(rows.tolist()) != list(range(len(cache_ids))):
        raise ValueError("B pool ordering is not a full permutation")
    if cache_ids[rows].tolist() != manifest["ordered_ids"]:
        raise ValueError("nested B molecule identity mismatch")
    return rows[:budget]
