"""Explicit fingerprints for the legacy adapter outside Context.code_hash."""

from pipeline.io import read_json, sha256_file, write_json

LEGACY_FILES = ("__init__.py", "data.py", "model.py", "training.py", "refine.py", "evaluate.py", "analysis.py")


def source_hashes(context):
    root = context.domain_root / "experiments" / "qm9"
    return {name: sha256_file(root / name) for name in LEGACY_FILES}


def verify_sources(context):
    actual = source_hashes(context)
    if context.config.get("legacy_source_sha256") != actual:
        raise ValueError("legacy QM9 adapter source changed; review it and create a new experiment identity")
    if context.config["dataset"] != "qm9_native_graph" or context.config["tasks"] != {
            "main": "gap", "auxiliary": [], "local": ["charge", "bond"]}:
        raise ValueError("this package requires its own dataset identity and gap/charge/bond tasks")
    return actual


def bind_sources(context):
    hashes = verify_sources(context)
    path = context.output_dir / "adapter_identity.json"
    value = {"identity": context.identity, "legacy_source_sha256": hashes,
             "reuse_policy": "new-namespace only; historical artifacts and markers are never imported"}
    if path.exists() and read_json(path) != value:
        raise ValueError("adapter source binding differs from the current run")
    write_json(path, value)
    return path
