"""Reusable work-unit execution and marker paths."""

from pipeline.units import unit_kind


def marker_path(context, unit):
    log_root = context.resolve(context.config.get("log_root", "cross-domain/logs"))
    safe = unit.replace(":", "__")
    return log_root / context.config["dataset"] / context.config["experiment"] / "units" / f"{safe}.json"


def execute(context, unit):
    kind = unit_kind(unit)
    if kind == "prepare":
        artifacts = list(context.module("data").download(context))
        artifacts.extend(context.module("data").prepare(context))
    elif kind == "upstream":
        artifacts = context.module("training").train(context)
    elif kind == "cache":
        artifacts = context.module("refine").cache(context)
    elif kind == "refine":
        artifacts = context.module("refine").train(context)
    elif kind == "evaluate":
        artifacts = context.module("evaluate").evaluate(context)
    elif kind == "analyze":
        artifacts = context.module("analysis").analyze(context)
    else:
        raise ValueError(f"unknown unit: {unit}")
    return artifacts
