"""Generic work-unit enumeration for cluster/scheduler use."""

DATASET_UNITS = ("prepare", "evaluate", "analyze")


def enumerate_units(context):
    config = context.config
    module = context.module("data")
    applicable = getattr(module, "applicable_recipes", None)
    units = ["prepare"]
    for variant in config["upstream"]["variants"]:
        for seed in config["upstream"]["seeds"]:
            units.append(f"upstream:{variant}:{seed}")
            units.append(f"cache:{variant}:{seed}")
    for variant in config["upstream"]["variants"]:
        recipes = applicable(context, variant) if applicable else list(config["refiner"]["recipes"])
        for upstream_seed in config["upstream"]["seeds"]:
            for recipe in recipes:
                for downstream_seed in config["refiner"]["seeds"]:
                    units.append(f"refine:{variant}:{upstream_seed}:{recipe}:{downstream_seed}")
    units.extend(["evaluate", "analyze"])
    return units


def unit_filters(unit):
    """Translate a unit id into the runtime filters consumed by domain modules."""
    parts = unit.split(":")
    kind = parts[0]
    if kind == "upstream" or kind == "cache":
        return {"variant": parts[1], "seed": int(parts[2])}
    if kind == "refine":
        return {"variant": parts[1], "upstream_seed": int(parts[2]), "recipe": parts[3], "downstream_seed": int(parts[4])}
    if kind in DATASET_UNITS:
        return {}
    raise ValueError(f"unknown unit: {unit}")


def unit_kind(unit):
    return unit.split(":")[0]
