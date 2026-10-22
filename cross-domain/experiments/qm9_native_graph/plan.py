"""Pure plan enumeration. Importing this file does not load torch or create files."""

from dataclasses import asdict, dataclass

SIZES = {"a_train": 56000, "a_val": 5000, "b_train": 14000, "b_val": 5000, "y_test": 20000}
BUDGETS = (2000, 7000, 14000)


@dataclass(frozen=True)
class Recipe:
    name: str
    bond_mode: str
    charge: bool = False
    hidden: bool = False
    multi_task_only: bool = True


RECIPES = (
    Recipe("geometry", "none", multi_task_only=False),
    Recipe("bond", "typed"),
    Recipe("bond_charge", "typed", charge=True),
    Recipe("uniform", "uniform", charge=True),
    Recipe("existence", "existence", charge=True),
    Recipe("shuffle", "shuffle", charge=True),
    Recipe("aux_hidden", "none", hidden=True),
)
RECIPE_BY_NAME = {recipe.name: recipe for recipe in RECIPES}


def applicable_recipes(config, variant):
    names = config["refiner"]["recipes"]
    if len(set(names)) != len(names) or set(names) - set(RECIPE_BY_NAME):
        raise ValueError("unknown or duplicate native graph recipe")
    return [name for name in names if variant == "multi_task" or not RECIPE_BY_NAME[name].multi_task_only]


def validate_config(config):
    if config["data"]["sizes"] != SIZES or config["data"].get("shared_validation", False):
        raise ValueError("fixed, independent A_train/A_val/B_pool/B_val/Y allocation is required")
    if tuple(config["nested_b"]["budgets"]) != BUDGETS:
        raise ValueError("B budgets must be the nested 2k/7k/14k prefixes")
    if set(config["upstream"]["variants"]) != {"single_task", "multi_task"}:
        raise ValueError("both ST and MT upstreams are required")
    if not config["refiner"].get("include_initial_checkpoint"):
        raise ValueError("epoch 0 must remain selectable on B_val")
    for variant in config["upstream"]["variants"]:
        applicable_recipes(config, variant)


def all_configs(config):
    """Auditable readout matrix; B is deliberately absent from upstream keys."""
    validate_config(config)
    return [{"variant": variant, "upstream_seed": us, "b_train_size": budget,
             "recipe": recipe, "downstream_seed": ds}
            for variant in config["upstream"]["variants"]
            for us in config["upstream"]["seeds"]
            for budget in config["nested_b"]["budgets"]
            for recipe in applicable_recipes(config, variant)
            for ds in config["refiner"]["seeds"]]


def build_plan(config):
    configs = all_configs(config)
    upstream = [{"variant": v, "upstream_seed": u}
                for v in config["upstream"]["variants"] for u in config["upstream"]["seeds"]]
    return {"status": "framework_not_tested", "sizes": SIZES, "budgets": list(BUDGETS),
            "recipes": [asdict(recipe) for recipe in RECIPES], "upstream": upstream,
            "cache": upstream, "all_configs": configs,
            "scheduler": "standalone plan only; generic pipeline units do not encode B budgets"}
