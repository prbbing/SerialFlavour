"""Train/load one legacy upstream per (variant, seed), shared by all B budgets."""

from experiments.qm9 import training as legacy
from experiments.qm9_native_graph.identity import bind_sources, verify_sources
from experiments.qm9_native_graph.plan import validate_config


def train(context):
    validate_config(context.config)
    binding = bind_sources(context)
    return [binding, *legacy.train(context)]


def load_upstream(context, variant, seed, device, frozen=True):
    verify_sources(context)
    model, metadata = legacy.load_upstream(context, variant, seed, device, frozen=frozen)
    if metadata.get("model") != context.config["model"] or metadata.get("variant") != variant or metadata.get("seed") != seed:
        raise ValueError("upstream local model identity differs")
    if metadata.get("normalization_split") != "a_train" or metadata.get("selection_split") != "a_val":
        raise ValueError("upstream normalization/selection protocol mismatch")
    expected_local = ["charge", "bond"] if variant == "multi_task" else []
    if (metadata.get("property_tasks") != ["gap"] or metadata.get("local_tasks") != expected_local
            or metadata.get("units") != {"gap": "eV"}):
        raise ValueError("upstream task or physical unit identity differs")
    return model, metadata
