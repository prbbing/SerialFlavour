"""
Model registry and factory.

Each model type is registered with a builder function that takes a Config
object and returns an nn.Module.  Adding a new architecture only requires
a new module file + one line in _MODEL_REGISTRY.
"""
from .staged_origin_vertex_jet import (
    StagedOriginVertexJetTransformer, build_staged_origin_vertex_jet)
from .staged_origin_vertex_jet_fix_refine import (
    StagedOriginVertexJetTransformerFixRefine,
    build_staged_origin_vertex_jet_fix_refine)
from .staged_origin_vertex_jet_no_refine import (
    StagedOriginVertexJetTransformerNoRefine,
    build_staged_origin_vertex_jet_no_refine)
from .parallel_origin_vertex_jet import (
    ParallelOriginVertexJetTransformer, build_parallel_origin_vertex_jet)
from .staged_origin_vertex_jet_residual_refine import (
    StagedOriginVertexJetResidualRefine,
    build_staged_origin_vertex_jet_residual_refine)
from .staged_origin_vertex_jet_track_ablation import (
    StagedOriginVertexJetTrackAblation,
    build_staged_origin_vertex_jet_track_ablation)
from .jet_only_transformer import (
    JetOnlyTransformer, build_jet_only_transformer)

# Map model_type string -> builder(config) -> nn.Module
_MODEL_REGISTRY = {
    # -- staged (multiplicative refine) --
    "staged_origin_vertex_jet":        build_staged_origin_vertex_jet,
    "staged_origin_vertex_jet_fix_dz": build_staged_origin_vertex_jet,  # compat

    # -- staged (additive refine) --
    "staged_origin_vertex_jet_fix_refine":          build_staged_origin_vertex_jet_fix_refine,
    "staged_origin_vertex_jet_fix_refine_fix_dz":   build_staged_origin_vertex_jet_fix_refine,  # compat

    # -- staged (no refine) --
    "staged_origin_vertex_jet_no_refine":           build_staged_origin_vertex_jet_no_refine,
    "staged_origin_vertex_jet_no_refine_fix_dz":    build_staged_origin_vertex_jet_no_refine,  # compat

    # -- staged (two-pass 3D WLS + small residual MLP) --
    "staged_origin_vertex_jet_residual_refine": build_staged_origin_vertex_jet_residual_refine,

    # -- track-origin ablation (learned assignment without origin path) --
    "staged_origin_vertex_jet_track_ablation": build_staged_origin_vertex_jet_track_ablation,

    # -- unified jet-only backbone ablations --
    "jet_only_transformer": build_jet_only_transformer,

    # -- parallel --
    "parallel_origin_vertex_jet":  build_parallel_origin_vertex_jet,
}

# Models listed here consume the dense (K, K) pair-supervision target during
# training or validation. Keep this capability explicit: loading that target
# for staged models costs several GiB for the production dataset.
_PAIR_TARGET_MODEL_TYPES = {
    "parallel_origin_vertex_jet",
}


def get_builder(model_type: str):
    """Look up the builder callable for a named model type."""
    if model_type not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[model_type]


def model_requires_pair_target(model_type: str) -> bool:
    """Return whether ``model_type`` consumes dense pair supervision."""
    get_builder(model_type)  # Validate the model name through the registry.
    return model_type in _PAIR_TARGET_MODEL_TYPES


def build_model(config):
    """Return an instantiated model (on CPU) from the given Config."""
    builder = get_builder(config.model_type)
    return builder(config)
