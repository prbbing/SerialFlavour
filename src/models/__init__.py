"""
Model registry and factory.

Each model type is registered with a builder function that takes a Config
object and returns an nn.Module.  Adding a new architecture only requires
a new module file + one line in _MODEL_REGISTRY.
"""
from .staged_origin_vertex_jet import (
    StagedOriginVertexJetTransformer, build_staged_origin_vertex_jet)
from .staged_origin_vertex_jet_fix_refine import (
    StagedOriginVertexJetTransformer as FixRefineStagedOriginVertexJetTransformer,
    build_staged_origin_vertex_jet_fix_refine)
from .staged_origin_vertex_jet_fix_dz import (
    StagedOriginVertexJetTransformerFixDz,
    build_staged_origin_vertex_jet_fix_dz)
from .staged_origin_vertex_jet_fix_refine_fix_dz import (
    StagedOriginVertexJetTransformerFixRefineFixDz,
    build_staged_origin_vertex_jet_fix_refine_fix_dz)
from .staged_origin_vertex_jet_no_refine_fix_dz import (
    StagedOriginVertexJetTransformerNoRefineFixDz,
    build_staged_origin_vertex_jet_no_refine_fix_dz)
from .parallel_origin_vertex_jet import (
    ParallelOriginVertexJetTransformer, build_parallel_origin_vertex_jet)

# Map model_type string → builder(config) → nn.Module
_MODEL_REGISTRY = {
    "staged_origin_vertex_jet":    build_staged_origin_vertex_jet,
    "staged_origin_vertex_jet_fix_refine": build_staged_origin_vertex_jet_fix_refine,
    "staged_origin_vertex_jet_fix_dz": build_staged_origin_vertex_jet_fix_dz,
    "staged_origin_vertex_jet_fix_refine_fix_dz": build_staged_origin_vertex_jet_fix_refine_fix_dz,
    "staged_origin_vertex_jet_no_refine_fix_dz": build_staged_origin_vertex_jet_no_refine_fix_dz,
    "parallel_origin_vertex_jet":  build_parallel_origin_vertex_jet,
}


def get_builder(model_type: str):
    """Look up the builder callable for a named model type."""
    if model_type not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[model_type]


def build_model(config):
    """Return an instantiated model (on CPU) from the given Config."""
    builder = get_builder(config.model_type)
    return builder(config)
