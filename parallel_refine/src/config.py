"""Configuration and deterministic seed-to-output mapping for Parallel Refine."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_SPLITS = ("a_train", "a_val", "b_train", "b_val", "y_test")
COMPONENT_CONFIG_VERSION = "parallel_refine_component_refs_v3"
COMPONENT_KEYS = {
    "data": {"data"},
    "parallel": {"parallel"},
    "refiner": {"feature_cache", "refiners"},
}
FEATURE_RECIPES = {
    "F1_embed": ("embedding",),
    "F2_jet_aux": ("jet_probability", "aux"),
    "F3_embed_aux": ("embedding", "aux"),
    "F4_all": ("jet_probability", "embedding", "aux"),
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _component_path(experiment_path: Path, reference: str) -> Path:
    path = Path(reference)
    if not path.is_absolute():
        path = experiment_path.parent / path
    return path.resolve()


def _load_component(experiment_path: Path, kind: str, reference: Any):
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"components.{kind} must be a non-empty path")
    path = _component_path(experiment_path, reference)
    if not path.is_file():
        raise FileNotFoundError(f"missing {kind} component config: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != COMPONENT_KEYS[kind]:
        raise ValueError(
            f"{kind} component must contain exactly "
            f"{sorted(COMPONENT_KEYS[kind])}, got "
            f"{sorted(values) if isinstance(values, dict) else type(values).__name__}")
    return values, {
        "path": reference,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _resolve_experiment(source: Path, raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw.get("experiment"), dict):
        raise ValueError("experiment must be an object")
    components = raw.get("components")
    if not isinstance(components, dict) or set(components) != set(COMPONENT_KEYS):
        raise ValueError(
            "experiment components must contain exactly data, parallel, and refiner")
    values = {
        "config_kind": "experiment",
        "experiment": copy.deepcopy(raw.get("experiment")),
        "composition": {
            "version": COMPONENT_CONFIG_VERSION,
            "experiment_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "sources": {},
        },
    }
    for kind, reference in components.items():
        component, provenance = _load_component(source, kind, reference)
        values.update(component)
        values["composition"]["sources"][kind] = provenance

    overrides = raw.get("overrides", {})
    allowed_overrides = {"data", "parallel", "feature_cache", "refiners"}
    if not isinstance(overrides, dict) or set(overrides) - allowed_overrides:
        raise ValueError(
            f"experiment overrides may only contain {sorted(allowed_overrides)}")
    for key, overlay in overrides.items():
        if not isinstance(overlay, dict):
            raise ValueError(f"overrides.{key} must be an object")
        values[key] = _deep_merge(values[key], overlay)
    return values


@dataclass(frozen=True)
class SeedRun:
    seed: int
    output_name: str


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    values: dict[str, Any]
    seeds: tuple[SeedRun, ...]

    @property
    def study_name(self) -> str:
        return str(self.values["experiment"]["name"])

    @property
    def experiment_name(self) -> str:
        return self.study_name

    @property
    def output_directory(self) -> Path:
        experiment = self.values["experiment"]
        return Path(experiment["output_root"]) / experiment["name"]

    @property
    def source_sha256(self) -> str:
        canonical = json.dumps(
            self.values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def experiment_file_sha256(self) -> str:
        return str(self.values["composition"]["experiment_file_sha256"])

    @property
    def component_sources(self) -> dict[str, Any]:
        return copy.deepcopy(self.values["composition"]["sources"])

    @property
    def data(self) -> dict[str, Any]:
        return self.values["data"]

    @property
    def parallel(self) -> dict[str, Any]:
        return self.values["parallel"]

    @property
    def cache(self) -> dict[str, Any]:
        return self.values["feature_cache"]

    @property
    def refiners(self) -> dict[str, Any]:
        return self.values["refiners"]

    def selected_seeds(self, requested: Iterable[int] | None = None) -> tuple[SeedRun, ...]:
        if requested is None:
            return self.seeds
        wanted = {int(value) for value in requested}
        selected = tuple(run for run in self.seeds if run.seed in wanted)
        missing = wanted - {run.seed for run in selected}
        if missing:
            raise ValueError(f"unknown configured seed(s): {sorted(missing)}")
        return selected

    def parallel_directory(self, run: SeedRun) -> Path:
        return self.output_directory / "parallel" / run.output_name

    def checkpoint(self, run: SeedRun) -> Path:
        return self.parallel_directory(run) / self.parallel.get(
            "checkpoint", "best_jet.pt")

    def refiner_directory(self, run: SeedRun, recipe: str, model: str) -> Path:
        return self.output_directory / "refiners" / run.output_name / recipe / model


def _require_positive_int(mapping: dict[str, Any], key: str) -> None:
    if not isinstance(mapping.get(key), int) or mapping[key] <= 0:
        raise ValueError(f"{key} must be a positive integer")


def load_study_config(path: str | Path) -> StudyConfig:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("runtime experiment config must be a JSON object")
    if "base_config" in raw:
        raise ValueError(
            "runtime experiment configs use components, not base_config")
    if raw.get("config_kind") != "experiment":
        raise ValueError(
            "runtime --config must be a config_kind='experiment' file")
    values = _resolve_experiment(source, raw)
    for key in ("experiment", "data", "parallel", "feature_cache", "refiners"):
        if key not in values:
            raise ValueError(f"missing required study config key: {key}")
    experiment = values["experiment"]
    name = experiment.get("name")
    if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}):
        raise ValueError("experiment.name must be a safe directory name")
    if not isinstance(experiment.get("output_root"), str) or not experiment[
            "output_root"]:
        raise ValueError("experiment.output_root must be a non-empty path")

    sizes = values["data"].get("sizes", {})
    for split in REQUIRED_SPLITS:
        _require_positive_int(sizes, split)
    _require_positive_int(values["data"], "data_seed")
    for key in ("train_file", "split_dir", "processed_cache_dir"):
        if not isinstance(values["data"].get(key), str) or not values["data"][key]:
            raise ValueError(f"data.{key} must be a non-empty path")
    _require_positive_int(values["data"], "top_k")

    raw_seeds = values["parallel"].get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("parallel.seeds must be a non-empty list")
    if values["parallel"].get("training", {}).get("gpu_ids", [-1]) != [-1]:
        raise ValueError(
            "parallel.training.gpu_ids must remain [-1]; select physical GPUs "
            "in the run script")
    runs = []
    for item in raw_seeds:
        if not isinstance(item, dict):
            raise ValueError("each parallel seed must define seed and output_name")
        seed = item.get("seed")
        output_name = item.get("output_name")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("parallel seed must be a non-negative integer")
        if not isinstance(output_name, str) or not output_name.strip():
            raise ValueError("parallel output_name must be a non-empty string")
        normalized_name = output_name.strip("/")
        if (
                not normalized_name
                or normalized_name == "."
                or Path(output_name).is_absolute()
                or ".." in Path(output_name).parts):
            raise ValueError("parallel output_name must be a safe relative directory")
        if "gpu_ids" in item:
            raise ValueError(
                "parallel seed entries cannot define gpu_ids; select physical "
                "GPUs in the run script")
        runs.append(SeedRun(seed, normalized_name))
    if len({run.seed for run in runs}) != len(runs):
        raise ValueError("parallel seeds must be unique")
    if len({run.output_name for run in runs}) != len(runs):
        raise ValueError("parallel output_name values must be unique")
    checkpoint = values["parallel"].get("checkpoint", "best_jet.pt")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("parallel.checkpoint must be a safe relative path")
    checkpoint_path = Path(checkpoint)
    if (
            checkpoint_path.is_absolute()
            or ".." in checkpoint_path.parts):
        raise ValueError("parallel.checkpoint must be a safe relative path")
    model = values["parallel"].get("model", {})
    for key in ("d_model", "n_heads", "n_layers", "d_ffn"):
        if key in model:
            _require_positive_int(model, key)
    if (
            "d_model" in model
            and "n_heads" in model
            and model["d_model"] % model["n_heads"]):
        raise ValueError("parallel.model.d_model must be divisible by n_heads")
    if "dropout" in model and (
            not isinstance(model["dropout"], (int, float))
            or not 0 <= model["dropout"] < 1):
        raise ValueError("parallel.model.dropout must be in [0, 1)")
    training = values["parallel"].get("training", {})
    for key in ("batch_size", "epochs", "checkpoint_interval"):
        _require_positive_int(training, key)
    if (
            not isinstance(training.get("lr"), (int, float))
            or training["lr"] <= 0):
        raise ValueError("parallel.training.lr must be positive")
    if (
            not isinstance(training.get("weight_decay"), (int, float))
            or training["weight_decay"] < 0):
        raise ValueError("parallel.training.weight_decay must be non-negative")
    for key, value in values["parallel"].get("loss_weights", {}).items():
        if key not in {"jet", "origin", "pair"}:
            raise ValueError(f"unknown parallel loss weight: {key}")
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"parallel.loss_weights.{key} must be non-negative")
    dtype = values["feature_cache"].get("dtype", "float32")
    if dtype not in {"float16", "float32"}:
        raise ValueError("feature_cache.dtype must be float16 or float32")
    cache_splits = tuple(values["feature_cache"].get(
        "splits", ("b_train", "b_val", "y_test")))
    if not cache_splits or set(cache_splits) - set(REQUIRED_SPLITS):
        raise ValueError("feature_cache.splits contains an unknown split")
    required_downstream = {"b_train", "b_val", "y_test"}
    if not required_downstream.issubset(cache_splits):
        raise ValueError(
            "feature_cache.splits must include b_train, b_val, and y_test")
    if not isinstance(values["feature_cache"].get("root"), str) or not values[
            "feature_cache"]["root"]:
        raise ValueError("feature_cache.root must be a non-empty path")
    _require_positive_int(values["feature_cache"], "batch_size")

    recipes = values["refiners"].get("recipes", [])
    if not recipes or set(recipes) - set(FEATURE_RECIPES):
        raise ValueError(f"refiners.recipes must use {sorted(FEATURE_RECIPES)}")
    if values["refiners"].get("seed_pairing", "same") != "same":
        raise ValueError("only same Parallel/DNN/BDT seed pairing is currently supported")
    dnn = values["refiners"].get("dnn", {})
    if dnn.get("gpu_ids", [-1]) != [-1]:
        raise ValueError(
            "refiners.dnn.gpu_ids must remain [-1]; select physical GPUs in "
            "the run script")
    if not isinstance(dnn.get("hidden_dims"), list) or not dnn["hidden_dims"]:
        raise ValueError("refiners.dnn.hidden_dims must be a non-empty list")
    if any(not isinstance(width, int) or width <= 0 for width in dnn["hidden_dims"]):
        raise ValueError("all refiners.dnn.hidden_dims must be positive integers")
    for key in ("batch_size", "epochs", "early_stopping_patience"):
        _require_positive_int(dnn, key)
    for key in ("learning_rate", "dropout"):
        if not isinstance(dnn.get(key), (int, float)) or dnn[key] < 0:
            raise ValueError(f"refiners.dnn.{key} must be non-negative")
    bdt = values["refiners"].get("bdt", {})
    for key in (
            "n_estimators", "max_depth", "max_bin",
            "early_stopping_rounds", "n_jobs"):
        _require_positive_int(bdt, key)
    if not isinstance(bdt.get("learning_rate"), (int, float)) or bdt["learning_rate"] <= 0:
        raise ValueError("refiners.bdt.learning_rate must be positive")
    for key in ("min_child_weight", "gamma", "reg_alpha", "reg_lambda"):
        if not isinstance(bdt.get(key), (int, float)) or bdt[key] < 0:
            raise ValueError(f"refiners.bdt.{key} must be non-negative")
    for key in ("subsample", "colsample_bytree"):
        if (
                not isinstance(bdt.get(key), (int, float))
                or not 0 < bdt[key] <= 1):
            raise ValueError(f"refiners.bdt.{key} must be in (0, 1]")
    if bdt.get("tree_method") != "hist":
        raise ValueError("refiners.bdt.tree_method must be 'hist'")
    if not isinstance(bdt.get("device"), str) or not bdt["device"]:
        raise ValueError("refiners.bdt.device must be a non-empty string")

    return StudyConfig(source, values, tuple(runs))


def parallel_values(study: StudyConfig, run: SeedRun, *, stage: str) -> dict[str, Any]:
    """Return the fully resolved runtime config for one Parallel seed.

    Project-wide defaults come from the production ``src`` package.  The
    A/B/Y split and output fields below belong only to this experiment and are
    consumed by :class:`ParallelRuntimeConfig`.
    """
    from src.config import _DEFAULTS

    data = study.data
    parallel = study.parallel
    values = copy.deepcopy(_DEFAULTS)
    values.update({
        "runtime_config_version": "parallel_refine_runtime_v1",
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "experiment_name": run.output_name,
        "stage": stage,
        "model_type": "parallel_origin_vertex_jet",
        "use_pair_target": True,
        "seed": run.seed,
        "data_seed": data["data_seed"],
        "train_file": data["train_file"],
        "split_dir": data["split_dir"],
        "cache_dir": data["processed_cache_dir"],
        "feature_cache_dir": study.cache["root"],
        "feature_cache_dtype": study.cache.get("dtype", "float32"),
        "output_dir": str(study.output_directory / "parallel"),
        "top_k": data.get("top_k", _DEFAULTS["top_k"]),
        "num_workers": data.get("num_workers", _DEFAULTS["num_workers"]),
        **data["sizes"],
    })
    values.update(parallel.get("model", {}))
    values.update(parallel.get("training", {}))
    loss_weights = parallel.get("loss_weights", {})
    values.update({
        "lambda_jet": loss_weights.get("jet", 1.0),
        "lambda_origin": loss_weights.get("origin", 1.0),
        "lambda_pair": loss_weights.get("pair", 1.0),
    })
    return values


class ParallelRuntimeConfig:
    """Small typed-by-convention container for the resolved experiment config.

    The production Parallel model only requires the standard ``src.config``
    attributes.  The additional attributes expose A/B/Y split locations and
    sizes without teaching the production Config class about this experiment.
    """

    def __init__(self, values: dict[str, Any]):
        self._raw = copy.deepcopy(values)
        for key, value in self._raw.items():
            setattr(self, key, value)
        self.flavour_to_label = {
            int(key): int(value)
            for key, value in self._raw["flavour_to_label"].items()
        }
        self.track_fields = list(self._raw["track_fields"])
        self.jet_class_names = list(self._raw["jet_class_names"])
        self.origin_class_names = list(self._raw["origin_class_names"])
        self.n_feats = len(self.track_fields)


def active_parallel_config(
        study: StudyConfig, run: SeedRun, *, stage: str = "parallel"
) -> ParallelRuntimeConfig:
    return ParallelRuntimeConfig(parallel_values(study, run, stage=stage))


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def materialize_parallel_config(
        study: StudyConfig, run: SeedRun, *, stage: str) -> Path:
    directory = study.output_directory / "configs" / "resolved" / stage
    path = directory / f"{run.output_name}.json"
    write_json_atomic(path, parallel_values(study, run, stage=stage))
    return path
