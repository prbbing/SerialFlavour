"""Private workspace for transient supplementary-refiner feature caches."""

import os
from pathlib import Path

PRIVATE_CACHE_ENV = "SERIALFLAVOUR_PRIVATE_REFINER_CACHE"


def cache_workspace(root, study_name):
    root = Path(root).resolve()
    directory = root / ".supplementary-refine" / study_name
    if directory.resolve() != directory:
        raise ValueError(f"Private cache workspace must not be symlinked: {directory}")
    return directory


def feature_cache_path(root, study_name, relative):
    shared = Path(root) / relative
    if os.environ.get(PRIVATE_CACHE_ENV) != "1":
        return shared
    private = cache_workspace(root, study_name) / relative
    if private.exists():
        return private
    if (shared / "manifest.json").is_file() and not (shared / ".building").exists():
        return shared
    return private
