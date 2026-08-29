"""Shared HDF5 and atomic-cache helpers used by the Parallel pipeline."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence

import h5py
import numpy as np


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _field_names(obj: h5py.Group | h5py.Dataset) -> set[str]:
    return set(obj.keys()) if isinstance(obj, h5py.Group) else set(obj.dtype.names or ())


def _require_fields(obj: h5py.Group | h5py.Dataset,
                    names: Sequence[str], object_name: str) -> None:
    missing = set(names) - _field_names(obj)
    if missing:
        raise KeyError(f"{object_name} is missing field(s): {sorted(missing)}")


def _row_count(obj: h5py.Group | h5py.Dataset, field: str) -> int:
    return len(obj[field]) if isinstance(obj, h5py.Group) else len(obj)


def _read_fields(obj: h5py.Group | h5py.Dataset,
                 names: Sequence[str], start: int, stop: int) -> dict[str, np.ndarray]:
    """Read several fields over one contiguous interval for group or compound HDF5."""
    names = _unique(names)
    if isinstance(obj, h5py.Group):
        return {name: obj[name][start:stop] for name in names}
    block = obj.fields(names)[start:stop]
    return {name: block[name] for name in names}


def _normalise_indices(indices, n_rows: int) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must be a one-dimensional integer array")
    if indices.size and (indices[0] < 0 or indices[-1] >= n_rows):
        raise IndexError(f"indices must lie in [0, {n_rows})")
    if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
        raise ValueError("indices must be strictly increasing")
    return indices


def _block_slices(indices: np.ndarray, n_rows: int, chunk_rows: int,
                  block_rows: int | None):
    """Yield selected-index positions and chunk-aligned source intervals."""
    if not indices.size:
        return
    if block_rows is None:
        block_rows = chunk_rows * 16
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    block_rows = max(chunk_rows, (block_rows // chunk_rows) * chunk_rows)
    position = 0
    while position < len(indices):
        logical_stop = (int(indices[position]) // block_rows + 1) * block_rows
        end_position = int(np.searchsorted(indices, logical_stop, side="left"))
        first, last = int(indices[position]), int(indices[end_position - 1])
        start = first // chunk_rows * chunk_rows
        stop = min(((last + chunk_rows) // chunk_rows) * chunk_rows, n_rows)
        yield position, end_position, start, stop
        position = end_position


def _save_cache_atomic(path: str, arrays: dict[str, np.ndarray]) -> None:
    """Atomically publish a NumPy archive after a complete write."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".parallel_", suffix=".npz", dir=directory, delete=False)
    temporary = handle.name
    handle.close()
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
