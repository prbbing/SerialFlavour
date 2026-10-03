"""Author-format QM9 XYZ parsing, molecule isolation, and packed graphs.

Only atomic numbers and geometry are model inputs. Mulliken charges, frequencies,
SMILES, and InChI never enter the encoder or refiner; SMILES identifies duplicates.
"""

import hashlib
from pathlib import Path
import tarfile

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.download import download_verified
from pipeline.io import read_json, sha256_file, write_json

ELEMENTS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
PROPERTY_NAMES = ("tag", "index", "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "cv")
UNITS = {"mu": "Debye", "alpha": "Bohr^3", "r2": "Bohr^2", "cv": "cal/(mol K)", "gap": "eV"}
# QM9 stores homo/lumo/gap in Hartree; report the gap in eV.
TARGET_SCALE = {"gap": 27.211386245988}
BOND_CLASSES = ("none", "single", "double", "triple", "aromatic")
BOND_TYPE_CLASS = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                   Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
SPLITS = ("a_train", "a_val", "b_train", "b_val", "y_test")


def tasks(context):
    return [context.config["tasks"]["main"], *context.config["tasks"]["auxiliary"]]


def local_tasks(context):
    return list(context.config["tasks"].get("local", []))


def applicable_recipes(context, variant):
    """Recipes that make sense for an upstream variant (used by unit enumeration)."""
    recipes = list(context.config["refiner"]["recipes"])
    if variant == "single_task":
        return [recipe for recipe in recipes if recipe in ("R0", "R3")]
    return recipes


def processed_dir(context):
    return context.data_dir / "processed" / context.config["experiment"]


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return path


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def download(context):
    artifacts, sources = [], []
    for source in context.config["data"]["sources"]:
        path = download_verified(source["url"], context.data_dir / "raw" / source["name"], source["md5"], context.config["data"]["max_file_bytes"])
        artifacts.append(path)
        sources.append({**source, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = context.data_dir / "raw" / "source_manifest.json"
    write_json(manifest, {"dataset": "QM9", "version": "original_dsgdb9nsd", "license": "CC0",
                          "license_source": "https://api.figshare.com/v2/articles/1057646",
                          "format_source": "https://ndownloader.figshare.com/files/3195392", "files": sources})
    return [*artifacts, manifest]


def parse_exclusions(text, expected_count=3054):
    # The author file has nine header lines, then index-bearing rows, then a
    # single trailing separator. Select integer rows instead of slicing so the
    # parser is robust to blank/separator lines.
    values = []
    for line in text.splitlines()[9:]:
        fields = line.split()
        if fields and fields[0].isdigit():
            values.append(int(fields[0]))
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError("QM9 exclusion list count/uniqueness mismatch")
    return set(values)


def numeric(value):
    return float(value.replace("*^", "e"))


def perceive_bonds(xyz_block):
    """Perceive bonds and bond orders from geometry, preserving XYZ atom order."""
    molecule = Chem.MolFromXYZBlock(xyz_block)
    if molecule is None:
        raise ValueError("bond perception: XYZ block could not be parsed")
    rdDetermineBonds.DetermineBonds(molecule, charge=0)
    bonds = []
    for bond in molecule.GetBonds():
        order = BOND_TYPE_CLASS.get(bond.GetBondType())
        if order is None:
            raise ValueError("bond perception: unsupported bond type")
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order))
    return bonds


def dense_pairs(count, bonds):
    """Return local unordered pair indices (2, P) and their bond class (P,)."""
    upper = np.triu_indices(count, k=1)
    pair_index = np.stack(upper).astype(np.int64)
    pair_class = np.zeros(pair_index.shape[1], dtype=np.int64)
    position = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(*upper))}
    for left, right, order in bonds:
        if left > right:
            left, right = right, left
        pair_class[position[(left, right)]] = order
    return pair_index, pair_class


def parse_xyz(text, target_names):
    lines = text.splitlines()
    count = int(lines[0])
    properties = lines[1].split()
    # The author readme says gdb9; the published archive actually uses gdb.
    if count < 1 or len(properties) != len(PROPERTY_NAMES) or properties[0] not in ("gdb", "gdb9"):
        raise ValueError("invalid QM9 XYZ property header")
    molecule_id = int(properties[1])
    values = dict(zip(PROPERTY_NAMES, properties))
    target = np.array([numeric(values[name]) * TARGET_SCALE.get(name, 1.0) for name in target_names], dtype=np.float32)
    z, pos, charge, atom_rows = [], [], [], []
    for line in lines[2:count + 2]:
        fields = line.split()
        if len(fields) != 5 or fields[0] not in ELEMENTS:
            raise ValueError("invalid QM9 atom row")
        z.append(ELEMENTS[fields[0]])
        pos.append([numeric(field) for field in fields[1:4]])
        charge.append(numeric(fields[4]))
        atom_rows.append(f"{fields[0]} {numeric(fields[1])} {numeric(fields[2])} {numeric(fields[3])}")
    pos = np.asarray(pos, dtype=np.float32)
    charge = np.asarray(charge, dtype=np.float32)
    if pos.shape != (count, 3) or not np.isfinite(pos).all() or not np.isfinite(target).all() or not np.isfinite(charge).all():
        raise ValueError("invalid/non-finite molecular arrays")
    xyz_block = f"{count}\n\n" + "\n".join(atom_rows) + "\n"
    bonds = perceive_bonds(xyz_block)
    smiles_fields = lines[count + 3].split()
    molecule = Chem.MolFromSmiles(smiles_fields[-1])
    if molecule is None:
        raise ValueError("relaxed-geometry SMILES could not be parsed")
    canonical = Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True)
    return {"id": molecule_id, "z": np.array(z, dtype=np.int64),
            "pos": pos - pos.mean(axis=0), "target": target, "smiles": canonical,
            "charge": charge, "bonds": bonds}


def distance_edges(pos, cutoff):
    distance = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    keep = (distance < cutoff) & ~np.eye(len(pos), dtype=bool)
    row, col = np.nonzero(keep)
    return np.stack([row, col]).astype(np.int64), distance[row, col].astype(np.float32)


def subset_size(settings):
    """Unique molecule count to sample, honouring optional shared validation."""
    sizes = settings["sizes"]
    if settings.get("shared_validation", False):
        for key in ("a_train", "b_train", "y_test"):
            if sizes.get(key, 0) <= 0:
                raise ValueError(f"shared validation requires a positive {key} size")
        validation = int(settings.get("validation_size", 0))
        if validation <= 0:
            raise ValueError("shared validation requires a positive validation_size")
        return sizes["a_train"] + sizes["b_train"] + validation + sizes["y_test"]
    if tuple(sizes) != SPLITS or any(sizes[key] <= 0 for key in SPLITS):
        raise ValueError("positive a_train/a_val/b_train/b_val/y_test sizes are required in that order")
    return sum(sizes.values())


def allocate_splits(size, settings, permutation):
    """Assign dataset positions to splits; shared validation reuses one block."""
    sizes = settings["sizes"]
    if not settings.get("shared_validation", False):
        splits, cursor = {}, 0
        for key in SPLITS:
            count = sizes[key]
            splits[key] = permutation[cursor:cursor + count]
            cursor += count
        return splits
    validation = int(settings["validation_size"])
    splits, cursor = {}, 0
    for key in ("a_train", "b_train"):
        count = sizes[key]
        splits[key] = permutation[cursor:cursor + count]
        cursor += count
    block = permutation[cursor:cursor + validation]
    cursor += validation
    splits["a_val"] = block
    splits["b_val"] = block
    splits["y_test"] = permutation[cursor:cursor + sizes["y_test"]]
    cursor += sizes["y_test"]
    if cursor != size:
        raise ValueError("split allocation does not cover the sampled subset")
    return splits


def prepare(context):
    settings = context.config["data"]
    size = subset_size(settings)
    names = tasks(context)
    if len(set(names)) != len(names) or any(name not in UNITS for name in names):
        raise ValueError("this QM9 smoke adapter supports distinct mu/alpha/r2/cv tasks")
    excluded = parse_exclusions((context.data_dir / "raw" / "uncharacterized.txt").read_text())
    eligible = np.array([index for index in range(1, 133886) if index not in excluded], dtype=np.int64)
    if size > len(eligible):
        raise ValueError("requested subset exceeds the eligible QM9 population")
    candidates = np.random.default_rng(settings["sampling_seed"]).permutation(eligible)[:min(len(eligible), size * settings["candidate_multiplier"])]
    priority = {int(index): rank for rank, index in enumerate(candidates)}
    records, invalid = {}, []
    archive_path = context.data_dir / "raw" / "dsgdb9nsd.xyz.tar.bz2"
    with rdBase.BlockLogs(), tarfile.open(archive_path, mode="r|bz2") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".xyz"):
                continue
            if member.size > settings["max_file_bytes"]:
                raise ValueError("an archive member exceeds the single-file limit")
            index = int(Path(member.name).stem.rsplit("_", 1)[-1])
            if index not in priority:
                continue
            try:
                record = parse_xyz(archive.extractfile(member).read().decode("utf-8"), names)
                if record["id"] != index:
                    raise ValueError("filename/property molecule ID mismatch")
                records[index] = record
            except Exception as error:
                invalid.append({"id": index, "reason": f"{type(error).__name__}: {error}"})
    selected, seen, duplicates = [], set(), []
    for index in candidates:
        record = records.get(int(index))
        if record is None:
            continue
        if record["smiles"] in seen:
            duplicates.append(record["id"])
            continue
        selected.append(record)
        seen.add(record["smiles"])
        if len(selected) == size:
            break
    if len(selected) != size:
        raise ValueError(f"only {len(selected)} valid unique candidates; increase candidate_multiplier")
    atom_offsets, edge_offsets, pair_offsets = [0], [0], [0]
    z, pos, edges, distance, charge, pair_index, pair_class = [], [], [], [], [], [], []
    for record in selected:
        local_edges, local_distance = distance_edges(record["pos"], context.config["model"]["cutoff"])
        local_pairs, local_classes = dense_pairs(len(record["z"]), record["bonds"])
        edges.append(local_edges + atom_offsets[-1])
        distance.append(local_distance)
        z.append(record["z"])
        pos.append(record["pos"])
        charge.append(record["charge"])
        pair_index.append(local_pairs)
        pair_class.append(local_classes)
        atom_offsets.append(atom_offsets[-1] + len(record["z"]))
        edge_offsets.append(edge_offsets[-1] + local_edges.shape[1])
        pair_offsets.append(pair_offsets[-1] + local_pairs.shape[1])
    directory = processed_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    dataset_path = save_arrays(directory / "dataset.npz", z=np.concatenate(z), pos=np.concatenate(pos),
                              edge_index=np.concatenate(edges, axis=1), distance=np.concatenate(distance),
                              atom_offsets=np.array(atom_offsets), edge_offsets=np.array(edge_offsets),
                              charge=np.concatenate(charge), pair_index=np.concatenate(pair_index, axis=1),
                              pair_class=np.concatenate(pair_class), pair_offsets=np.array(pair_offsets),
                              target=np.stack([record["target"] for record in selected]),
                              ids=np.array([record["id"] for record in selected]),
                              smiles=np.array([record["smiles"] for record in selected]))
    permutation = np.random.default_rng(settings["split_seed"]).permutation(size)
    splits = allocate_splits(size, settings, permutation)
    split_path = save_arrays(directory / "splits.npz", **splits)
    source = read_json(context.data_dir / "raw" / "source_manifest.json")
    manifest = context.output_dir / "data" / "preparation_manifest.json"
    write_json(manifest, {
        "identity": context.identity, "source": source, "raw_molecules": 133885,
        "official_exclusions": len(excluded), "eligible_before_subset_validation": len(eligible),
        "candidate_count": len(candidates), "invalid_candidates": invalid,
        "duplicate_candidates_before_subset_filled": duplicates, "subset_size": size,
        "grouping": "unique canonical isomeric heavy-atom SMILES of relaxed geometry",
        "sampling_seed": settings["sampling_seed"], "split_seed": settings["split_seed"],
        "shared_validation": bool(settings.get("shared_validation", False)),
        "validation_size": settings.get("validation_size"),
        "tasks": names, "units": {name: UNITS[name] for name in names}, "rdkit": rdBase.rdkitVersion,
        "dataset_path": str(dataset_path), "dataset_sha256": sha256_file(dataset_path),
        "splits_path": str(split_path), "splits_sha256": sha256_file(split_path),
        "splits": {key: {"count": len(value), "ids_sha256": array_hash(np.array([selected[int(i)]["id"] for i in value], dtype=np.int64))} for key, value in splits.items()},
        "atom_count_min": min(map(len, z)), "atom_count_max": max(map(len, z)),
        "inputs": ["atomic_number", "geometry_in_angstrom"], "pretrained": False,
        "auxiliary": {
            "local_tasks": local_tasks(context),
            "charge_unit": "elementary_charge",
            "bond_classes": list(BOND_CLASSES),
            "bond_class_counts": np.bincount(np.concatenate(pair_class), minlength=len(BOND_CLASSES)).tolist(),
            "target_scales": TARGET_SCALE,
        },
    })
    print(f"prepared {size} molecules; split sizes: {settings['sizes']}", flush=True)
    return [dataset_path, split_path, manifest]


def load_arrays(context):
    with np.load(processed_dir(context) / "dataset.npz", allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def split_indices(context, split):
    with np.load(processed_dir(context) / "splits.npz", allow_pickle=False) as splits:
        return splits[split].copy()


def gather_values(arrays, indices, offsets_key, values_key):
    """Concatenate a per-molecule variable-length field over dataset indices."""
    offsets = arrays[offsets_key]
    parts = [arrays[values_key][offsets[int(index)]:offsets[int(index) + 1]] for index in indices]
    if not parts:
        return np.empty(0, dtype=arrays[values_key].dtype)
    return np.concatenate(parts)


class Molecules(Dataset):
    def __init__(self, arrays, indices):
        self.arrays, self.indices = arrays, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        lo, hi = self.arrays["atom_offsets"][index:index + 2]
        edge_lo, edge_hi = self.arrays["edge_offsets"][index:index + 2]
        pair_lo, pair_hi = self.arrays["pair_offsets"][index:index + 2]
        return {"z": torch.from_numpy(self.arrays["z"][lo:hi]),
                "edge_index": torch.from_numpy(self.arrays["edge_index"][:, edge_lo:edge_hi] - lo),
                "distance": torch.from_numpy(self.arrays["distance"][edge_lo:edge_hi]),
                "charge": torch.from_numpy(self.arrays["charge"][lo:hi]),
                "pair_index": torch.from_numpy(self.arrays["pair_index"][:, pair_lo:pair_hi]),
                "pair_class": torch.from_numpy(self.arrays["pair_class"][pair_lo:pair_hi]),
                "target": torch.from_numpy(self.arrays["target"][index]),
                "id": int(self.arrays["ids"][index])}


def collate_molecules(items):
    offset, edges, pairs, membership, pair_membership = 0, [], [], [], []
    for index, item in enumerate(items):
        edges.append(item["edge_index"] + offset)
        pairs.append(item["pair_index"] + offset)
        membership.append(torch.full((len(item["z"]),), index, dtype=torch.long))
        pair_membership.append(torch.full((item["pair_index"].shape[1],), index, dtype=torch.long))
        offset += len(item["z"])
    return {"z": torch.cat([item["z"] for item in items]), "edge_index": torch.cat(edges, dim=1),
            "distance": torch.cat([item["distance"] for item in items]), "batch": torch.cat(membership),
            "charge": torch.cat([item["charge"] for item in items]),
            "pair_index": torch.cat(pairs, dim=1), "pair_class": torch.cat([item["pair_class"] for item in items]),
            "pair_batch": torch.cat(pair_membership),
            "target": torch.stack([item["target"] for item in items]),
            "ids": torch.tensor([item["id"] for item in items], dtype=torch.long)}


def make_loader(context, split, seed, batch_size, shuffle=False):
    return DataLoader(Molecules(load_arrays(context), split_indices(context, split)), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed),
                      collate_fn=collate_molecules, num_workers=context.config["runtime"]["num_workers"])
