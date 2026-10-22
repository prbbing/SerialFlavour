"""PASCAL_MT schema, bounded acquisition, duplicate groups, and A/B/Y data."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
import urllib.request

from pipeline.io import json_hash, read_json, sha256_file, write_json
from experiments.cv_pascal_context.source import ROOT, author_module, verify_source

URL = "https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz"
SPLITS = ("a_train", "a_val", "b_train", "b_val", "y_test")
RECIPES = ("embedding_matched", "aux_prediction", "aux_hidden", "aux_shuffle")
CONTEXT_CATEGORY_LABELS = (0, 2, 23, 25, 31, 34, 45, 59, 65, 72, 98, 397, 113, 207, 258, 284, 308, 347, 368, 416, 427)


def applicable_recipes(context, variant):
    configured = list(context.config["refiner"]["recipes"])
    if any(recipe not in RECIPES for recipe in configured):
        raise ValueError("unknown PASCAL readout recipe")
    return configured if variant != "single_task" else [r for r in configured if r == "embedding_matched"]


def raw_dir(context):
    return context.resolve(context.config["data"].get("raw_root", str(context.data_dir / "PASCAL_MT")))


def processed_dir(context):
    return context.output_dir / "prepared"


def safe_extract(archive, target, max_total_bytes, max_member_bytes, max_members=100000):
    """Reject links/devices, traversal, Windows paths, bombs and existing links."""
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    total, count = 0, 0
    with tarfile.open(archive, "r:*") as stream:
        for member in stream:
            count += 1
            name = member.name
            relative = PurePosixPath(name)
            if count > max_members or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("unsafe archive member path/count")
            destination = (target / Path(*relative.parts)).resolve()
            if not destination.is_relative_to(target) or member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError("archive links/devices/traversal are forbidden")
            if member.size < 0 or member.size > max_member_bytes:
                raise ValueError("archive member exceeds configured size limit")
            total += member.size
            if total > max_total_bytes:
                raise ValueError("archive expanded size exceeds limit")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(f"refusing archive overwrite: {destination}")
                with stream.extractfile(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024*1024)
    return {"members": count, "expanded_bytes": total}


def bounded_download(url, destination, max_bytes):
    if url != URL:
        raise ValueError("dataset URL must match the pinned author source")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    total = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("xb") as output:
            length = source.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError("dataset exceeds configured local download limit; provide locally")
            while True:
                chunk = source.read(1024*1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("dataset exceeded download limit")
                output.write(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total


def download(context):
    source = verify_source()
    settings = context.config["data"]
    raw = raw_dir(context)
    acquisition = {"url": URL, "source": source, "mode": "local_extracted", "raw_root": str(raw)}
    if not (raw / "ImageSets/Context/train.txt").is_file():
        archive = context.resolve(settings["archive_path"])
        if not archive.is_file():
            if not settings.get("allow_download", False):
                raise FileNotFoundError(f"provide extracted PASCAL_MT at {raw}, or local archive {archive}; automatic large download is disabled")
            bounded_download(URL, archive, int(settings["max_archive_bytes"]))
        if archive.stat().st_size > int(settings["max_archive_bytes"]):
            raise ValueError("local archive exceeds max_archive_bytes; explicitly review its size and update config")
        digest = sha256_file(archive)
        expected = settings.get("archive_sha256")
        if expected and digest != expected:
            raise ValueError("PASCAL archive checksum mismatch")
        if not settings.get("allow_extract", False):
            raise ValueError("set data.allow_extract only after reviewing local archive source and size")
        extraction = context.resolve(settings["extract_root"])
        stats = safe_extract(archive, extraction, int(settings["max_expanded_bytes"]), int(settings["max_member_bytes"]))
        acquisition.update(mode="local_archive", archive_sha256=digest, archive_bytes=archive.stat().st_size, extraction=stats)
    if not (raw / "ImageSets/Context/train.txt").is_file():
        raise ValueError("archive root does not match configured raw_root; locate ImageSets/Context/train.txt explicitly")
    return [write_json(processed_dir(context) / "acquisition.json", acquisition),
            write_json(processed_dir(context) / "source.json", {"source": source, "source_lock": read_json(ROOT / "source_lock.json")})]


def read_ids(path):
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)) or any(not value.replace("_", "").isalnum() for value in ids):
        raise ValueError("empty, duplicate, or unsafe official image IDs")
    return ids


def schema_paths(raw, image_id):
    primary = raw / "semseg/VOC12" / f"{image_id}.png"
    if not primary.is_file():
        primary = raw / "semseg/pascal-context" / f"{image_id}.png"
    return {"image": raw / "JPEGImages" / f"{image_id}.jpg", "semseg": primary,
            "context": raw / "pascal-context/trainval" / f"{image_id}.mat",
            "human_parts": raw / "human_parts" / f"{image_id}.mat",
            "normals": raw / "normals_distill" / f"{image_id}.png",
            "sal": raw / "sal_distill" / f"{image_id}.png"}


def partition_counts(counts, shared_validation=False, unused_train_count=0):
    """Count distinct partitions and any explicitly excluded training images."""
    names = ("a_train", "a_val", "b_train", "b_val")
    if set(counts) != set(names) or any(type(counts[name]) is not int or counts[name] <= 0 for name in names):
        raise ValueError("positive integer a_train/a_val/b_train/b_val counts are required")
    if type(shared_validation) is not bool:
        raise ValueError("shared_validation must be boolean")
    if shared_validation and counts["a_val"] != counts["b_val"]:
        raise ValueError("shared validation requires equal a_val and b_val counts")
    if type(unused_train_count) is not int or unused_train_count < 0:
        raise ValueError("unused_train_count must be a nonnegative integer")
    result = {name: counts[name] for name in names if not (shared_validation and name == "b_val")}
    if unused_train_count:
        result["unused_train"] = unused_train_count
    return result


def _assign_groups(train_ids, groups, counts, rng):
    train_groups = {}
    for image_id in train_ids:
        train_groups.setdefault(groups[image_id], []).append(image_id)
    values = list(train_groups.values())
    rng.shuffle(values)
    values.sort(key=len, reverse=True)
    remaining = dict(counts)
    result = {name: [] for name in counts}
    for group in values:
        possible = [name for name in counts if remaining[name] >= len(group)]
        if not possible:
            raise ValueError("exact counts incompatible with duplicate/source groups; review split config")
        rng.shuffle(possible)
        choice = max(possible, key=lambda name: remaining[name])
        result[choice].extend(sorted(group))
        remaining[choice] -= len(group)
    if any(remaining.values()):
        raise ValueError("group split did not meet exact counts")
    return result


def grouped_split(train_ids, val_ids, groups, counts, seed, shared_validation=False, unused_train_count=0):
    if set(train_ids) & set(val_ids):
        raise ValueError("official train/val image overlap")
    if {groups[i] for i in train_ids} & {groups[i] for i in val_ids}:
        raise ValueError("duplicate/source group crosses official train and locked Y")
    distinct_counts = partition_counts(counts, shared_validation, unused_train_count)
    if sum(distinct_counts.values()) != len(train_ids):
        raise ValueError("A/B/validation/unused counts must partition the exact official train IDs")
    initial_counts = dict(distinct_counts)
    if shared_validation and unused_train_count:
        # Preserve A/B membership from the previous larger shared validation
        # pool; only split that pool into active validation and excluded IDs.
        initial_counts["a_val"] += initial_counts.pop("unused_train")
    result = _assign_groups(train_ids, groups, initial_counts, random.Random(seed))
    if shared_validation and unused_train_count:
        result.update(_assign_groups(result["a_val"], groups,
            {"a_val": counts["a_val"], "unused_train": unused_train_count}, random.Random(seed+1)))
    if shared_validation:
        result["b_val"] = list(result["a_val"])
    result["y_test"] = list(val_ids)
    return result


def prepare(context):
    import numpy as np
    from PIL import Image
    raw = raw_dir(context)
    settings = context.config["data"]
    train_ids = read_ids(raw / "ImageSets/Context/train.txt")
    val_ids = read_ids(raw / "ImageSets/Context/val.txt")
    if len(train_ids) != settings["expected_train"] or len(val_ids) != settings["expected_val"]:
        raise ValueError("official PASCAL_MT counts differ; inspect actual IDs and explicitly revise config, never silently truncate")
    mapping = read_json(context.resolve(settings["group_map"])) if settings.get("group_map") else {}
    if mapping and set(mapping) != set(train_ids + val_ids):
        raise ValueError("external source-group map must cover exactly all official IDs")
    records, parents = {}, {}
    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]
    def union(left, right):
        parents[find(left)] = find(right)
    seen_bytes, seen_pixels, seen_sources = {}, {}, {}
    duplicate_pairs = []
    for image_id in train_ids + val_ids:
        files = schema_paths(raw, image_id)
        checksums = {task: sha256_file(path) for task, path in files.items()}
        image = np.array(Image.open(files["image"]).convert("RGB"))
        pixel_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        target = np.array(Image.open(files["semseg"]))
        if not set(np.unique(target).tolist()) <= set(range(21)) | {255}:
            raise ValueError(f"{image_id}: semantic target is not author's 21-class+void schema (59-class version is forbidden)")
        find(image_id)
        for table, value, reason in ((seen_bytes, checksums["image"], "jpeg_sha256"), (seen_pixels, pixel_hash, "decoded_rgb_sha256"), (seen_sources, str(mapping.get(image_id, image_id)), "source_group")):
            if value in table:
                union(image_id, table[value])
                duplicate_pairs.append([image_id, table[value], reason])
            table[value] = image_id
        records[image_id] = {"files": {task: {"path": str(path.relative_to(raw)), "sha256": checksums[task]} for task, path in files.items()}, "rgb_shape": list(image.shape), "pixel_sha256": pixel_hash}
    groups = {image_id: find(image_id) for image_id in records}
    splits = grouped_split(train_ids, val_ids, groups, settings["counts"], settings["split_seed"], settings.get("shared_validation", False), settings.get("unused_train_count", 0))
    unused_ids = splits.pop("unused_train", [])
    manifest = {"identity": context.identity, "source": verify_source(), "dataset": "author PASCAL_MT 21 classes; VOC12 priority then context", "raw_root": str(raw), "records": records, "splits": splits, "groups": groups, "duplicate_pairs": duplicate_pairs,
                "official_split_sha256": {name: sha256_file(raw / f"ImageSets/Context/{name}.txt") for name in ("train", "val")},
                "group_map_sha256": sha256_file(context.resolve(settings["group_map"])) if mapping else None,
                "grouping_limit": "exact byte and decoded RGB duplicates plus optional provenance groups; near-duplicates without supplied groups not certified",
                "teacher_labels": {"normals": "distilled teacher predictions; valid classes matched NYU/context plus tvmonitor by author", "sal": "distilled teacher predictions thresholded >0.5 by author", "edge": "thin Laplacian of context LabelMap; derived from annotation"},
                "shared_validation": settings.get("shared_validation", False),
                "unused_train_ids": unused_ids, "unused_train_count": len(unused_ids),
                "validation_policy": "identical ordered a_val/b_val IDs for upstream/readout selection" if settings.get("shared_validation", False) else "separate a_val/b_val IDs",
                "unique_train_images": len(set().union(*(set(splits[name]) for name in ("a_train", "a_val", "b_train", "b_val")))),
                "locked_y": "official val; no selection", "auxiliary_truth_downstream": False}
    return [write_json(processed_dir(context) / "manifest.json", manifest)]


def data_manifest(context):
    manifest = read_json(processed_dir(context) / "manifest.json")
    if manifest["identity"] != context.identity or manifest["source"] != verify_source():
        raise ValueError("prepared data/source identity changed")
    shared = context.config["data"].get("shared_validation", False)
    if manifest.get("shared_validation", False) != shared:
        raise ValueError("prepared validation policy differs from config")
    if shared and manifest["splits"]["a_val"] != manifest["splits"]["b_val"]:
        raise ValueError("shared validation requires identical ordered a_val and b_val IDs")
    unused = manifest.get("unused_train_ids", [])
    if len(unused) != context.config["data"].get("unused_train_count", 0) or len(unused) != len(set(unused)):
        raise ValueError("unused training pool differs from config")
    if set(unused) & set().union(*(set(ids) for ids in manifest["splits"].values())):
        raise ValueError("unused images cannot appear in an active split")
    return manifest


def data_identity(context):
    data_manifest(context)
    return sha256_file(processed_dir(context) / "manifest.json")


class PascalSamples:
    def __init__(self, context, split, auxiliary=False, augment=False):
        if split not in SPLITS or (auxiliary and split != "a_train"):
            raise ValueError("auxiliary truth is restricted to A_train")
        self.manifest = data_manifest(context)
        self.raw = raw_dir(context)
        self.ids = self.manifest["splits"][split]
        self.auxiliary, self.augment = auxiliary, augment
        self.size = context.config["data"]["image_size"]
        self.verified = set()
        self.author = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        import numpy as np
        import torch
        import cv2
        import scipy.io as sio
        from PIL import Image
        image_id = self.ids[index]
        record = self.manifest["records"][image_id]
        needed = ("image", "semseg", "context", "human_parts", "normals", "sal") if self.auxiliary else ("image", "semseg")
        for task in needed:
            key = (image_id, task)
            if key not in self.verified:
                if sha256_file(self.raw / record["files"][task]["path"]) != record["files"][task]["sha256"]:
                    raise ValueError(f"raw data changed after prepare: {image_id}/{task}")
                self.verified.add(key)
        if self.author is None and self.auxiliary:
            cls = author_module("data.pascal_context").PASCALContext
            author = cls.__new__(cls)  # avoid author's auto-download/raw-root writes
            author.root = str(self.raw)
            author.im_ids = self.ids
            author.images = [str(self.raw / self.manifest["records"][i]["files"]["image"]["path"]) for i in self.ids]
            author.edges = [str(self.raw / self.manifest["records"][i]["files"]["context"]["path"]) for i in self.ids]
            author.parts = [str(self.raw / self.manifest["records"][i]["files"]["human_parts"]["path"]) for i in self.ids]
            author.normals = [str(self.raw / self.manifest["records"][i]["files"]["normals"]["path"]) for i in self.ids]
            author.sals = [str(self.raw / self.manifest["records"][i]["files"]["sal"]["path"]) for i in self.ids]
            author.has_human_parts = [False] * len(self.ids)
            author.human_parts_category = 15
            author.cat_part = read_json(ROOT / "vendor/author/data/db_info/pascal_part.json")
            author.cat_part["15"] = cls.HUMAN_PART[6]
            nyu = read_json(ROOT / "vendor/author/data/db_info/nyu_classes.json")
            categories = read_json(ROOT / "vendor/author/data/db_info/context_classes.json")
            author.normals_valid_classes = [categories[c] for c in nyu if c in categories and c != "unknown"] + [categories["tvmonitor"]]
            self.author = author
        sample = {"image": np.array(Image.open(self.raw / record["files"]["image"]["path"]).convert("RGB")).astype(np.float32),
                  "semseg": np.array(Image.open(self.raw / record["files"]["semseg"]["path"])).astype(np.float32)}
        if self.auxiliary:
            objects = sio.loadmat(self.author.parts[index])["anno"][0][0][1][0]
            self.author.has_human_parts[index] = any(int(obj[1].reshape(-1)[0]) == 15 and np.sum(obj[2]) > 0 for obj in objects)
            sample.update(human_parts=self.author._load_human_parts(index)[0], edge=self.author._load_edge(index), normals=self.author._load_normals_distilled(index), sal=self.author._load_sal_distilled(index))
        # Author-equivalent fixed resize and optional horizontal flip. Rotation
        # and random scaling are intentionally omitted and recorded in config.
        flip = self.augment and random.random() < 0.5
        output = {"id": image_id}
        for task, array in sample.items():
            interpolation = cv2.INTER_CUBIC if task in ("image", "normals") else cv2.INTER_NEAREST
            normal_valid = cv2.resize((np.linalg.norm(array, axis=2) > 0).astype(np.uint8), tuple(self.size[::-1]), interpolation=cv2.INTER_NEAREST).astype(bool) if task == "normals" else None
            array = cv2.resize(array, tuple(self.size[::-1]), interpolation=interpolation)
            if flip:
                array = np.flip(array, axis=1).copy()
                if task == "normals":
                    array[:, :, 0] *= -1
                    normal_valid = np.flip(normal_valid, axis=1).copy()
            if task == "image":
                tensor = torch.from_numpy(array.clip(0, 255).astype(np.uint8).copy()).permute(2, 0, 1).float()/255
                output[task] = (tensor - torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
            elif task in ("semseg", "human_parts"):
                if task == "human_parts" and not array.any():
                    array[:] = 255
                output[task] = torch.from_numpy(array.copy()).long()
            elif task == "normals":
                norm = np.linalg.norm(array, axis=2, keepdims=True)
                array = array/(norm + np.finfo(np.float32).eps)
                array[~normal_valid | (np.squeeze(norm, 2) == 0)] = 255
                output[task] = torch.from_numpy(array.astype(np.float32).copy()).permute(2, 0, 1)
            else:
                output[task] = torch.from_numpy(array.astype(np.float32).copy())[None]
        return output


def make_loader(context, split, seed, batch_size, shuffle=False, auxiliary=False):
    import torch
    from torch.utils.data import DataLoader
    workers = context.config.get("loader", {}).get("workers", 0)
    dataset = PascalSamples(context, split, auxiliary=auxiliary, augment=shuffle and split == "a_train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed), pin_memory=context.config["runtime"]["device"].startswith("cuda"), persistent_workers=workers > 0)
