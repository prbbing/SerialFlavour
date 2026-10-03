"""Real NYUv2 shard download and image-disjoint smoke-test preparation."""

from pathlib import Path
import hashlib
import numpy as np
import requests
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from pipeline.io import read_json, write_json, sha256_file

REVISION = 'b367b8b53c4dcefbcb4d9310b74976a63cc7f306'
SHARDS = {
    'train': ('train-00000-of-00008.parquet', 208564689, '2c340d96fa2d9d225e39721805fa302f93cf62d4506bc60d438f3aa766d30bd7'),
    'val': ('val-00000-of-00006.parquet', 224448496, '74366586abc14e81dc1bdd759dd655be0af9c123d8fa3112181cb6f03bd0db14'),
}


def selected_shards(context):
    """Resolve checksums at a pinned revision; smoke stays fully offline-capable."""
    if context.config['data'].get('download_scope', 'first_shards') == 'first_shards':
        return {pool: [spec] for pool, spec in SHARDS.items()}
    if context.config['data']['download_scope'] != 'all_shards':
        raise ValueError('download_scope must be first_shards or all_shards')
    raw = context.data_dir / 'raw'
    metadata = raw / f'shard_index_{REVISION}.json'
    if metadata.exists():
        entries = read_json(metadata)
    else:
        url = f'https://huggingface.co/api/datasets/tanganke/nyuv2/tree/{REVISION}/data'
        response = requests.get(url, timeout=(30, 60))
        response.raise_for_status()
        entries = response.json()
        write_json(metadata, entries)
    selected = {}
    for pool, count in [('train', 8), ('val', 6)]:
        selected[pool] = []
        for i in range(count):
            name = f'{pool}-{i:05d}-of-{count:05d}.parquet'
            entry = next(e for e in entries if e['path'] == f'data/{name}')
            size, digest = entry['lfs']['size'], entry['lfs']['oid']
            if size != entry['size'] or not 0 < size < 5_000_000_000 or len(digest) != 64:
                raise ValueError('invalid pinned shard metadata')
            if i == 0 and (name, size, digest) != SHARDS[pool]:
                raise ValueError('pinned shard index differs from known smoke checksum')
            selected[pool].append((name, size, digest))
    return selected


def download(context):
    raw = context.data_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    artifacts = []
    shards = selected_shards(context)
    for split, (name, size, digest) in [(pool, spec) for pool, specs in shards.items() for spec in specs]:
        path = raw / name
        url = f'https://huggingface.co/datasets/tanganke/nyuv2/resolve/{REVISION}/data/{name}'
        if not path.exists() or path.stat().st_size != size or sha256_file(path) != digest:
            temporary = path.with_suffix('.part')
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                count = 0
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError('download larger than pinned shard')
                        stream.write(chunk)
            if temporary.stat().st_size != size or sha256_file(temporary) != digest:
                raise ValueError('download checksum mismatch')
            temporary.replace(path)
        print(f'NYUv2 {split}: {size} bytes SHA256 verified', flush=True)
        artifacts.append(path)
    artifacts.append(write_json(raw / 'source.json', {
        'repository': 'tanganke/nyuv2', 'revision': REVISION,
        'source_card': f'https://huggingface.co/datasets/tanganke/nyuv2/blob/{REVISION}/README.md',
        'lineage': 'mirror credits ForkMerge / Tsinghua Cloud; not byte-verified against original MTAN Dropbox',
        'license': 'mirror does not declare a dataset license; follow original NYUv2 terms',
        'scene_ids': None, 'official_pools': {'train': 795, 'val_as_test': 654},
        'noise_column': 'ignored, never used as supervision or input',
        'files': {v[0]: {'pool': k, 'name': v[0], 'bytes': v[1], 'sha256': v[2]} for k, specs in shards.items() for v in specs},
    }))
    return artifacts


def processed_dir(context):
    return context.data_dir / 'processed' / context.config['experiment']


def prepare(context):
    import pyarrow.parquet as pq
    settings = context.config['data']
    if settings.get('storage') == 'per_image':
        return prepare_streaming(context)
    if settings.get('split_mode') != 'image_disjoint_smoke':
        raise ValueError('mirror lacks scene identity; only image_disjoint_smoke is supported')
    target_size = settings['resolution']
    if any(int(v) % 32 for v in target_size) or min(target_size) < 64:
        raise ValueError('resolution must be multiples of 32 and >=64 for five-scale BatchNorm')
    out = processed_dir(context)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: [] for k in ('image', 'segmentation', 'depth', 'normal')}
    ids, image_hashes, counts = [], [], []
    for pool, count in [('train', settings['train_samples']), ('val', settings['test_samples'])]:
        rows = pq.read_table(context.data_dir / 'raw' / SHARDS[pool][0], columns=list(tensors)).slice(0, count).to_pylist()
        if len(rows) != count:
            raise ValueError('sample request exceeds downloaded shard')
        counts.append(count)
        for i, row in enumerate(rows):
            arrays = {k: np.array(row[k], dtype=np.int64 if k == 'segmentation' else np.float32) for k in tensors}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('nonfinite raw labels/input')
            if not np.isin(arrays['segmentation'], np.arange(-1, 13)).all():
                raise ValueError('expected 13-class labels with -1 invalid pixels')
            ids.append(f'{pool}:shard0:row{i}')
            image_hashes.append(hashlib.sha256(arrays['image'].tobytes()).hexdigest())
            for key, value in arrays.items():
                tensor = torch.from_numpy(value).float()
                if key == 'segmentation':
                    tensor = F.interpolate(tensor[None, None], size=target_size, mode='nearest')[0, 0].long()
                else:
                    # Nearest preserves invalid geometry masks without mixing invalid zero values.
                    tensor = F.interpolate(tensor[None], size=target_size, mode='bilinear' if key == 'image' else 'nearest',
                                           **({'align_corners': False} if key == 'image' else {}))[0]
                    if key == 'normal':
                        tensor = F.normalize(tensor, dim=0, eps=1e-8)
                tensors[key].append(tensor)
    if len(set(image_hashes)) != len(ids):
        raise ValueError('duplicate RGB frames in selected pools')
    permutation = np.random.default_rng(settings['split_seed']).permutation(counts[0])
    sizes = settings['split_counts']
    if sum(sizes.values()) != counts[0] or any(n < 1 for n in sizes.values()):
        raise ValueError('split_counts must partition selected train pool into nonempty parts')
    offset, splits = 0, {}
    for name in ('a_train', 'a_val', 'b_train', 'b_val'):
        splits[name] = permutation[offset:offset + sizes[name]].tolist()
        offset += sizes[name]
    splits['y_test'] = list(range(counts[0], sum(counts)))
    payload = {k: torch.stack(v) for k, v in tensors.items()}
    payload['ids'] = ids
    torch.save(payload, out / 'dataset.pt')
    split_path = write_json(out / 'splits.json', splits)
    manifest = write_json(out / 'manifest.json', {
        'identity': context.identity, 'dataset_sha256': sha256_file(out / 'dataset.pt'),
        'splits_sha256': sha256_file(split_path), 'ids': ids, 'image_sha256': image_hashes,
        'split_ids': {k: [ids[i] for i in v] for k, v in splits.items()},
        'counts': {k: len(v) for k, v in splits.items()}, 'resolution': target_size,
        'grouping': 'image only; scene disjointness UNVERIFIED; smoke test only',
        'label_mapping': 'mirror 0..12, -1 ignore; no remapping',
        'normal_labels': 'mirror precomputed normals, generation version unverified',
        'depth_units': 'mirror continuous depth convention; physical scale not independently calibrated',
        'normalization': 'none fitted; RGB values retained; normals unit-normalized',
        'pretrained': False, 'augmentation': False,
    })
    return [out / 'dataset.pt', split_path, manifest]


def load_data(context):
    return torch.load(processed_dir(context) / 'dataset.pt', map_location='cpu', weights_only=True)


class Images(Dataset):
    def __init__(self, arrays, indices, auxiliary=True):
        self.arrays, self.indices, self.auxiliary = arrays, indices, auxiliary

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        keys = ['image', 'segmentation'] + (['depth', 'normal'] if self.auxiliary else [])
        return {k: self.arrays[k][i] for k in keys}


def make_loader(context, split, seed, batch_size, shuffle=False, auxiliary=True):
    indices = read_json(processed_dir(context) / 'splits.json')[split]
    dataset = DiskImages(context, indices, auxiliary) if context.config['data'].get('storage') == 'per_image' else Images(load_data(context), indices, auxiliary)
    workers = int(context.config.get('loader', {}).get('workers', 0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=workers,
                      pin_memory=context.config['runtime']['device'].startswith('cuda'), persistent_workers=workers > 0)


def applicable_recipes(context, variant):
    recipes = list(context.config['refiner']['recipes'])
    return [r for r in recipes if r in ('embedding', 'embedding_matched')] if variant == 'single_task' else recipes


def data_identity(context):
    name = 'manifest.json' if context.config['data'].get('storage') == 'per_image' else 'dataset.pt'
    return {f'{key}_sha256': sha256_file(processed_dir(context) / key) for key in (name, 'splits.json')}


def sample_ids(context):
    if context.config['data'].get('storage') == 'per_image':
        return read_json(processed_dir(context) / 'manifest.json')['ids']
    return load_data(context)['ids']


def allocate_splits(ids, train_count, settings, scenes=None):
    sizes = settings['split_counts']
    names = ('a_train', 'a_val', 'b_train', 'b_val')
    if set(sizes) != set(names) or sum(sizes.values()) != train_count or any(n < 1 for n in sizes.values()):
        raise ValueError('split_counts must partition train pool into four nonempty parts')
    rng = np.random.default_rng(settings['split_seed'])
    splits = {k: [] for k in names}
    mode = settings['split_mode']
    if mode in ('image_disjoint_smoke', 'image_disjoint_exploratory'):
        if scenes is not None:
            raise ValueError('scene manifest requires scene_grouped split_mode')
        permutation = rng.permutation(train_count).tolist()
        offset = 0
        for name in names:
            splits[name] = permutation[offset:offset+sizes[name]]
            offset += sizes[name]
    elif mode == 'scene_grouped':
        if scenes is None or any(i not in scenes or not isinstance(scenes[i], str) or not scenes[i] for i in ids):
            raise ValueError('scene_grouped requires verified scene IDs for every selected train and test row')
        train_scenes = {scenes[i] for i in ids[:train_count]}
        if train_scenes & {scenes[i] for i in ids[train_count:]}:
            raise ValueError('official train/test scene overlap; cannot silently relabel test pool')
        groups = {}
        for i in range(train_count):
            groups.setdefault(scenes[ids[i]], []).append(i)
        if len(groups) < 4:
            raise ValueError('at least four training scenes required')
        keys = list(groups)
        rng.shuffle(keys)
        # Allocate large groups first; seeded tie order. Counts can deviate from targets.
        keys.sort(key=lambda k: len(groups[k]), reverse=True)
        for key in keys:
            name = max(names, key=lambda n: (sizes[n]-len(splits[n]))/sizes[n])
            splits[name].extend(groups[key])
        if any(not splits[n] for n in names):
            raise ValueError('scene allocation created empty partition')
    else:
        raise ValueError('unsupported split_mode')
    splits['y_test'] = list(range(train_count, len(ids)))
    return splits


def resize_row(row, resolution):
    tensors, arrays = {}, {}
    for key in ('image', 'segmentation', 'depth', 'normal'):
        arrays[key] = np.array(row[key], dtype=np.int64 if key == 'segmentation' else np.float32)
    if any(not np.isfinite(v).all() for v in arrays.values()) or not np.isin(arrays['segmentation'], np.arange(-1, 13)).all():
        raise ValueError('invalid NYUv2 values or class mapping')
    for key, value in arrays.items():
        tensor = torch.from_numpy(value).float()
        if key == 'segmentation':
            tensor = F.interpolate(tensor[None, None], size=resolution, mode='nearest')[0, 0].long()
        else:
            tensor = F.interpolate(tensor[None], size=resolution, mode='bilinear' if key == 'image' else 'nearest',
                                   **({'align_corners': False} if key == 'image' else {}))[0]
            if key == 'normal':
                tensor = F.normalize(tensor, dim=0, eps=1e-8)
        tensors[key] = tensor
    return tensors, hashlib.sha256(arrays['image'].tobytes()).hexdigest()


def prepare_streaming(context):
    import pyarrow.parquet as pq
    settings = context.config['data']
    resolution = settings['resolution']
    if len(resolution) != 2 or any(v % 32 or v < 64 for v in resolution):
        raise ValueError('resolution must contain two multiples of 32, each >=64')
    out = processed_dir(context)
    (out / 'samples').mkdir(parents=True, exist_ok=True)
    shards = selected_shards(context)
    records, ids, hashes, artifacts = [], [], [], []
    for pool, desired in [('train', settings['train_samples']), ('val', settings['test_samples'])]:
        remaining = desired
        for shard, (name, size, digest) in enumerate(shards[pool]):
            if not remaining:
                break
            raw = context.data_dir / 'raw' / name
            if raw.stat().st_size != size or sha256_file(raw) != digest:
                raise ValueError('raw shard identity mismatch')
            rows = pq.ParquetFile(raw).iter_batches(batch_size=1, columns=['image', 'segmentation', 'depth', 'normal'])
            for row_index, batch in enumerate(rows):
                if not remaining:
                    break
                payload, rgb_hash = resize_row(batch.to_pylist()[0], resolution)
                path = out / 'samples' / f'{len(ids):05d}.pt'
                temporary = path.with_suffix('.tmp')
                torch.save(payload, temporary)
                temporary.replace(path)
                ids.append(f'{pool}:shard{shard}:row{row_index}')
                hashes.append(rgb_hash)
                records.append({'path': str(path.relative_to(out)), 'sha256': sha256_file(path)})
                artifacts.append(path)
                remaining -= 1
        if remaining:
            raise ValueError(f'not enough downloaded {pool} rows; still need {remaining}')
    if len(set(hashes)) != len(hashes):
        raise ValueError('duplicate RGB frames in selected pools')
    scene_path = settings.get('scene_manifest')
    scenes = read_json(context.resolve(scene_path)) if scene_path else None
    splits = allocate_splits(ids, settings['train_samples'], settings, scenes)
    split_path = write_json(out / 'splits.json', splits)
    manifest = write_json(out / 'manifest.json', {
        'identity': context.identity, 'storage': 'per_image', 'ids': ids, 'samples': records,
        'image_sha256': hashes, 'source_sha256': sha256_file(context.data_dir / 'raw' / 'source.json'),
        'splits_sha256': sha256_file(split_path), 'counts': {k: len(v) for k, v in splits.items()},
        'split_ids': {k: [ids[i] for i in v] for k, v in splits.items()}, 'resolution': resolution,
        'grouping': settings['split_mode'], 'scene_manifest_sha256': sha256_file(context.resolve(scene_path)) if scene_path else None,
        'label_mapping': '0..12, -1 ignore', 'normalization': 'RGB retained; normals normalized; no fitted statistics',
        'normal_label_version': 'mirror; not independently verified', 'pretrained': False,
    })
    return [*artifacts, split_path, manifest]


class DiskImages(Dataset):
    def __init__(self, context, indices, auxiliary):
        self.directory = processed_dir(context)
        self.manifest = read_json(self.directory / 'manifest.json')
        if self.manifest['identity'] != context.identity:
            raise ValueError('processed dataset identity changed')
        self.indices, self.auxiliary, self.verified = indices, auxiliary, set()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record = self.manifest['samples'][self.indices[index]]
        path = self.directory / record['path']
        if record['path'] not in self.verified:
            if sha256_file(path) != record['sha256']:
                raise ValueError('processed sample checksum mismatch')
            self.verified.add(record['path'])
        payload = torch.load(path, map_location='cpu', weights_only=True)
        keys = ['image', 'segmentation'] + (['depth', 'normal'] if self.auxiliary else [])
        return {k: payload[k] for k in keys}
