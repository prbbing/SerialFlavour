"""Real NYUv2 shard download and image-disjoint smoke-test preparation."""

from pathlib import Path
import hashlib
import numpy as np
import requests
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from pipeline.io import read_json, write_json, sha256_file, json_hash

REVISION = 'b367b8b53c4dcefbcb4d9310b74976a63cc7f306'
SHARDS = {
    'train': ('train-00000-of-00008.parquet', 208564689, '2c340d96fa2d9d225e39721805fa302f93cf62d4506bc60d438f3aa766d30bd7'),
    'val': ('val-00000-of-00006.parquet', 224448496, '74366586abc14e81dc1bdd759dd655be0af9c123d8fa3112181cb6f03bd0db14'),
}


def download(context):
    raw = context.data_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for split, (name, size, digest) in SHARDS.items():
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
        'license': 'mirror does not declare a dataset license; follow original NYUv2 terms, research smoke only',
        'scene_ids': None, 'official_pools': {'train': 795, 'val_as_test': 654},
        'noise_column': 'ignored, never used as supervision or input',
        'files': {k: {'name': v[0], 'bytes': v[1], 'sha256': v[2]} for k, v in SHARDS.items()},
    }))
    return artifacts


def processed_dir(context):
    return context.data_dir / 'processed' / context.config['experiment']


def prepare(context):
    import pyarrow.parquet as pq
    settings = context.config['data']
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
    return DataLoader(Images(load_data(context), indices, auxiliary), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0)


def applicable_recipes(context, variant):
    return ['embedding', 'embedding_matched'] if variant == 'single_task' else list(context.config['refiner']['recipes'])


def data_identity(context):
    return {f'{name}_sha256': sha256_file(processed_dir(context) / name) for name in ('dataset.pt', 'splits.json')}
