"""Local, streaming HDF5 extraction with parent-SMILES disjoint A/B/Y splits.

Official nested schema: file[molecule_id][Geom-mr-is-ct-u][property].
Only five declared properties are read, never eL or HLgap.
"""
from collections import Counter, defaultdict
import hashlib
import heapq
from itertools import zip_longest
from pathlib import Path
import re
import urllib.request

from pipeline.io import read_json, write_json, sha256_file, json_hash
from experiments.sci_qm7x.common import (
    SPLITS, FIELDS, UNITS, SOURCE_URL, PAPER_URL, DUPLICATES_MD5,
    validate_config, applicable_recipes, data_identity, assert_prepared,
)

GEOMETRY = re.compile(r'^Geom-m(?P<r>[1-9]\d*)-i(?P<s>[1-9]\d*)-c(?P<t>[1-9]\d*)-(?P<u>opt|d?[1-9]\d*)$')


def parse_geometry(name):
    match = GEOMETRY.fullmatch(name)
    if match is None:
        raise ValueError(f'unrecognized released geometry ID: {name}')
    parts = match.groupdict()
    if parts['u'] != 'opt' and not 1 <= int(parts['u'].lstrip('d')) <= 100:
        raise ValueError('displacement index outside 1..100')
    return parts


def rank(seed, identity):
    return int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest(), 16)


def file_digest(path, algorithm='md5'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_download(url, target, maximum, expected_md5=None):
    """Explicit opt-in only; no automatic extraction or unbounded single files."""
    if not url.startswith('https://zenodo.org/records/4288677/files/'):
        raise ValueError('only the locked official Zenodo record is allowed')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            length = response.headers.get('Content-Length')
            if length and int(length) > maximum:
                raise ValueError('download exceeds declared single-file limit')
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > maximum:
                    raise ValueError('download exceeds declared single-file limit')
                output.write(chunk)
        if expected_md5 and file_digest(temporary) != expected_md5:
            raise ValueError('download MD5 mismatch')
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_paths(context):
    result = []
    for source in context.config['data']['sources']:
        path = Path(source['path']).expanduser()
        path = path if path.is_absolute() else context.data_dir / path
        if path.suffix.lower() not in ('.hdf5', '.h5'):
            raise ValueError('prepare accepts extracted local .hdf5/.h5 only; no automatic XZ expansion')
        result.append((source, path.resolve()))
    return result


def duplicate_path(context):
    path = Path(context.config['data']['duplicates_file']).expanduser()
    return (path if path.is_absolute() else context.data_dir / path).resolve()


def download(context):
    validate_config(context.config)
    settings = context.config['data']
    # Network writes are possible only through an explicit opt-in AND file list.
    downloads = settings.get('download_files', [])
    if downloads and not settings.get('allow_download', False):
        raise ValueError('download_files require explicit data.allow_download=true')
    downloaded_paths = []
    for entry in downloads:
        name = entry['name']
        if Path(name).name != name or not entry.get('md5'):
            raise ValueError('download file must be a basename with locked MD5')
        path = context.data_dir / name
        if not path.exists():
            bounded_download(entry['url'], path, settings['max_file_bytes'], entry['md5'])
        if path.stat().st_size > settings['max_file_bytes'] or file_digest(path) != entry['md5']:
            raise ValueError('existing download size/checksum mismatch')
        downloaded_paths.append(path)
    sources, pending = [], []
    for specification, path in source_paths(context):
        if not path.is_file():
            if settings.get('allow_download', False) and downloads:
                pending.append(str(path))
                continue
            raise FileNotFoundError(f'provide local extracted QM7-X HDF5: {path}; compressed shards are never auto-expanded')
        digest = sha256_file(path)
        if specification.get('sha256') and specification['sha256'] != digest:
            raise ValueError(f'local source SHA256 mismatch: {path}')
        sources.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
    duplicate = duplicate_path(context)
    duplicate_record = None
    if duplicate.is_file():
        if file_digest(duplicate) != settings['duplicates_md5']:
            raise ValueError('DupMols.dat checksum mismatch; require official version 2 bytes')
        duplicate_record = {'path': str(duplicate), 'sha256': sha256_file(duplicate), 'md5': file_digest(duplicate)}
    elif not (settings.get('allow_download', False) and downloads):
        raise FileNotFoundError(f'official DupMols.dat is required: {duplicate}')
    manifest = context.output_dir / 'download_manifest.json'
    write_json(manifest, {'identity': context.identity, 'record': SOURCE_URL, 'version': '2.0',
                         'license': 'CC BY 4.0', 'license_metadata': 'https://zenodo.org/api/records/4288677',
                         'sources': sources, 'pending_local_hdf5': pending, 'duplicates': duplicate_record,
                         'network_enabled': settings.get('allow_download', False), 'automatic_decompression': False})
    return [manifest, *downloaded_paths]


def duplicate_families(path):
    """Official list has excluded opt IDs, not representative-pair mappings.

    Exclude every listed equilibrium family AND its 100 displacements. All
    remaining stereo/conformer families with the same r are already one group.
    No nonexistent cross-r equivalence mapping is inferred from the list.
    """
    families = set()
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        name = line.strip()
        if not name:
            continue
        if parse_geometry(name)['u'] != 'opt':
            raise ValueError('official duplicate list must contain opt IDs')
        families.add(name[:-4])
    return families


def partition_parents(parents, sizes, seed):
    """Partition molecular groups BEFORE selecting any conformations."""
    parents = sorted(set(parents), key=lambda parent: (rank(seed, parent), parent))
    if len(parents) < len(SPLITS):
        raise ValueError('fewer than five independent molecular groups')
    total = sum(sizes.values())
    counts = {s: max(1, int(len(parents) * sizes[s] / total)) for s in SPLITS}
    while sum(counts.values()) > len(parents):
        split = max(SPLITS, key=lambda s: counts[s])
        counts[split] -= 1
    while sum(counts.values()) < len(parents):
        split = max(SPLITS, key=lambda s: len(parents) * sizes[s] / total - counts[s])
        counts[split] += 1
    result, start = {}, 0
    for split in SPLITS:
        result[split] = parents[start:start + counts[split]]
        start += counts[split]
    return result


def scan_candidates(context):
    """Metadata scan stores at most cap candidates per parent, never all labels."""
    import h5py
    settings = context.config['data']
    duplicates = duplicate_families(duplicate_path(context))
    candidates, seen_families, rejected = defaultdict(list), {}, Counter()
    cap, scanned = settings['max_conformations_per_parent'], 0
    for file_index, (_, path) in enumerate(source_paths(context)):
        with h5py.File(path, 'r') as source:
            for molecule in sorted(source, key=str):
                if not isinstance(source[molecule], h5py.Group):
                    raise ValueError('expected molecule groups at HDF5 root')
                for name in source[molecule]:
                    scanned += 1
                    parts = parse_geometry(name)
                    if str(int(molecule)) != parts['r']:
                        raise ValueError('root molecule identity disagrees with geometry r')
                    family = name.rsplit('-', 1)[0]
                    location = (file_index, molecule)
                    previous = seen_families.setdefault(family, location)
                    if previous != location:
                        raise ValueError('same geometry family appears in multiple source groups/shards')
                    if family in duplicates:
                        rejected['official_duplicate_family'] += 1
                        continue
                    if settings.get('equilibrium_only', False) and parts['u'] != 'opt':
                        rejected['non_equilibrium'] += 1
                        continue
                    parent = f"m{parts['r']}"
                    # Sample only identity, without inspecting target/auxiliary values.
                    item = (-rank(settings['sampling_seed'], name), name, file_index, molecule)
                    heap = candidates[parent]
                    if len(heap) < cap:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
    return {parent: sorted(values, key=lambda value: (-value[0], value[1])) for parent, values in candidates.items()}, {
        'scanned_structures': scanned, 'rejections': dict(rejected), 'official_duplicate_families': len(duplicates)}


def select_candidates(candidates, groups, sizes):
    selected = {}
    for split in SPLITS:
        # Deterministic round-robin draws from separately assigned parent groups.
        rows = [item for layer in zip_longest(*(candidates[p] for p in groups[split]))
                for item in layer if item is not None]
        if len(rows) < sizes[split]:
            raise ValueError(f'{split}: group-disjoint capped capacity {len(rows)} < requested {sizes[split]}; change budget/cap explicitly')
        selected[split] = rows[:sizes[split]]
    return selected


def read_structure(group, name, maximum_atoms):
    import numpy as np
    for field in FIELDS.values():
        if field not in group:
            raise ValueError(f'{name}: missing official field {field}')
    z_raw = np.asarray(group[FIELDS['z']][()])
    xyz = np.asarray(group[FIELDS['coords']][()], dtype=np.float64)
    main = np.asarray(group[FIELDS['main']][()], dtype=np.float64)
    charge = np.asarray(group[FIELDS['charge']][()], dtype=np.float64)
    polar = np.asarray(group[FIELDS['polarizability']][()], dtype=np.float64)
    n = len(z_raw)
    if z_raw.shape != (n,) or xyz.shape != (n, 3) or main.shape not in ((1,), ()) or charge.shape != (n,) or polar.shape != (n,):
        raise ValueError(f'{name}: invalid official property shape')
    if not 1 <= n <= maximum_atoms or not np.isin(z_raw, [1, 6, 7, 8, 16, 17]).all():
        raise ValueError(f'{name}: invalid atom count/elements')
    if not all(np.isfinite(value).all() for value in (z_raw, xyz, main, charge, polar)):
        raise ValueError(f'{name}: nonfinite required label or geometry')
    if (polar <= 0).any():
        raise ValueError(f'{name}: nonpositive atomic polarizability')
    return z_raw.astype(np.int64), xyz.astype(np.float32), float(main.reshape(-1)[0]), np.stack([charge, polar], -1).astype(np.float32)


def prepare(context):
    import h5py
    import torch
    from contextlib import ExitStack
    from pipeline.fit import save_checkpoint
    validate_config(context.config)
    settings = context.config['data']
    # Recheck immutable local source bytes even when called as a standalone stage.
    download(context)
    download_record = read_json(context.output_dir / 'download_manifest.json')
    if download_record['pending_local_hdf5'] or download_record['duplicates'] is None:
        raise FileNotFoundError('prepare requires all extracted local HDF5 sources and official DupMols.dat; download never decompresses archives')
    candidates, scan = scan_candidates(context)
    groups = partition_parents(candidates, settings['sizes'], settings['split_seed'])
    selection = select_candidates(candidates, groups, settings['sizes'])
    n, maximum = sum(settings['sizes'].values()), settings['max_atoms']
    tensors = {'z': torch.zeros(n, maximum, dtype=torch.long), 'coords': torch.zeros(n, maximum, 3),
               'main': torch.zeros(n), 'auxiliary': torch.zeros(n, maximum, 2), 'mask': torch.zeros(n, maximum, dtype=torch.bool)}
    ids, parents, indices, cursor = [], [], {}, 0
    with ExitStack() as stack:
        sources = [stack.enter_context(h5py.File(path, 'r')) for _, path in source_paths(context)]
        for split in SPLITS:
            indices[split] = []
            for _, name, file_index, molecule in selection[split]:
                z, coords, main, auxiliary = read_structure(sources[file_index][molecule][name], name, maximum)
                length = len(z)
                tensors['z'][cursor, :length] = torch.from_numpy(z)
                tensors['coords'][cursor, :length] = torch.from_numpy(coords)
                tensors['main'][cursor] = main
                tensors['auxiliary'][cursor, :length] = torch.from_numpy(auxiliary)
                tensors['mask'][cursor, :length] = True
                ids.append(name)
                parents.append(f"m{parse_geometry(name)['r']}")
                indices[split].append(cursor)
                cursor += 1
    a = torch.tensor(indices['a_train'])
    local = tensors['auxiliary'][a][tensors['mask'][a]].double()
    main = tensors['main'][a].double()
    norm = {'main_mean': main.mean().item(), 'main_std': main.std(unbiased=False).clamp_min(1e-8).item(),
            'aux_mean': local.mean(0).tolist(), 'aux_std': local.std(0, unbiased=False).clamp_min(1e-8).tolist(),
            'fit_split': 'a_train', 'auxiliary_weighting': 'one observation per valid atom', 'fields': FIELDS, 'units': UNITS}
    paths = [context.output_dir / name for name in ('prepared.pt', 'split_manifest.json', 'target_normalization.json', 'source_manifest.json')]
    # Conservative tensor bound before writing. No huge consolidated raw database.
    byte_count = sum(value.numel() * value.element_size() for value in tensors.values())
    if byte_count + len(str(ids).encode()) + 1_000_000 > settings['max_file_bytes']:
        raise ValueError('prepared file exceeds automatic single-file budget')
    save_checkpoint(paths[0], {**tensors, 'ids': ids, 'parents': parents, 'identity': context.identity})
    if paths[0].stat().st_size > settings['max_file_bytes']:
        paths[0].unlink()
        raise ValueError('prepared serialization exceeded single-file budget')
    represented = {s: sorted({parents[i] for i in indices[s]}) for s in SPLITS}
    write_json(paths[1], {'identity': context.identity, 'indices': indices, 'ids': {s: [ids[i] for i in indices[s]] for s in SPLITS},
                        'groups': groups, 'represented_groups': represented, 'group_counts': {s: len(groups[s]) for s in SPLITS},
                        'represented_group_counts': {s: len(represented[s]) for s in SPLITS}, 'structure_counts': settings['sizes'],
                        'group_rule': 'all Geom-mr-is-ct-u sharing r; no stereoisomer/conformer separation',
                        'duplicate_policy': 'exclude each official listed opt family and every displacement; remaining r families merged',
                        'sampling': 'parent split first; bottom hash capped conformations per parent, round-robin parent draw',
                        'sampling_seed': settings['sampling_seed'], 'split_seed': settings['split_seed'],
                        'equilibrium_only': settings.get('equilibrium_only', False), 'max_conformations_per_parent': settings['max_conformations_per_parent'],
                        'normalization_fit': 'a_train', **scan})
    write_json(paths[2], norm)
    write_json(paths[3], {'identity': context.identity, 'record': SOURCE_URL, 'paper': PAPER_URL,
                        'source_files': download_record['sources'], 'duplicates': download_record['duplicates'], 'license': 'CC BY 4.0',
                        'fields': FIELDS, 'units': UNITS, 'prepared_sha256': sha256_file(paths[0]),
                        'normalization_sha256': sha256_file(paths[2]), 'split_manifest_sha256': sha256_file(paths[1]), 'model_source': 'package-local project SchNet-style implementation; no cross-package imports'})
    return paths + [context.output_dir / 'download_manifest.json']


def load_prepared(context):
    import torch
    assert_prepared(context)
    saved = torch.load(context.output_dir / 'prepared.pt', map_location='cpu', weights_only=True)
    if saved['identity'] != context.identity:
        raise ValueError('prepared tensor identity mismatch')
    manifest = read_json(context.output_dir / 'split_manifest.json')
    count = sum(context.config['data']['sizes'].values())
    if len(saved['ids']) != count or len(saved['parents']) != count or any(len(saved[k]) != count for k in ('z', 'coords', 'main', 'mask', 'auxiliary')):
        raise ValueError('prepared tensor row count mismatch')
    for split in SPLITS:
        indices = manifest['indices'][split]
        if [saved['ids'][i] for i in indices] != manifest['ids'][split]:
            raise ValueError('prepared tensor and split row IDs disagree')
        expected = [f"m{parse_geometry(saved['ids'][i])['r']}" for i in indices]
        if [saved['parents'][i] for i in indices] != expected or not set(expected) <= set(manifest['groups'][split]):
            raise ValueError('prepared tensor row parents disagree with split groups')
    return saved


def make_loader(context, split, seed=1, batch_size=32, shuffle=False, auxiliary=False, diagnostics=False):
    import torch
    from torch.utils.data import Dataset, DataLoader
    if split not in SPLITS:
        raise ValueError('unknown split')
    if auxiliary and split not in ('a_train', 'a_val') and not (split == 'y_test' and diagnostics):
        raise ValueError('B auxiliary truth forbidden; Y truth requires explicit diagnostics flag')
    prepared = load_prepared(context)
    indices = read_json(context.output_dir / 'split_manifest.json')['indices'][split]

    class Molecules(Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, index):
            i = indices[index]
            result = {key: prepared[key][i] for key in ('z', 'coords', 'main', 'mask')}
            result['id'] = prepared['ids'][i]
            if auxiliary:
                result['auxiliary'] = prepared['auxiliary'][i]
            return result

    # Dataset implementation is local, so num_workers=0 is mandatory and portable.
    return DataLoader(Molecules(), batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
