"""Bounded snapshot download, material/structure deduplication and A/B/Y loaders."""
import gzip
import hashlib
import heapq
from collections import Counter, defaultdict
from functools import lru_cache
import urllib.request

import ijson
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pipeline.io import read_json, write_json, sha256_file, json_hash

SPLITS = ('a_train', 'a_val', 'b_train', 'b_val', 'y_test')
class NonfiniteJSONReader:
    """Replace bare NaN/Infinity with null, preserving strings across chunk boundaries.

    Historical pandas JSON contains non-standard constants even in unused fields.
    Never turn missing properties into numerical zeros.
    """
    def __init__(self, stream, chunk_bytes=65536):
        import re
        self.stream, self.chunk_bytes = stream, chunk_bytes
        self.pending, self.output, self.eof = b'', b'', False
        self.tokens = re.compile(rb'"(?:[^"\\]|\\.)*"|(?P<nf>-?Infinity|NaN)|(?P<open>"(?:[^"\\]|\\.)*(?:\\)?$)')

    def read(self, size=-1):
        if size == 0:
            return b''
        while not self.eof and (size < 0 or len(self.output) < size):
            raw = self.stream.read(self.chunk_bytes)
            self.eof = not raw
            self.pending += raw
            cutoff = len(self.pending) if self.eof else max(0, len(self.pending)-16)
            for match in self.tokens.finditer(self.pending):
                if match.group('open') is not None:
                    cutoff = min(cutoff, match.start())
                    break
                if match.start() < cutoff < match.end():
                    cutoff = match.start()
            ready, self.pending = self.pending[:cutoff], self.pending[cutoff:]
            self.output += self.tokens.sub(lambda m: b'null' if m.group('nf') else m.group(), ready)
            if self.eof and self.pending:
                raise ValueError('unterminated JSON string at EOF')
        if size < 0:
            result, self.output = self.output, b''
        else:
            result, self.output = self.output[:size], self.output[size:]
        return result


def download(context):
    source = context.config['data']['source']
    path = context.data_dir / source['name']
    limit = context.config['data']['max_file_bytes']
    if not path.exists():
        temp = path.with_name(path.name + '.part')
        with urllib.request.urlopen(source['url'], timeout=60) as response, temp.open('wb') as stream:
            size = response.headers.get('Content-Length')
            if size and int(size) > limit:
                raise ValueError('source exceeds single-file limit')
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > limit:
                    raise ValueError('source exceeds single-file limit')
                stream.write(chunk)
        if sha256_file(temp) != source['sha256']:
            raise ValueError('snapshot SHA256 mismatch')
        temp.replace(path)
    if path.stat().st_size > limit or sha256_file(path) != source['sha256']:
        raise ValueError('snapshot checksum/size mismatch')
    manifest = context.output_dir / 'download_manifest.json'
    write_json(manifest, {**source, 'bytes': path.stat().st_size, 'license': 'CC BY 4.0',
                         'version': 'MP 2018-10-18 via matminer mp_all_20181018'})
    return [path, manifest]


def make_matcher():
    return StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,
                            scale=True, attempt_supercell=False)


def unique_structures(records):
    """Keep one material per equivalent structure, before any split assignment."""
    matcher = make_matcher()
    buckets, seen_ids, accepted, rejected = defaultdict(list), set(), [], Counter()
    for record in records:
        if record['id'] in seen_ids:
            rejected['duplicate_id'] += 1
            continue
        seen_ids.add(record['id'])
        structure = Structure.from_dict(record['structure'])
        key = structure.composition.reduced_formula
        if any(matcher.fit(structure, previous) for previous in buckets[key]):
            rejected['equivalent_structure'] += 1
            continue
        buckets[key].append(structure)
        accepted.append((record, structure))
    return accepted, dict(rejected)


def prepare(context):
    settings = context.config['data']
    source = context.data_dir / settings['source']['name']
    with gzip.open(source, 'rb') as stream:
        columns = next(ijson.items(stream, 'columns'))
    field = {name: i for i, name in enumerate(columns)}
    for name in ('mpid', 'structure', 'gap pbe', 'e_form'):
        if name not in field:
            raise ValueError(f'missing required field: {name}')
    pool = []
    pool_size = sum(settings['sizes'].values()) * settings['candidate_multiplier']
    rejected, scanned = Counter(), 0
    with gzip.open(source, 'rb') as stream:
        for row in ijson.items(NonfiniteJSONReader(stream), 'data.item', use_float=True):
            scanned += 1
            sid, structure = row[field['mpid']], row[field['structure']]
            try:
                targets = [float(row[field['gap pbe']]), float(row[field['e_form']])]
            except (TypeError, ValueError):
                rejected['missing_target'] += 1
                continue
            if not np.isfinite(targets).all() or targets[0] < 0:
                rejected['invalid_target'] += 1
                continue
            sites = structure.get('sites', []) if isinstance(structure, dict) else []
            if not 1 <= len(sites) <= settings['max_atoms']:
                rejected['atom_count'] += 1
                continue
            if any(len(site['species']) != 1 or site['species'][0].get('occu', 1) != 1 for site in sites):
                rejected['disordered'] += 1
                continue
            # MP identity determines sampling, independent of targets and file ordering.
            rank = int(hashlib.sha256(f"{settings['sampling_seed']}:{sid}".encode()).hexdigest(), 16)
            record = {'id': str(sid), 'structure': structure, 'targets': targets}
            item = (-rank, scanned, record)
            if len(pool) < pool_size:
                heapq.heappush(pool, item)
            elif item[0] > pool[0][0]:
                heapq.heapreplace(pool, item)
    candidates = [item[2] for item in sorted(pool, key=lambda item: -item[0])]
    valid = []
    for record in candidates:
        structure = Structure.from_dict(record['structure'])
        if any(not 1 <= site.specie.Z <= 100 for site in structure):
            rejected['atomic_number'] += 1
            continue
        if not np.isfinite(structure.cart_coords).all() or structure.volume <= 0:
            rejected['invalid_geometry'] += 1
            continue
        valid.append(record)
    unique, dedup = unique_structures(valid)
    n = sum(settings['sizes'].values())
    if len(unique) < n:
        raise ValueError(f'only {len(unique)} unique candidates, need {n}')
    selected = unique[:n]
    rng = np.random.default_rng(settings['split_seed'])
    order = rng.permutation(n)
    records = []
    for record, structure in selected:
        # Retain the released cell and site order, like the author's CIF loader.
        records.append({'id': record['id'], 'z': [site.specie.Z for site in structure],
                        'coords': structure.cart_coords.tolist(), 'lattice': structure.lattice.matrix.tolist(),
                        'structure': record['structure'], 'targets': record['targets']})
    splits, start = {}, 0
    for split in SPLITS:
        count = settings['sizes'][split]
        splits[split] = order[start:start+count].tolist()
        start += count
    a_targets = np.array([records[i]['targets'] for i in splits['a_train']])
    norm = {'mean': a_targets.mean(0).tolist(), 'std': np.maximum(a_targets.std(0), 1e-8).tolist(),
            'fit_split': 'a_train', 'tasks': ['gap_pbe_eV', 'e_form_eV_per_atom']}
    paths = [context.output_dir / name for name in ('prepared.json', 'split_manifest.json', 'target_normalization.json')]
    write_json(paths[0], records)
    write_json(paths[1], {'indices': splits, 'ids': {s: [records[i]['id'] for i in ids] for s, ids in splits.items()},
                        'scanned_rows': scanned, 'candidate_pool': len(pool), 'unique_candidates': len(unique),
                        'filter_rejections': dict(rejected), 'dedup_rejections': dedup,
                        'sampling': 'lowest SHA256(seed:mpid), then seeded partition',
                        'matcher': {'ltol': 0.2, 'stol': 0.3, 'angle_tol': 5, 'primitive_cell': True, 'scale': True},
                        'cell': 'released relaxed cell', 'site_order': 'released order',
                        'max_atoms': settings['max_atoms'], 'normalization_fit': 'a_train'})
    write_json(paths[2], norm)
    print(f'prepared {n} unique MP structures from {scanned} records; rejected equivalents {dedup}', flush=True)
    return paths


def data_identity(context):
    return json_hash({name: sha256_file(context.output_dir / name)
                      for name in ('prepared.json', 'split_manifest.json', 'target_normalization.json')})


@lru_cache(maxsize=8)
def load_records(path):
    return read_json(path)


class Crystals(Dataset):
    def __init__(self, context, split, auxiliary=False):
        if auxiliary and split not in ('a_train', 'a_val'):
            raise ValueError('auxiliary truth may only be accessed on A')
        self.records = load_records(str(context.output_dir / 'prepared.json'))
        self.indices = read_json(context.output_dir / 'split_manifest.json')['indices'][split]
        self.auxiliary = auxiliary

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.records[self.indices[index]]
        result = {'id': row['id'], 'z': torch.tensor(row['z']),
                  'coords': torch.tensor(row['coords'], dtype=torch.float32),
                  'main': torch.tensor(row['targets'][0], dtype=torch.float32)}
        if self.auxiliary:
            result['auxiliary'] = torch.tensor(row['targets'][1], dtype=torch.float32)
        return result


def collate(rows):
    n = max(len(row['z']) for row in rows)
    z = torch.zeros(len(rows), n, dtype=torch.long)
    coords = torch.zeros(len(rows), n, 3)
    mask = torch.ones(len(rows), n, dtype=torch.bool)
    for i, row in enumerate(rows):
        size = len(row['z'])
        z[i, :size], coords[i, :size], mask[i, :size] = row['z'], row['coords'], False
    result = {'id': [row['id'] for row in rows], 'z': z, 'coords': coords, 'mask': mask,
              'main': torch.stack([row['main'] for row in rows])}
    if 'auxiliary' in rows[0]:
        result['auxiliary'] = torch.stack([row['auxiliary'] for row in rows])
    return result


def make_loader(context, split, seed=1, batch_size=32, shuffle=False, auxiliary=False):
    return DataLoader(Crystals(context, split, auxiliary), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=context.config['runtime'].get('num_workers', 0),
                      generator=torch.Generator().manual_seed(seed))