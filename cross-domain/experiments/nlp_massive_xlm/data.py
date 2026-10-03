"""Pinned MASSIVE 1.0, source/text grouping, and first-subword slot alignment."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
import random
import re
import tarfile
import unicodedata

import requests
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizerFast

from pipeline.io import read_json, write_json, sha256_file, json_hash

OFFICIAL_COMMIT = 'f966f21846043aabef9b0f974fa7970027f43738'
TOKENIZER_REVISION = 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
ARCHIVE_URL = 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz'
PINNED_FILES = {
    'amazon-massive-dataset-1.0.tar.gz': '7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8',
    'tokenizer/tokenizer.json': 'a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c',
    'tokenizer/tokenizer_config.json': '994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce',
    'tokenizer/sentencepiece.bpe.model': 'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'tokenizer/config.json': 'd66ed8cd4f2a93b358c245e50736fa389ed4f35c0bae7aad0b32abb20c62b579',
}
SPLITS = ('a_train', 'a_val', 'b_train', 'b_val', 'y_test')
BASE_WEIGHTS_SHA256 = '6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb'
BASE_WEIGHTS_BYTES = 1115567652
BASE_CONFIG_SHA256 = PINNED_FILES['tokenizer/config.json']

def pretrained_directory(context):
    return context.data_dir / 'pretrained' / 'xlm_roberta_base'

def pretrained_enabled(context):
    return context.config['model'].get('initialization', 'random') == 'pretrained_base'

def pretrained_assets(context, allow_download=False):
    directory = pretrained_directory(context)
    paths = []
    for name, expected, maximum in [('config.json', BASE_CONFIG_SHA256, 100_000),
                                    ('model.safetensors', BASE_WEIGHTS_SHA256, 2_000_000_000)]:
        path = directory / name
        if not allow_download and not path.is_file():
            raise FileNotFoundError(f'missing offline base asset {path}; use scripts/prepare_assets.py on a network host')
        paths.append(fetch(f'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/{TOKENIZER_REVISION}/{name}',
                           path, expected, max_bytes=maximum))
    return paths

def processed_dir(context):
    return context.data_dir / 'processed' / context.config['experiment']

def fetch(url, path, expected, max_bytes=100_000_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + '.part')
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            if int(response.headers.get('Content-Length', 0)) > max_bytes:
                raise ValueError('download exceeds configured single-file limit')
            received = 0
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > max_bytes:
                        raise ValueError('download exceeds configured single-file limit')
                    stream.write(chunk)
        if sha256_file(temporary) != expected:
            raise ValueError('download checksum mismatch')
        temporary.replace(path)
    if path.stat().st_size > max_bytes or sha256_file(path) != expected:
        raise ValueError(f'raw input checksum/size mismatch: {path}')
    return path

def download(context):
    raw = context.data_dir / 'raw'
    artifacts, records = [], {}
    for name, expected in PINNED_FILES.items():
        url = ARCHIVE_URL if name.endswith('tar.gz') else (
            f'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/{TOKENIZER_REVISION}/{name.split("/")[-1]}')
        if not context.config['data'].get('allow_download', True) and not (raw / name).is_file():
            raise FileNotFoundError(f'missing offline raw input {raw / name}; use scripts/prepare_assets.py first')
        path = fetch(url, raw / name, expected)
        artifacts.append(path)
        records[name] = {'url': url, 'sha256': expected, 'bytes': path.stat().st_size}
    with tarfile.open(raw / 'amazon-massive-dataset-1.0.tar.gz', 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.name.split('/')[-1] == 'en-US.jsonl']
        if len(members) != 1 or not members[0].isfile() or members[0].size > 100_000_000:
            raise ValueError('invalid en-US archive member')
        content = archive.extractfile(members[0]).read()
    path = raw / 'en-US.jsonl'
    if path.exists() and path.read_bytes() != content:
        raise ValueError('existing en-US extraction differs from pinned archive')
    path.write_bytes(content)
    artifacts.append(path)
    records['en-US.jsonl'] = {'archive_member': members[0].name, 'sha256': sha256_file(path), 'bytes': len(content)}
    if pretrained_enabled(context):
        artifacts.extend(pretrained_assets(context, allow_download=context.config['data'].get('allow_download', False)))
        records['pretrained/model.safetensors'] = {
            'model_id': 'FacebookAI/xlm-roberta-base', 'revision': TOKENIZER_REVISION,
            'sha256': BASE_WEIGHTS_SHA256, 'bytes': BASE_WEIGHTS_BYTES,
            'initialization': 'public base masked-language-model encoder only; not MASSIVE fine-tuned'}
    artifacts.append(write_json(context.output_dir / 'download_manifest.json', {
        'dataset': 'MASSIVE 1.0', 'locale': 'en-US', 'data_license': 'CC-BY-4.0',
        'model_code_license': 'Apache-2.0', 'official_commit': OFFICIAL_COMMIT,
        'tokenizer_revision': TOKENIZER_REVISION, 'records': records,
        'official_model_sha256': sha256_file(context.experiment_root / 'official_xlmr.py'),
        'input_text': 'utt only; annot_utt is supervision only; no scenario or judgments as features',
        'pretrained_encoder': pretrained_enabled(context),
        'pretraining_exposure': ('public XLM-R base encoder and tokenizer; corpus overlap with B/Y not ruled out'
                                if pretrained_enabled(context) else 'public pretrained tokenizer only; smoke encoder random'),
    }))
    return artifacts

def parse_slots(row):
    # Equivalent en-US type/Other encoding to scripts/create_hf_dataset.py,
    # with an additional reconstruction check against the unannotated input.
    tokens, labels, cursor = [], [], 0
    annotation = row['annot_utt']
    for match in re.finditer(r'\[([^\[\]:]+?)\s*:\s*([^\[\]]+)\]', annotation):
        outside = annotation[cursor:match.start()].split()
        tokens.extend(outside)
        labels.extend(['Other'] * len(outside))
        inside = match.group(2).split()
        tokens.extend(inside)
        labels.extend([match.group(1).strip()] * len(inside))
        cursor = match.end()
    outside = annotation[cursor:].split()
    tokens.extend(outside)
    labels.extend(['Other'] * len(outside))
    if tokens != row['utt'].split() or len(labels) != len(tokens):
        raise ValueError(f'annotation/text mismatch for source id {row["id"]}')
    return tokens, labels

def normalized_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def group_rows(rows):
    parent = list(range(len(rows)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    keys = {}
    for i, row in enumerate(rows):
        for key in (('source', row['id']), ('text', normalized_text(row['utt']))):
            if key in keys:
                parent[find(i)] = find(keys[key])
            keys[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return [indices for _, indices in sorted(groups.items())]

def select_groups(groups, rows, budget, rng):
    # Round-robin intents, shuffled groups within intent; never split a group.
    # A conflicting-intent duplicate group is assigned by its smallest label.
    strata = defaultdict(list)
    for group in groups:
        strata[min(rows[i]['intent'] for i in group)].append(group)
    for value in strata.values():
        rng.shuffle(value)
    labels = sorted(strata)
    selected, used = [], set()
    while len(selected) < budget:
        progress = False
        for label in labels:
            if strata[label]:
                group = strata[label].pop()
                selected.extend(group)
                used.add(tuple(group))
                progress = True
                if len(selected) >= budget:
                    break
        if not progress:
            raise ValueError(f'insufficient group-disjoint records for requested budget {budget}')
    return selected, [g for g in groups if tuple(g) not in used]

def ratio_partition(groups, rows, fraction, rng):
    if not 0 < fraction < 1:
        raise ValueError('A fraction must lie strictly between zero and one')
    strata = defaultdict(list)
    for group in groups:
        strata[min(rows[i]['intent'] for i in group)].append(group)
    a, b = [], []
    for label in sorted(strata):
        values = strata[label]
        rng.shuffle(values)
        target = fraction * sum(len(g) for g in values)
        selected = 0
        for group in values:
            # Fill toward per-intent target without splitting source/text groups.
            if abs(selected + len(group) - target) <= abs(selected - target):
                a.extend(group)
                selected += len(group)
            else:
                b.extend(group)
    if not a or not b:
        raise ValueError('empty full-data A/B partition')
    return a, b

def split_records(rows, settings):
    groups = group_rows(rows)
    pools, removed = {'train': [], 'dev': [], 'test': []}, []
    for group in groups:
        partitions = {rows[i]['partition'] for i in group}
        # Protect the complete official test, then official dev. Discard earlier
        # partitions of a cross-partition duplicate group instead of exposing Y.
        destination = 'test' if 'test' in partitions else 'dev' if 'dev' in partitions else 'train'
        keep = [i for i in group if rows[i]['partition'] == destination]
        removed.extend(i for i in group if i not in keep)
        pools[destination].append(keep)
    rng = random.Random(settings['split_seed'])
    counts = settings.get('split_counts')
    splits = {}
    if settings.get('split_mode', 'grouped_subset') == 'official_grouped_full':
        if counts is not None:
            raise ValueError('full split must not specify subset split_counts')
        splits['a_train'], splits['b_train'] = ratio_partition(pools['train'], rows, settings['a_train_fraction'], rng)
        splits['a_val'], splits['b_val'] = ratio_partition(pools['dev'], rows, settings['a_val_fraction'], rng)
        splits['y_test'] = sorted(i for g in pools['test'] for i in g)
        if len(splits['y_test']) != sum(r['partition'] == 'test' for r in rows):
            raise AssertionError('full official test not preserved')
    else:
        if settings.get('split_mode', 'grouped_subset') != 'grouped_subset':
            raise ValueError('unknown split_mode')
        splits['a_train'], remainder = select_groups(pools['train'], rows, counts['a_train'], rng)
        splits['b_train'], _ = select_groups(remainder, rows, counts['b_train'], rng)
        splits['a_val'], remainder = select_groups(pools['dev'], rows, counts['a_val'], rng)
        splits['b_val'], _ = select_groups(remainder, rows, counts['b_val'], rng)
        splits['y_test'], _ = select_groups(pools['test'], rows, counts['y_test'], rng)
    group_id = {i: str(min(group)) for group in groups for i in group}
    assignments = {}
    for split, indices in splits.items():
        for i in indices:
            if group_id[i] in assignments and assignments[group_id[i]] != split:
                raise AssertionError('source/text group leakage')
            assignments[group_id[i]] = split
    audit = {'official_counts': dict(Counter(r['partition'] for r in rows)),
             'removed_cross_partition_ids': [rows[i]['id'] for i in removed],
             'duplicate_groups': sum(len(g) > 1 for g in groups),
             'conflicting_intent_groups': sum(len({rows[i]['intent'] for i in g}) > 1 for g in groups),
             'grouping': 'union of official source id and NFKC-casefold-whitespace text',
             'priority': 'test > dev > train', 'group_disjoint': True,
             'split_seed': settings['split_seed'], 'requested_counts': counts,
             'split_mode': settings.get('split_mode', 'grouped_subset'),
             'a_train_fraction': settings.get('a_train_fraction'), 'a_val_fraction': settings.get('a_val_fraction')}
    return splits, group_id, audit

@lru_cache(maxsize=4)
def tokenizer_at(path):
    return XLMRobertaTokenizerFast.from_pretrained(path, local_files_only=True)

def prepare(context):
    raw = context.data_dir / 'raw'
    rows = [json.loads(line) for line in (raw / 'en-US.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if Counter(r['partition'] for r in rows) != Counter(train=11514, dev=2033, test=2974):
        raise ValueError('MASSIVE 1.0 en-US record count mismatch')
    if any(r['locale'] != 'en-US' for r in rows):
        raise ValueError('unexpected locale')
    parsed = [parse_slots(r) for r in rows]
    # Label ontology is defined from official train, not inferred from held-out Y.
    intents = sorted({r['intent'] for r in rows if r['partition'] == 'train'})
    slots = ['Other'] + sorted({s for r, (_, labels) in zip(rows, parsed) if r['partition'] == 'train' for s in labels if s != 'Other'})
    if len(intents) != 60 or len(slots) != 56:
        raise ValueError(f'unexpected label schema: {len(intents)} intents, {len(slots)} slot channels')
    intent_map, slot_map = {s: i for i, s in enumerate(intents)}, {s: i for i, s in enumerate(slots)}
    splits, groups, audit = split_records(rows, context.config['data'])
    directory = processed_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_at(str(raw / 'tokenizer'))
    length = context.config['data']['max_length']
    artifacts, records = [], {}
    for split, indices in splits.items():
        words = [parsed[i][0] for i in indices]
        encoded = tokenizer(words, is_split_into_words=True, padding='max_length', max_length=length, truncation=False)
        payload = {'input_ids': [], 'attention_mask': [], 'word_mask': [], 'slots_num': [],
                   'intent_num': [intent_map[rows[i]['intent']] for i in indices],
                   'ids': [rows[i]['id'] for i in indices]}
        for j, i in enumerate(indices):
            ids, mask = encoded['input_ids'][j], encoded['attention_mask'][j]
            if len(ids) != length:
                raise ValueError(f'source id {rows[i]["id"]} exceeds max_length; raise length in new experiment, no silent truncation')
            alignment = encoded.word_ids(j)
            labels, first, previous = [], [], None
            for word in alignment:
                active = word is not None and word != previous
                labels.append(slot_map[parsed[i][1][word]] if active else -100)
                first.append(active)
                previous = word
            payload['input_ids'].append(ids)
            payload['attention_mask'].append(mask)
            payload['word_mask'].append(first)
            payload['slots_num'].append(labels)
        payload = {k: torch.tensor(v, dtype=torch.bool if k == 'word_mask' else torch.long) if k != 'ids' else v for k, v in payload.items()}
        path = directory / f'{split}.pt'
        torch.save(payload, path)
        artifacts.append(path)
        records[split] = {'sha256': sha256_file(path), 'ids': payload['ids'],
                          'groups': [groups[i] for i in indices], 'count': len(indices),
                          'intent_counts': dict(Counter(rows[i]['intent'] for i in indices)),
                          'first_subword_count': int(payload['word_mask'].sum()), 'truncated_examples': 0}
    schema = write_json(directory / 'labels.json', {'intents': intents, 'slots': slots, 'vocab_size': len(tokenizer),
        'schema_source': 'official train labels; stable sorted type encoding, Other=0', 'slot_encoding': 'type, no BIO head; first subword only'})
    artifacts.append(schema)
    artifacts.append(write_json(directory / 'split_manifest.json', {
        'identity': context.identity, 'records': records, 'audit': audit, 'max_length': length,
        'raw_sha256': sha256_file(raw / 'en-US.jsonl'), 'labels_sha256': sha256_file(schema),
        'tokenizer_revision': TOKENIZER_REVISION, 'tokenizer_sha256': PINNED_FILES['tokenizer/tokenizer.json'],
        'scope': ('all decontaminated official train/dev grouped into A/B; full official en-US test'
                  if context.config['data'].get('split_mode') == 'official_grouped_full' else
                  'intent-stratified source/text-group-disjoint small en-US subset; selected test subset only'),
    }))
    return artifacts

def data_identity(context):
    directory = processed_dir(context)
    manifest = read_json(directory / 'split_manifest.json')
    if manifest['identity'] != context.identity:
        raise ValueError('processed code/config identity mismatch')
    if manifest['labels_sha256'] != sha256_file(directory / 'labels.json'):
        raise ValueError('label schema changed')
    return sha256_file(directory / 'split_manifest.json')

def load_split(context, split, auxiliary=False):
    data_identity(context)
    directory = processed_dir(context)
    manifest = read_json(directory / 'split_manifest.json')
    path = directory / f'{split}.pt'
    if sha256_file(path) != manifest['records'][split]['sha256']:
        raise ValueError('processed split checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not auxiliary:
        payload.pop('slots_num')
    return payload

class Records(Dataset):
    def __init__(self, payload):
        self.payload = payload
    def __len__(self):
        return len(self.payload['intent_num'])
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.payload.items()}

def make_loader(context, split, seed, batch_size, shuffle=False, auxiliary=False):
    return DataLoader(Records(load_split(context, split, auxiliary)), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)

def applicable_recipes(context, variant):
    return [r for r in context.config['refiner']['recipes'] if variant == 'multi_task' or r in ('embedding', 'embedding_matched')]
