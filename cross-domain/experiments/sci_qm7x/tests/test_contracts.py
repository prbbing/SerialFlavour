"""Offline protocol contracts only. No real download, training or optimizer run."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

DOMAIN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DOMAIN))
from pipeline.context import Context
from pipeline.io import read_json, write_json, sha256_file
from pipeline.units import enumerate_units, unit_filters
from experiments.sci_qm7x import data
from experiments.sci_qm7x.common import SPLITS, DUPLICATES_MD5, applicable_recipes, assert_prepared
from experiments.sci_qm7x.model import MolecularModel, Readout
from experiments.sci_qm7x.refine import features, normalization, FrozenDataset, shuffle_atoms, load_cache
from experiments.sci_qm7x.analysis import hierarchical, paired
from experiments.sci_qm7x.evaluate import matrix_keys

MODEL = {'width': 16, 'head_hidden': 8, 'num_gaussians': 6, 'num_interactions': 2, 'cutoff': 5.0}
REFINER = {'hidden': [12, 6], 'geometry_bins': 4}


def configuration():
    return read_json(DOMAIN / 'experiments/sci_qm7x/config/cluster_50k.json')


@pytest.fixture
def context(tmp_path, monkeypatch):
    config = configuration()
    config['data_root'] = str(tmp_path / 'raw')
    config['output_root'] = str(tmp_path / 'out')
    config['log_root'] = str(tmp_path / 'logs')
    config['data']['sources'] = [{'path': 'tiny.hdf5'}]
    config['data']['sizes'] = {split: 3 for split in SPLITS}
    config['data']['max_conformations_per_parent'] = 3
    config['model'] = dict(MODEL)
    config['runtime']['device'] = 'cpu'
    ctx = Context(config, DOMAIN, tmp_path / 'config.json')
    ctx.data_dir.mkdir(parents=True)
    ctx.output_dir.mkdir(parents=True)
    duplicate = ctx.data_dir / 'DupMols.dat'
    duplicate.write_text('Geom-m1-i1-c1-opt\n', encoding='utf-8')
    original = data.file_digest
    monkeypatch.setattr(data, 'file_digest', lambda path, algorithm='md5': DUPLICATES_MD5 if Path(path) == duplicate else original(path, algorithm))
    with h5py.File(ctx.data_dir / 'tiny.hdf5', 'w') as output:
        for parent in range(1, 26):
            group = output.create_group(str(parent))
            for name in [f'Geom-m{parent}-i1-c1-opt', f'Geom-m{parent}-i1-c1-1',
                         f'Geom-m{parent}-i2-c2-opt', f'Geom-m{parent}-i2-c2-d2']:
                row = group.create_group(name)
                row['atNUM'] = np.array([6, 1, 1, 1, 1])
                row['atXYZ'] = np.arange(15, dtype=float).reshape(5, 3) / 4
                row['eH'] = np.array([-5.0 - parent / 100])
                row['hCHG'] = np.array([-0.4, 0.1, 0.1, 0.1, 0.1])
                row['atPOL'] = np.array([10., 2., 2., 2., 2.])
    return ctx


def sample():
    torch.manual_seed(7)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return {'z': torch.tensor([[6, 1, 17, 0], [16, 1, 0, 0]]),
            'coords': torch.randn(2, 4, 3), 'mask': mask, 'main': torch.tensor([-4., -6.])}


def saved_cache():
    torch.manual_seed(9)
    mask = sample()['mask']
    return {'H': torch.randn(2, 4, 16), 'g': torch.randn(2, 16), 'native': torch.tensor([-5., -6.]),
            'aux_prediction': torch.randn(2, 4, 2), 'aux_hidden': torch.randn(2, 4, 8),
            'coords': sample()['coords'], 'mask': mask, 'main': torch.tensor([-4., -7.]),
            'ids': ['Geom-m1-i1-c1-opt', 'Geom-m2-i1-c1-opt']}


def test_official_field_schema_and_units():
    assert data.FIELDS == {'z': 'atNUM', 'coords': 'atXYZ', 'main': 'eH', 'charge': 'hCHG', 'polarizability': 'atPOL'}
    assert data.UNITS['main'] == 'eV'
    assert data.UNITS['polarizability'] == 'bohr^3'
    assert data.parse_geometry('Geom-m7-i2-c3-opt')['r'] == '7'
    assert data.parse_geometry('Geom-m7-i2-c3-d100')['u'] == 'd100'
    with pytest.raises(ValueError):
        data.parse_geometry('Geom-m7-i2-c3-101')


def test_parent_assignment_cannot_separate_stereo_conformer_or_displacement():
    parents = [f"m{data.parse_geometry(name)['r']}" for name in ['Geom-m1-i1-c1-opt', 'Geom-m1-i2-c2-50', 'Geom-m1-i2-c3-d5']]
    assert len(set(parents)) == 1
    groups = data.partition_parents([f'm{i}' for i in range(1, 31)], {s: 3 for s in SPLITS}, 73)
    assert len(set().union(*(set(groups[s]) for s in SPLITS))) == 30
    assert all(not set(groups[a]) & set(groups[b]) for a in SPLITS for b in SPLITS if a != b)
    assert groups == data.partition_parents(reversed([f'm{i}' for i in range(1, 31)]), {s: 3 for s in SPLITS}, 73)


def test_duplicate_family_is_removed_with_all_displacements(context):
    candidates, scan = data.scan_candidates(context)
    assert scan['official_duplicate_families'] == 1
    assert scan['rejections']['official_duplicate_family'] == 2
    assert all('Geom-m1-i1-c1-' not in row[1] for rows in candidates.values() for row in rows)
    assert all(len(rows) <= 3 for rows in candidates.values())
    assert candidates == data.scan_candidates(context)[0]


def test_equilibrium_filter_and_insufficient_budget_are_explicit(context):
    context.config['data']['equilibrium_only'] = True
    candidates, _ = data.scan_candidates(context)
    assert all(row[1].endswith('-opt') for rows in candidates.values() for row in rows)
    groups = data.partition_parents(candidates, {s: 100 for s in SPLITS}, 2)
    with pytest.raises(ValueError, match='capacity'):
        data.select_candidates(candidates, groups, {s: 100 for s in SPLITS})


def test_prepare_parent_counts_train_only_norm_and_no_B_truth(context):
    data.prepare(context)
    manifest = assert_prepared(context)
    assert manifest['structure_counts'] == {s: 3 for s in SPLITS}
    assert sum(manifest['group_counts'].values()) == 25
    saved = torch.load(context.output_dir / 'prepared.pt', weights_only=True)
    norm = read_json(context.output_dir / 'target_normalization.json')
    a = manifest['indices']['a_train']
    assert norm['main_mean'] == pytest.approx(saved['main'][a].double().mean().item())
    assert norm['aux_mean'] == pytest.approx(saved['auxiliary'][a][saved['mask'][a]].double().mean(0).tolist())
    batch = next(iter(data.make_loader(context, 'b_train')))
    assert 'auxiliary' not in batch
    with pytest.raises(ValueError, match='truth forbidden'):
        data.make_loader(context, 'b_train', auxiliary=True)
    with pytest.raises(ValueError, match='explicit diagnostics'):
        data.make_loader(context, 'y_test', auxiliary=True)
    assert 'auxiliary' in next(iter(data.make_loader(context, 'y_test', auxiliary=True, diagnostics=True)))


def test_modified_prepared_tensors_fail_identity(context):
    data.prepare(context)
    with (context.output_dir / 'prepared.pt').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='SHA256'):
        assert_prepared(context)


def test_schema_rejects_missing_target_instead_of_deriving_homo():
    row = {'atNUM': np.array([6]), 'atXYZ': np.zeros((1, 3)), 'hCHG': np.zeros(1), 'atPOL': np.ones(1),
           'eL': np.array([0.]), 'HLgap': np.array([5.])}
    with pytest.raises(ValueError, match='eH'):
        data.read_structure(row, 'Geom-m1-i1-c1-opt', 23)


def test_ST_MT_architecture_and_main_initialization_match():
    torch.manual_seed(3)
    st = MolecularModel(MODEL, 'single_task')
    torch.manual_seed(3)
    mt = MolecularModel(MODEL, 'multi_task')
    assert st.state_dict().keys() == mt.state_dict().keys()
    assert all(torch.equal(st.state_dict()[k], mt.state_dict()[k]) for k in st.state_dict())
    assert torch.equal(st(sample())['main'], mt(sample())['main'])
    assert sum(p.numel() for p in st.parameters()) == sum(p.numel() for p in mt.parameters())


def test_model_padding_translation_rotation_and_permutation():
    model = MolecularModel(MODEL).eval()
    batch = sample()
    base = model(batch)
    changed = {**batch, 'coords': batch['coords'] + 7.0}
    assert torch.allclose(base['main'], model(changed)['main'], atol=2e-6)
    padded = {**batch, 'coords': batch['coords'].clone()}
    padded['coords'][~batch['mask']] = 1e4
    assert torch.allclose(base['main'], model(padded)['main'], atol=2e-6)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert torch.allclose(base['main'], model({**batch, 'coords': batch['coords'] @ rotation})['main'], atol=2e-6)
    order = torch.tensor([2, 0, 3, 1])
    permuted = {k: value[:, order] if k in ('z', 'coords', 'mask') else value for k, value in batch.items()}
    assert torch.allclose(base['main'], model(permuted)['main'], atol=2e-6)
    assert torch.allclose(base['auxiliary'][:, order], model(permuted)['auxiliary'], atol=2e-6)
    assert (base['H'][~batch['mask']] == 0).all()


@pytest.mark.parametrize('recipe', ['embedding', 'embedding_capacity', 'embedding_aux', 'embedding_hidden', 'embedding_aux_shuffle'])
def test_native_zero_residual_and_equal_readout_architecture(recipe):
    saved = saved_cache()
    values = features(saved, recipe, 1101, 8)
    stats = normalization(values)
    dataset = FrozenDataset(values, stats)
    batch = {key: torch.stack([dataset[i][key] for i in range(len(dataset))]) for key in dataset[0]}
    model = Readout(MODEL, REFINER, 2.0)
    assert torch.equal(model(batch), saved['native'])
    assert model.node[0].in_features == 16 + 8 + 4
    assert 'auxiliary' not in batch


def test_shuffle_is_within_molecule_preserves_paired_local_channels():
    values = torch.tensor([[[1., 11.], [2., 12.], [3., 13.], [0., 0.]],
                           [[4., 14.], [5., 15.], [0., 0.], [0., 0.]]])
    mask = sample()['mask']
    shuffled = shuffle_atoms(values, mask, ['a', 'b'], 100)
    assert torch.equal(shuffled, shuffle_atoms(values, mask, ['a', 'b'], 100))
    assert torch.equal(shuffled[~mask], values[~mask])
    for i in range(2):
        assert sorted(map(tuple, shuffled[i][mask[i]].tolist())) == sorted(map(tuple, values[i][mask[i]].tolist()))
        assert not torch.equal(shuffled[i][mask[i]], values[i][mask[i]])


def test_normalization_ignores_padding_and_never_uses_Y():
    saved = saved_cache()
    values = features(saved, 'embedding_aux', 3, 8)
    stats = normalization(values)
    changed = {k: v.clone() for k, v in values.items()}
    changed['H'][~values['mask']] = 1e8
    changed['extra'][~values['mask']] = -1e9
    altered = normalization(changed)
    assert all(torch.equal(stats[k], altered[k]) for k in stats)
    assert torch.equal(stats['H_mean'], values['H'][values['mask']].double().mean(0).float())


def test_hierarchical_and_paired_ddof_one():
    rows = [{'upstream_seed': us, 'downstream_seed': ds, 'mae_eV': float(us + ds)} for us in [1, 3] for ds in [10, 20]]
    result = hierarchical(rows, [1, 3], [10, 20])
    assert result['mean'] == 17.0
    assert result['sample_sd'] == pytest.approx(2 ** 0.5)
    assert result['ddof'] == 1 and result['n_upstream_seeds'] == 2
    with pytest.raises(ValueError, match='missing'):
        hierarchical(rows[:-1], [1, 3], [10, 20])
    indexed = [dict(row, variant='multi_task', recipe=recipe, mae_eV=row['mae_eV'] - gain)
               for row in rows for recipe, gain in [('embedding_capacity', 0.), ('embedding_aux', 0.5)]]
    effect = paired(indexed, 'multi_task', 'embedding_capacity', 'multi_task', 'embedding_aux', [1, 3], [10, 20])
    assert effect['mean'] == 0.5 and effect['sample_sd'] == 0.


def test_full_matrix_units_filters_and_cache_missing_is_fatal(context):
    formal = Context(configuration(), DOMAIN, Path('cluster_50k.json'))
    units = enumerate_units(formal)
    assert len(units) == 138 and sum(u.startswith('refine:') for u in units) == 125
    assert len(matrix_keys(formal)) == 130
    ablation = read_json(DOMAIN / 'experiments/sci_qm7x/config/cluster_50k_st_mt_ablation.json')
    both = Context(ablation, DOMAIN, Path('cluster_50k_st_mt_ablation.json'))
    assert len(enumerate_units(both)) == 198 and len(matrix_keys(both)) == 185
    assert applicable_recipes(formal, 'single_task') == ['embedding', 'embedding_capacity']
    assert unit_filters('refine:multi_task:11:embedding_aux:101')['upstream_seed'] == 11
    with pytest.raises(FileNotFoundError):
        load_cache(context, 'multi_task', 11, 'b_train')


def test_cluster_dry_run_has_no_tensor_import_or_output_side_effects(tmp_path):
    config = configuration()
    config['output_root'] = str(tmp_path / 'never-output')
    config['data_root'] = str(tmp_path / 'never-data')
    path = tmp_path / 'config.json'
    write_json(path, config)
    script = DOMAIN / 'experiments/sci_qm7x/scripts/run_cluster.py'
    code = "import runpy,sys; sys.argv=" + repr([str(script), '--config', str(path), '--dry-run']) + "; runpy.run_path(" + repr(str(script)) + ",run_name='__main__'); assert 'torch' not in sys.modules; assert 'h5py' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['total_units'] == 138
    assert not (tmp_path / 'never-output').exists() and not (tmp_path / 'never-data').exists()


def test_split_manifest_corruption_is_rejected(context):
    data.prepare(context)
    path = context.output_dir / 'split_manifest.json'
    split = read_json(path)
    split['indices']['b_train'][0] = split['indices']['a_train'][0]
    write_json(path, split)
    with pytest.raises(ValueError, match='split manifest SHA256'):
        assert_prepared(context)

def test_opt_in_archive_download_never_decompresses_or_claims_prepared(context, monkeypatch):
    source = context.data_dir / 'tiny.hdf5'
    source.unlink()
    context.config['data']['allow_download'] = True
    content = b'offline archive fixture, not a real download'
    import hashlib
    checksum = hashlib.md5(content).hexdigest()
    context.config['data']['download_files'] = [{'name': '8000.xz', 'url': 'https://zenodo.org/records/4288677/files/8000.xz', 'md5': checksum}]
    monkeypatch.setattr(data, 'bounded_download', lambda url, target, maximum, expected_md5: Path(target).write_bytes(content))
    artifacts = data.download(context)
    manifest = read_json(context.output_dir / 'download_manifest.json')
    assert manifest['pending_local_hdf5'] == [str(source.resolve())]
    assert manifest['automatic_decompression'] is False
    assert context.data_dir / '8000.xz' in artifacts
    assert not source.exists()
    with pytest.raises(FileNotFoundError, match='prepare requires'):
        data.prepare(context)

def test_resource_plan_keeps_budgets_and_only_one_seed_cache():
    from experiments.sci_qm7x.workflow import plan
    from experiments.sci_qm7x.common import validate_config
    formal = Context(configuration(), DOMAIN, Path('cluster_50k.json'))
    description = plan(formal)
    assert description['variants'] == ['multi_task']
    assert description['structure_budget'] == 50000
    assert description['upstream_jobs'] == 5 and description['refine_jobs'] == 125
    assert description['max_active_seeds'] == 1 and description['stream_Y']
    assert description['cache_splits'] == ['b_train','b_val']
    assert description['peak_regenerable_cache_bytes_estimate'] == 124760000
    assert not description['save_predictions']
    cfg = configuration()
    cfg['upstream']['variants'] = ['single_task']
    validate_config(cfg)
    cfg['upstream']['variants'] = []
    with pytest.raises(ValueError,match='subset'):
        validate_config(cfg)


def test_streamed_Y_equals_same_tensor_cached_score_without_fitting_Y():
    from experiments.sci_qm7x.evaluate import stream_rows, regression
    from experiments.sci_qm7x.training import physical_output
    from experiments.sci_qm7x.refine import frozen_loader
    upstream = MolecularModel(MODEL).eval().requires_grad_(False)
    norm = {'main_mean':-5.,'main_std':2.,'aux_mean':[0.,1.], 'aux_std':[1.,2.]}
    batch = sample() | {'id':['Geom-m1-i1-c1-opt','Geom-m2-i1-c1-opt']}
    with torch.inference_mode():
        output = physical_output(upstream(batch),norm)
    saved = {key:output[source] for key,source in [('H','H'),('g','g'),('native','main'),
             ('aux_prediction','auxiliary'),('aux_hidden','aux_hidden')]}
    saved.update({key:batch[key] for key in ('coords','mask','main')})
    saved['ids'] = batch['id']
    candidates, expected, stats_before = [],{},[]
    for recipe in ('embedding','embedding_capacity','embedding_aux','embedding_hidden','embedding_aux_shuffle'):
        # Stand-in B-only training tensor differs from Y and is the sole norm source.
        training = copy.deepcopy(saved)
        training['H'] += 3.
        stats = normalization(features(training,recipe,1101,8))
        stats_before.append({key:value.clone() for key,value in stats.items()})
        model = Readout(MODEL,REFINER,2.).eval().requires_grad_(False)
        torch.nn.init.normal_(model.regression[-1].weight,std=.02)
        candidate = {'recipe':recipe,'downstream_seed':101,'model':model,'stats':stats}
        candidates.append(candidate)
        values = features(saved,recipe,3101,8)
        with torch.inference_mode():
            prediction = torch.cat([model(part) for part in frozen_loader(values,stats,1)])
        expected[(recipe,101)] = regression(saved['main'],prediction)
    one_row_batches = [{key:value[i:i+1] for key,value in batch.items()} for i in range(2)]
    rows,diagnostic,payload = stream_rows(upstream,candidates,one_row_batches,norm,torch.device('cpu'),8,batch['id'])
    assert diagnostic is None and payload is None
    assert len(rows) == 6 and sum(r['recipe']=='native' for r in rows) == 1
    for row in rows:
        key = row['recipe'],row['downstream_seed']
        reference = regression(saved['main'],saved['native']) if key[0]=='native' else expected[key]
        for metric in ('mae_eV','rmse_eV','bias_eV'):
            assert row[metric] == pytest.approx(reference[metric],abs=2e-6)
    assert all(torch.equal(c['stats'][key],before[key]) for c,before in zip(candidates,stats_before) for key in before)
    # Truth affects scoring only when explicitly requested; it is never a readout feature.
    with_truth = [dict(part,auxiliary=torch.full((1,4,2),100.)) for part in one_row_batches]
    diagnosed,report,_ = stream_rows(upstream,candidates,with_truth,norm,torch.device('cpu'),8,batch['id'],diagnostics=True)
    assert diagnosed == rows and report['valid_atoms'] == 5


def test_resource_cache_publication_is_B_only_and_cleanup_is_exact(context):
    from experiments.sci_qm7x import refine, workflow
    from experiments.sci_qm7x.common import expected_metadata,upstream_dir,cache_dir
    data.prepare(context)
    context.filters = {'variant':'multi_task','seed':11}
    model = MolecularModel(MODEL)
    directory = upstream_dir(context,'multi_task',11)
    directory.mkdir(parents=True)
    torch.save({'state_dict':model.state_dict(),'metadata':{**expected_metadata(context),'variant':'multi_task','seed':11}},directory/'best.pt')
    artifacts = refine.cache(context)
    cached = cache_dir(context,'multi_task',11)
    assert {path.name for path in artifacts} == {'b_train.pt','b_val.pt','cache_manifest.json'}
    assert set(read_json(cached/'cache_manifest.json')['cache_sha256']) == {'b_train.pt','b_val.pt'}
    assert not (cached/'y_test.pt').exists()
    for split in ('b_train','b_val'):
        assert len(load_cache(context,'multi_task',11,split)['main']) == 3
    with pytest.raises(ValueError,match='streamed'):
        load_cache(context,'multi_task',11,'y_test')
    unknown = cached/'author-note.txt'
    unknown.write_text('never delete me')
    records = workflow.cleanup_records(context,'multi_task',11)
    assert {Path(record['path']).name for record in records} == {'b_train.pt','b_val.pt'}
    assert unknown.exists() and (context.output_dir/'prepared.pt').exists()


def publish_offline_seed(context,seed):
    """Synthetic checkpoints + exact marker artifacts; no fit/optimizer/training."""
    from pipeline.io import artifact_record
    from pipeline.worker import marker_path
    from experiments.sci_qm7x import refine
    from experiments.sci_qm7x.common import expected_metadata,upstream_dir,cache_dir,refiner_dir
    def mark(unit,paths):
        write_json(marker_path(context,unit),{'status':'complete','identity':context.identity,
                   'artifacts':[artifact_record(path) for path in paths]})
    variant = 'multi_task'
    up = upstream_dir(context,variant,seed)
    up.mkdir(parents=True)
    metadata = {**expected_metadata(context),'variant':variant,'seed':seed}
    torch.save({'state_dict':MolecularModel(MODEL).state_dict(),'metadata':metadata,'epoch':2},up/'best.pt')
    write_json(up/'training_manifest.json',metadata)
    write_json(up/'history.json',[])
    (up/'history.csv').write_text('epoch,validation_metric')
    mark(f'upstream:{variant}:{seed}',[up/name for name in ('best.pt','training_manifest.json','history.json','history.csv')])
    context.filters = {'variant':variant,'seed':seed}
    cached = refine.cache(context)
    mark(f'cache:{variant}:{seed}',cached)
    b = load_cache(context,variant,seed,'b_train')
    norm = read_json(context.output_dir/'target_normalization.json')
    for recipe in applicable_recipes(context,variant):
        for ds in context.config['refiner']['seeds']:
            directory = refiner_dir(context,variant,seed,recipe,ds)
            directory.mkdir(parents=True)
            stats = normalization(features(b,recipe,ds+1000,MODEL['head_hidden']))
            torch.save(stats,directory/'input_normalization.pt')
            model = Readout(MODEL,context.config['refiner'],norm['main_std'])
            meta = {**expected_metadata(context),'variant':variant,'upstream_seed':seed,'recipe':recipe,'seed':ds,
                    'training_split':'b_train','selection_split':'b_val','auxiliary_truth_used':False,
                    'native_epoch0_candidate':True,'input_normalization_fit':'b_train_valid_atoms_and_molecules',
                    'upstream_checkpoint_sha256':sha256_file(up/'best.pt'),
                    'cache_manifest_sha256':sha256_file(cache_dir(context,variant,seed)/'cache_manifest.json'),
                    'input_normalization_sha256':sha256_file(directory/'input_normalization.pt')}
            torch.save({'state_dict':model.state_dict(),'metadata':meta,'epoch':0},directory/'best.pt')
            write_json(directory/'training_manifest.json',meta)
            write_json(directory/'history.json',[])
            (directory/'history.csv').write_text('epoch,validation_metric')
            mark(f'refine:{variant}:{seed}:{recipe}:{ds}',[directory/name for name in (
                 'best.pt','input_normalization.pt','training_manifest.json','history.json','history.csv')])


def test_seed_receipts_survive_cache_retirement_and_require_full_matrix(context):
    from pipeline.lifecycle import evaluate_one,retire_cache,valid_receipt,verify_seed_ready
    from pipeline.io import artifact_record
    from pipeline.worker import marker_path
    from experiments.sci_qm7x import workflow,evaluate
    from experiments.sci_qm7x.common import cache_dir,refiner_dir
    context.config['upstream']['seeds'] = [11,22]
    context.config['refiner']['seeds'] = [101,102]
    context.config['refiner'].update(REFINER)
    prepared = data.prepare(context)
    write_json(marker_path(context,'prepare'),{'status':'complete','identity':context.identity,
               'artifacts':[artifact_record(path) for path in prepared]})
    for seed in context.config['upstream']['seeds']:
        publish_offline_seed(context,seed)
        if seed == 11:
            missing = marker_path(context,'refine:multi_task:11:embedding_aux:101')
            original = read_json(missing)
            missing.unlink()
            with pytest.raises(ValueError,match='incomplete'):
                evaluate.evaluate_seed(context,'multi_task',11)
            assert not (context.output_dir/'evaluation').exists()
            write_json(missing,original)
        verify_seed_ready(context,'multi_task',seed)
        receipt = evaluate_one(context,'multi_task',seed)
        cached = cache_dir(context,'multi_task',seed)
        unknown = cached/'author-note.txt'
        unknown.write_text('keep this')
        assert receipt.exists() and valid_receipt(context,'multi_task',seed)
        retire_cache(context,'multi_task',seed)
        assert not (cached/'b_train.pt').exists() and not (cached/'b_val.pt').exists()
        assert unknown.exists() and (cached/'cache_manifest.json').exists()
        assert valid_receipt(context,'multi_task',seed)
        if seed == 11:
            with pytest.raises(FileNotFoundError):
                workflow.aggregate(context)
    artifacts = workflow.aggregate(context)
    assert all(path.exists() for path in artifacts)
    metrics = read_json(context.output_dir/'evaluation/metrics.json')
    assert len(metrics['rows']) == 22
    summary = read_json(context.output_dir/'analysis/summary.json')
    assert all('single_task' not in key for key in summary['comparisons'])
    assert not (context.output_dir/'evaluation/predictions').exists()
    assert not list((context.output_dir/'evaluation/seeds').rglob('predictions'))
    durable = refiner_dir(context,'multi_task',11,'embedding_aux',101)/'input_normalization.pt'
    durable.write_bytes(b'corrupt')
    assert not valid_receipt(context,'multi_task',11)
    with pytest.raises(ValueError,match='checksum'):
        workflow.aggregate(context)


def test_resource_generic_stage_loops_refuse_unfiltered_matrix(context):
    from experiments.sci_qm7x import training,refine
    for operation in (training.train,refine.cache,refine.train):
        with pytest.raises(ValueError,match='exact unit filters'):
            operation(context)
