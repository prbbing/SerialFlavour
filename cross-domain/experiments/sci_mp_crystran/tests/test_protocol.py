"""Scientific contracts: author fidelity, masks, grouping, main-only downstream."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.sci_mp_crystran.data import collate, unique_structures, Crystals
from experiments.sci_mp_crystran.model import CrystalTransformer, Readout
from experiments.sci_mp_crystran.refine import features, normalization, FrozenDataset

SETTINGS = {'feature_size': 32, 'num_heads': 4, 'num_layers': 2, 'dim_feedforward': 64,
            'dropout': 0.0, 'st_head_hidden': 128}


@pytest.mark.parametrize('variant,file', [('single_task','model.py'),('multi_task','model_mt_2.py')])
def test_author_forward_equivalence(variant, file):
    spec = importlib.util.spec_from_file_location('author_'+variant, Path(__file__).parent / 'reference' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(3)
    author = module.CrystalTransformer(32, 2, 4, 64, dropout=0).eval()
    model = CrystalTransformer(SETTINGS, variant).eval()
    missing, unexpected = model.load_state_dict(author.state_dict(), strict=False)
    assert not missing
    assert set(unexpected) == {'positional_encoding', 'coord_diff_embed.weight', 'coord_diff_embed.bias'}
    z = torch.tensor([[1, 8, 0], [14, 8, 8]])
    coords = torch.randn(2, 3, 3)
    mask = z == 0
    atom = torch.nn.functional.one_hot((z-1).clamp_min(0),100).float().masked_fill(mask[...,None],0)
    with torch.inference_mode():
        expected = author(atom, coords, mask)
        actual = model(z, coords, mask)
    if variant == 'single_task':
        torch.testing.assert_close(expected.squeeze(-1), actual['main'])
    else:
        torch.testing.assert_close(expected[0].squeeze(-1), actual['auxiliary'])
        torch.testing.assert_close(expected[1].squeeze(-1), actual['main'])


def test_padding_cannot_change_valid_outputs():
    model = CrystalTransformer(SETTINGS).eval()
    z = torch.tensor([[1,8]])
    coords = torch.randn(1,2,3)
    with torch.inference_mode():
        expected = model(z,coords,z==0)
        actual = model(torch.tensor([[1,8,0,0]]), torch.cat([coords,torch.randn(1,2,3)*100],1),
                       torch.tensor([[False,False,True,True]]))
    torch.testing.assert_close(expected['main'],actual['main'])
    torch.testing.assert_close(expected['H'],actual['H'][:,:2])


def test_equivalent_structures_removed_before_splitting():
    a = Structure(Lattice.cubic(5), ['Na','Cl'], [[0,0,0],[.5,.5,.5]])
    b = a.copy()
    b.translate_sites([0,1],[.2,.1,.3])
    c = Structure(Lattice.cubic(5), ['Na','Cl'], [[0,0,0],[.05,.05,.05]])
    rows = [{'id': sid, 'structure': s.as_dict(), 'targets':[1.,-1.]} for sid,s in [('mp-1',a),('mp-2',b),('mp-3',c),('mp-1',c)]]
    kept,rejected = unique_structures(rows)
    assert len(kept)==2
    assert rejected == {'equivalent_structure':1,'duplicate_id':1}


def test_auxiliary_truth_forbidden_on_b_and_y():
    for split in ('b_train','b_val','y_test'):
        with pytest.raises(ValueError, match='auxiliary truth'):
            Crystals(None,split,True)


def fixture_cache():
    return {'H':torch.randn(8,4,32),'mask':torch.zeros(8,4,dtype=torch.bool), 'g':torch.randn(8,32),
            'native':torch.randn(8),'main':torch.randn(8),'aux_prediction':torch.arange(8).float(),
            'aux_hidden':torch.randn(8,32)}


def test_shuffle_retrains_with_reproducible_predicted_features():
    saved = fixture_cache()
    a = features(saved,'embedding_aux_shuffle',1001)
    b = features(saved,'embedding_aux_shuffle',1001)
    torch.testing.assert_close(a['extra'],b['extra'])
    assert torch.equal(a['extra'].sort().values,saved['aux_prediction'])
    assert not torch.equal(a['extra'],saved['aux_prediction'])
    for key in ('H','g','native','main'):
        torch.testing.assert_close(a[key],saved[key])


def test_b_statistics_and_native_epoch_zero_capacity_match():
    saved = fixture_cache()
    parameters = []
    for recipe in ('embedding','embedding_capacity','embedding_aux','embedding_aux_shuffle','embedding_hidden'):
        values = features(saved,recipe,1001)
        stats = normalization(values)
        dataset = FrozenDataset(values,stats)
        torch.testing.assert_close(dataset.values['g'].mean(0),torch.zeros(32),atol=1e-6,rtol=0)
        model = Readout(32,[64,32],1.2,0.8)
        torch.testing.assert_close(model(dataset.values),saved['native'])
        parameters.append(sum(p.numel() for p in model.parameters()))
    assert len(set(parameters))==1
@pytest.mark.parametrize('chunk_bytes', [1,4,17,64])
def test_historical_nonfinite_reader_preserves_strings(chunk_bytes):
    import io, json
    from experiments.sci_mp_crystran.data import NonfiniteJSONReader
    raw = b'{"data":[[NaN,Infinity,-Infinity,"NaN, [ Infinity,", "escaped \\\" NaN"]],"tail":123}'
    reader = NonfiniteJSONReader(io.BytesIO(raw),chunk_bytes)
    result = json.loads(reader.read())
    assert result['data'][0][:3] == [None,None,None]
    assert result['data'][0][3:] == json.loads(raw)['data'][0][3:]
    assert result['tail'] == 123

def test_mt_main_only_has_same_parameters_and_main_loss_no_aux_gradient():
    torch.manual_seed(7)
    matched = CrystalTransformer(SETTINGS, 'mt_main_only')
    torch.manual_seed(7)
    multi = CrystalTransformer(SETTINGS, 'multi_task')
    for name, value in matched.state_dict().items():
        torch.testing.assert_close(value,multi.state_dict()[name])
    z = torch.tensor([[1,8,8]])
    out = matched(z,torch.randn(1,3,3),z==0)
    out['main'].square().mean().backward()
    assert all(p.grad is None for p in matched.output_linear1.parameters())
    assert all(p.grad is not None for p in matched.output_linear2.parameters())
    assert matched.atom_embed.weight.grad is not None