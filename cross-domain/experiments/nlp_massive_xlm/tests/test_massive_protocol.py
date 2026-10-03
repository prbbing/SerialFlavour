"""Protocol, author-interface, padding, and identity checks; no network needed."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.nlp_massive_xlm.data import parse_slots, split_records, load_split, applicable_recipes
from experiments.nlp_massive_xlm.model import Upstream, Readout
from experiments.nlp_massive_xlm.refine import features
from experiments.nlp_massive_xlm.training import upstream_loss
from experiments.nlp_massive_xlm.evaluate import spans, metrics
from pipeline.io import write_json, sha256_file
from pipeline.context import load_context
from pipeline.units import enumerate_units


def tiny_model(multi=True):
    return Upstream({'hidden_size': 8, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                     'intermediate_size': 16, 'hidden_dropout_prob': 0.0, 'attention_probs_dropout_prob': 0.0},
                    {'vocab_size': 101, 'intents': [str(i) for i in range(60)], 'slots': [str(i) for i in range(56)]}, multi)


def sample_batch():
    return {'input_ids': torch.tensor([[0, 4, 5, 2, 1], [0, 6, 2, 1, 1]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
            'word_mask': torch.tensor([[False, True, True, False, False], [False, True, False, False, False]]),
            'intent_num': torch.tensor([0, 1])}


def test_annotation_not_used_as_input_and_plain_type_schema():
    row = {'id': 'x', 'utt': 'weather in new york tomorrow',
           'annot_utt': 'weather in [place_name : new york] [date : tomorrow]'}
    words, labels = parse_slots(row)
    assert words == row['utt'].split()
    assert labels == ['Other', 'Other', 'place_name', 'place_name', 'date']
    row['utt'] = 'different text'
    with pytest.raises(ValueError, match='mismatch'):
        parse_slots(row)


def test_source_and_duplicate_text_binding_protects_test():
    rows = []
    for partition, count in [('train', 10), ('dev', 8), ('test', 8)]:
        for j in range(count):
            rows.append({'id': f'{partition}{j}', 'partition': partition, 'intent': str(j % 2), 'utt': f'{partition} word {j}'})
    rows[0]['utt'] = ' TEST  WORD 0 '
    rows[1]['id'] = 'dev1'
    splits, groups, audit = split_records(rows, {'split_seed': 17, 'split_counts': dict(a_train=3, b_train=3, a_val=2, b_val=2, y_test=4)})
    assert rows[0]['id'] in audit['removed_cross_partition_ids']
    assert rows[1]['id'] in audit['removed_cross_partition_ids']
    sets = [{groups[i] for i in indices} for indices in splits.values()]
    assert all(not a & b for j, a in enumerate(sets) for b in sets[j+1:])
    assert all(rows[i]['partition'] == 'test' for i in splits['y_test'])
    assert splits == split_records(rows, {'split_seed': 17, 'split_counts': dict(a_train=3, b_train=3, a_val=2, b_val=2, y_test=4)})[0]


def test_single_task_never_accepts_slot_truth_and_only_intent_trains():
    torch.manual_seed(7)
    model = tiny_model(False)
    batch = sample_batch()
    objective, n = upstream_loss(model, batch, False)
    objective.backward()
    assert n == 2
    assert any(p.grad is not None for p in model.network.intent_classifier.parameters())
    assert any(p.grad is not None for p in model.network.xlmr.encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.network.slot_classifier.parameters())
    with pytest.raises(ValueError, match='must not expose'):
        upstream_loss(model, {**batch, 'slots_num': torch.zeros(2, 5, dtype=torch.long)}, False)


def test_multi_task_slots_are_supervised_only_at_first_subword():
    model = tiny_model()
    batch = sample_batch()
    labels = torch.full((2, 5), -100)
    labels[batch['word_mask']] = torch.tensor([2, 3, 4])
    objective, _ = upstream_loss(model, {**batch, 'slots_num': labels}, True)
    objective.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.network.slot_classifier.parameters())


def test_export_is_one_official_encoder_forward_with_identical_logits():
    model = tiny_model().eval()
    batch = sample_batch()
    calls = []
    hook = model.network.xlmr.register_forward_hook(lambda *args: calls.append(1))
    with torch.inference_mode():
        output = model(batch)
    hook.remove()
    assert len(calls) == 1
    with torch.inference_mode():
        _, (intent, slot) = model.network(batch['input_ids'], batch['attention_mask'], None, None)
    assert torch.equal(output['logits'], intent)
    assert torch.equal(output['slot_logits'], slot)
    assert output['embedding'].shape == (2, 5, 8)


def test_matched_capacity_native_initialization_and_padding_invariance():
    batch = sample_batch()
    payload = {**batch, 'embedding': torch.randn(2, 5, 8), 'slot_logits': torch.randn(2, 5, 56),
               'native_logits': torch.randn(2, 60)}
    models = []
    for recipe in ('embedding_matched', 'aux_prediction'):
        torch.manual_seed(29)
        model = Readout(64, 60, 4).eval()
        inp = {**payload, 'features': features(payload, recipe)}
        assert torch.equal(model(inp), payload['native_logits'])
        # Give the residual nonzero weights so padding invariance isn't vacuous.
        with torch.no_grad():
            model.output.weight.normal_()
        before = model(inp)
        altered = inp['features'].clone()
        altered[batch['attention_mask'] == 0] = 10000
        assert torch.equal(before, model({**inp, 'features': altered}))
        models.append(model)
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())


def test_processed_truth_stripping_and_tamper_rejection(tmp_path):
    context = SimpleNamespace(data_dir=tmp_path, identity='identity', config={'experiment': 'test'})
    directory = tmp_path / 'processed/test'
    directory.mkdir(parents=True)
    labels = write_json(directory / 'labels.json', {'slots': ['Other']})
    path = directory / 'b_train.pt'
    torch.save({**sample_batch(), 'slots_num': torch.zeros(2, 5), 'ids': ['a', 'b']}, path)
    write_json(directory / 'split_manifest.json', {'identity': 'identity', 'labels_sha256': sha256_file(labels),
                                                 'records': {'b_train': {'sha256': sha256_file(path)}}})
    assert 'slots_num' not in load_split(context, 'b_train')
    assert 'slots_num' in load_split(context, 'b_train', auxiliary=True)
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        load_split(context, 'b_train')


def test_spans_and_accuracy_fixed_class_macro_f1():
    assert spans([0, 3, 3, 0, 4]) == {(3, 1, 3), (4, 4, 5)}
    class Constant(torch.nn.Module):
        def forward(self, batch):
            out = torch.zeros(2, 60)
            out[:, 0] = 1
            return out
    result = metrics(Constant(), [sample_batch()], torch.device('cpu'))
    assert result['accuracy'] == 0.5
    assert result['classes_present'] == 2
    assert result['macro_f1_60'] == pytest.approx((2/3) / 60)


def test_generic_adapter_loading_units_and_st_recipes():
    context = load_context(ROOT / 'experiments/nlp_massive_xlm/config/smoke.json', ROOT)
    for kind in ('data', 'model', 'training', 'refine', 'evaluate', 'analysis'):
        assert context.module(kind).__name__ == f'experiments.nlp_massive_xlm.{kind}'
    assert applicable_recipes(context, 'single_task') == ['embedding', 'embedding_matched']
    assert len(enumerate_units(context)) == 12
