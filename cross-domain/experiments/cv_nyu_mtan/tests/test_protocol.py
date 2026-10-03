"""Offline parity against pinned author code and frozen-readout contracts."""

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cv_nyu_mtan.model import MTAN, head
from experiments.cv_nyu_mtan.refine import features, initialize_native
from experiments.cv_nyu_mtan.evaluate import metrics, segmentation_loss


@pytest.mark.parametrize('multi', [True, False])
def test_numerical_parity_with_author(multi):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    name = 'model_segnet_mtan.py' if multi else 'model_segnet_stan.py'
    source = (Path(__file__).parent / 'reference' / name).read_text()
    source = source.replace('filter = [64, 128, 256, 512, 512]', 'filter = [8, 16, 32, 32, 32]')
    tree = ast.parse(source)
    # Execute only the author's class: importing original script starts training.
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SegNet')
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'opt': SimpleNamespace(task='semantic')}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), name, 'exec'), namespace)
    author = namespace['SegNet']().eval()
    adapted = MTAN([8, 16, 32, 32, 32], multi).eval()
    def layers(model):
        return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
    original, current = layers(author), layers(adapted)
    assert len(original) == len(current)
    for a, b in zip(original, current):
        assert type(a) == type(b)
        b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        ref = author(x)
        value = adapted(x)
    semantic = ref[0][0] if multi else ref
    torch.testing.assert_close(F.log_softmax(value['logits'], 1), semantic, rtol=1e-5, atol=1e-6)
    if multi:
        torch.testing.assert_close(value['depth'], ref[0][1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(value['normal'], ref[0][2], rtol=1e-5, atol=1e-6)


def test_native_initialization_capacity_and_frozen_buffers():
    torch.use_deterministic_algorithms(False)
    model = MTAN([8, 16, 32, 32, 32]).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 96))
    payload = {'embedding': torch.cat((output['semantic_hidden'], output['shared']), 1),
               'aux_prediction': torch.cat((output['depth'], output['normal']), 1), 'aux_hidden': output['aux_hidden']}
    counts = []
    for recipe in ('embedding_matched', 'aux_prediction', 'aux_hidden'):
        readout = head(32, 13)
        initialize_native(readout, model)
        prediction = readout(features(payload, recipe))
        torch.testing.assert_close(prediction, output['logits'], rtol=1e-5, atol=1e-6)
        segmentation_loss(prediction, torch.zeros(2, 64, 96, dtype=torch.long)).backward()
        counts.append(sum(p.numel() for p in readout.parameters()))
    assert len(set(counts)) == 1
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_dataset_confusion_and_invalid_pixels():
    class Identity(nn.Module):
        def forward(self, x):
            return x
    # Class 0: intersection 1, union 2; class 1: intersection 1, union 2.
    target = torch.tensor([[[0, 0, 1, -1]]])
    logits = torch.full((1, 13, 1, 4), -10.)
    logits[:, 0, :, 0] = 10
    logits[:, 1, :, 1:3] = 10
    result = metrics(Identity(), [{'features': logits, 'segmentation': target}], torch.device('cpu'))
    assert result['miou'] == .5
    assert result['valid_pixels'] == 3
    assert result['pixel_accuracy'] == pytest.approx(2/3)
