"""Verify completed artifacts, cache identity and independent material IDs."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.context import load_context
from pipeline.io import read_json, artifacts_match, sha256_file, write_json
from experiments.sci_mp_crystran.refine import load_cache, cache_dir, refiner_dir, recipes
from experiments.sci_mp_crystran.training import upstream_dir, load_upstream
from experiments.sci_mp_crystran.data import make_loader
from pipeline.runtime import configure, to_device

parser=argparse.ArgumentParser()
parser.add_argument('--config',required=True)
args=parser.parse_args()
context=load_context(args.config,Path(__file__).resolve().parents[3])
state=read_json(context.output_dir/'stage_state.json')
for name in ('download','prepare','train','cache','refine','evaluate','analyze'):
    assert state[name]['status']=='complete' and state[name]['identity']==context.identity
    assert artifacts_match(state[name]['artifacts'])
splits=read_json(context.output_dir/'split_manifest.json')['ids']
assert sum(len(v) for v in splits.values())==len(set(sum(splits.values(),[])))
for variant in context.config['upstream']['variants']:
    for seed in context.config['upstream']['seeds']:
        checkpoint=upstream_dir(context,variant,seed)/'best.pt'
        manifest=read_json(cache_dir(context,variant,seed)/'cache_manifest.json')
        assert sha256_file(checkpoint)==manifest['checkpoint_sha256']
        for split in ('b_train','b_val','y_test'):
            saved=load_cache(context,variant,seed,split)
            assert saved['ids']==splits[split]
            assert 'auxiliary' not in saved and 'aux_truth' not in saved
            assert saved['metadata']['frozen_parameters'] and saved['metadata']['eval_mode']
        for recipe in recipes(context,variant):
            for down_seed in context.config['refiner']['seeds']:
                directory=refiner_dir(context,variant,seed,recipe,down_seed)
                training=read_json(directory/'training_manifest.json')
                assert training['best_validation_metric']<=training['initial_validation_metric']+1e-6
pred=np.load(context.output_dir/'predictions.npz')
assert np.array_equal(pred['single_task_up1_ids'],pred['multi_task_up1_ids'])
# Architecture limitation audit on B_val only, after all selection; never used for model choice.
device=configure(context.config['runtime'])
model,_=load_upstream(context,'multi_task',1,device)
audit=[]
with torch.inference_mode():
    for batch in make_loader(context,'b_val',batch_size=1):
        if len(audit)==16: break
        batch=to_device(batch,device)
        z,c,m=batch['z'],batch['coords'],batch['mask']
        base=model(z,c,m)['main']
        order=torch.arange(z.shape[1]-1,-1,-1,device=device)
        perm=model(z[:,order],c[:,order],m[:,order])['main']
        shifted=model(z,c+torch.tensor([1.,2.,3.],device=device),m)['main']
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=device)
        rotated=model(z,c@rotation,m)['main']
        scale=read_json(context.output_dir/'target_normalization.json')['std'][0]
        audit.append({'id':batch['id'][0], 'atoms':z.shape[1],
                      'reverse_order_delta_eV':float((perm-base).abs())*scale,
                      'translation_delta_eV':float((shifted-base).abs())*scale,
                      'rotation_delta_eV':float((rotated-base).abs())*scale})
evidence={'stage_artifacts_verified':True,'disjoint_material_ids':True,'upstream_unchanged_after_refine':True,
          'cache_no_auxiliary_truth':True,'native_epoch0_retained':True,'paired_test_ids':True,
          'symmetry_audit_split':'b_val','symmetry_audit':audit,
          'symmetry_mean_delta_eV':{k:float(np.mean([row[k] for row in audit])) for k in
             ('reverse_order_delta_eV','translation_delta_eV','rotation_delta_eV')}}
write_json(context.output_dir/'verification.json',evidence)
print('VERIFIED: seven artifact-backed stages; disjoint IDs; unchanged frozen checkpoints; main-only caches; native epoch0 candidates.')
print('Architecture audit mean absolute deltas (eV):',evidence['symmetry_mean_delta_eV'])