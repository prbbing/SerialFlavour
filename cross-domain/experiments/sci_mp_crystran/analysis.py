"""Report paired smoke comparisons without treating one seed as inference."""
from pipeline.io import read_json, write_json


def analyze(context):
    evaluation = read_json(context.output_dir / 'evaluation.json')
    records = evaluation['records']
    mt = {r['recipe']: r for r in records if r['variant'] == 'multi_task'}
    st = {r['recipe']: r for r in records if r['variant'] == 'single_task'}
    matched = {r['recipe']: r for r in records if r['variant'] == 'mt_main_only'}
    comparisons = {
        'mt_minus_matched_main_only_native_mae_eV': mt['native']['metrics']['mae']-matched['native']['metrics']['mae'],
        'mt_minus_st_native_mae_eV': mt['native']['metrics']['mae']-st['native']['metrics']['mae'],
        'mt_embedding_minus_native_mae_eV': mt['embedding']['metrics']['mae']-mt['native']['metrics']['mae'],
        'mt_aux_minus_embedding_capacity_mae_eV': mt['embedding_aux']['metrics']['mae']-mt['embedding_capacity']['metrics']['mae']}
    summary = context.output_dir / 'summary.json'
    write_json(summary, {'comparisons': comparisons, 'records': records,
                         'evidence': '1024-structure CPU engineering smoke, single upstream/downstream seed; no uncertainty estimate',
                         'auxiliary_information': 'deterministic frozen computation; no new Shannon information'})
    report = context.output_dir / 'results_zh.md'
    lines = ['# MP CrystalTransformer 本地小规模结果', '', '只有一个 seed；仅验证工程流程，不能据此推断方法有效性。', '',
             '| 上游 | 读出 | Y MAE (eV) | RMSE (eV) | R² | 参数 |', '|---|---|---:|---:|---:|---:|']
    for r in records:
        m = r['metrics']
        lines.append(f"| {r['variant']} | {r['recipe']} | {m['mae']:.6f} | {m['rmse']:.6f} | {m['r_squared']:.6f} | {r['parameters']} |")
    lines.extend(['', '差值均为前者减后者，MAE 负值为改善：', ''] + [f'- {k}: {v:.6f}' for k,v in comparisons.items()])
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return [summary, report]