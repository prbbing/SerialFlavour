"""Exact-matrix paired hierarchical summaries; upstream seeds are repetitions."""
from statistics import mean, stdev
from pipeline.io import read_json, write_json, sha256_file, artifacts_match
from experiments.sci_qm7x.common import check_metadata, expected_metadata, applicable_recipes
from experiments.sci_qm7x.evaluate import matrix_keys, verify_matrix


def summary(values):
    if len(values) < 2:
        raise ValueError('sample SD requires at least two upstream seeds')
    return {'mean': mean(values), 'sample_sd': stdev(values), 'ddof': 1, 'n_upstream_seeds': len(values)}


def hierarchical(rows, upstream_seeds, downstream_seeds):
    """Exactly one value per (us, ds); collapse ds before sample SD across us."""
    keyed = {(row['upstream_seed'], row['downstream_seed']): row['mae_eV'] for row in rows}
    expected = {(us, ds) for us in upstream_seeds for ds in downstream_seeds}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError('missing/duplicate hierarchical cells')
    per_upstream = {str(us): mean(keyed[(us, ds)] for ds in downstream_seeds) for us in upstream_seeds}
    return {**summary(list(per_upstream.values())), 'per_upstream_mean': per_upstream,
            'n_downstream_seeds': len(downstream_seeds), 'statistical_unit': 'upstream initialization, fixed molecule-disjoint split'}


def paired(rows, variant_left, recipe_left, variant_right, recipe_right, upstream_seeds, downstream_seeds):
    """Positive MAE(left)-MAE(right) favors right; retain same us/ds pairing."""
    index = {(r['variant'], r['upstream_seed'], r['recipe'], r['downstream_seed']): r['mae_eV'] for r in rows}
    per_upstream, raw = {}, []
    ds_values = [None] if recipe_left == recipe_right == 'native' else downstream_seeds
    for us in upstream_seeds:
        differences = []
        for ds in ds_values:
            left = index[(variant_left, us, recipe_left, None if recipe_left == 'native' else ds)]
            right = index[(variant_right, us, recipe_right, None if recipe_right == 'native' else ds)]
            differences.append(left - right)
            raw.append({'upstream_seed': us, 'downstream_seed': ds, 'mae_improvement_eV': left - right})
        per_upstream[str(us)] = mean(differences)
    return {**summary(list(per_upstream.values())), 'per_upstream_mean': per_upstream,
            'paired_cells': raw, 'positive_favors': f'{variant_right}/{recipe_right}',
            'unit': 'eV', 'statistical_unit': 'upstream initialization after averaging paired downstream seeds'}


def analyze(context):
    provenance = verify_matrix(context)
    directory = context.output_dir / 'evaluation'
    evaluation = read_json(directory / 'evaluation_manifest.json')
    check_metadata(context, evaluation)
    if evaluation['provenance'] != provenance or not artifacts_match(evaluation['artifacts']):
        raise ValueError('locked evaluation evidence/provenance mismatch')
    metrics = read_json(directory / 'metrics.json')
    check_metadata(context, metrics)
    rows = metrics['rows']
    observed = [(r['variant'], r['upstream_seed'], r['recipe'], r['downstream_seed']) for r in rows]
    if len(set(observed)) != len(rows) or set(observed) != set(matrix_keys(context)):
        raise ValueError('analysis refuses missing/extra/duplicate matrix cells')
    upstream, downstream = context.config['upstream']['seeds'], context.config['refiner']['seeds']
    summaries = {}
    for variant in context.config['upstream']['variants']:
        for recipe in ['native', *applicable_recipes(context, variant)]:
            cells = [r for r in rows if r['variant'] == variant and r['recipe'] == recipe]
            summaries[f'{variant}/{recipe}'] = hierarchical(cells, upstream, [None] if recipe == 'native' else downstream)
    comparisons = {}
    variants = context.config['upstream']['variants']
    pairs = [('single_task', 'native', 'multi_task', 'native')] if set(variants) == {'single_task','multi_task'} else []
    for variant in context.config['upstream']['variants']:
        pairs.extend([(variant, 'native', variant, 'embedding'), (variant, 'embedding', variant, 'embedding_capacity')])
    if 'multi_task' in variants:
        for baseline in ('embedding', 'embedding_capacity', 'embedding_hidden', 'embedding_aux_shuffle'):
            pairs.append(('multi_task', baseline, 'multi_task', 'embedding_aux'))
    for vl, rl, vr, rr in pairs:
        comparisons[f'{vl}/{rl} -> {vr}/{rr}'] = paired(rows, vl, rl, vr, rr, upstream, downstream)
    output = context.output_dir / 'analysis' / 'summary.json'
    write_json(output, {**expected_metadata(context), 'summaries': summaries, 'comparisons': comparisons,
                        'evaluation_sha256': sha256_file(directory / 'evaluation_manifest.json'),
                        'interpretation': 'descriptive paired initialization effects; no significance or causal mechanism claim'})
    report = context.output_dir / 'analysis' / 'report_zh.md'
    lines = ['# QM7-X HOMO post-refinement 结果', '',
             '锁定 Y；单位 eV。每个上游 seed 内先平均下游 seed，再报告上游均值与样本 SD（ddof=1）。',
             '5×5 是 5 个上游初始化及其嵌套读出重复，不能按 25 个独立分子实验解释。', '',
             '| 上游 / 读出 | HOMO MAE 均值 | 上游样本 SD |', '|---|---:|---:|']
    for key, record in summaries.items():
        lines.append(f"| {key} | {record['mean']:.6f} | {record['sample_sd']:.6f} |")
    lines.extend(['', '配对改进为左侧 MAE 减右侧 MAE；正值有利于右侧。', '',
                  '| 配对比较 | 改进均值 eV | 上游样本 SD |', '|---|---:|---:|'])
    for key, record in comparisons.items():
        lines.append(f"| {key} | {record['mean']:.6f} | {record['sample_sd']:.6f} |")
    lines.extend(['', '该统计只描述固定母分子划分上的初始化变化，不包含数据划分不确定性。',
                  '辅助预测来自冻结表征，收益只能支持该有限读出协议中的可用性；局部对应打乱、hidden 与容量对照不能单独证明因果机制。'])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return [output, report]
