"""Hierarchical multi-seed metrics and paired effects on the locked Y set."""
from statistics import mean, stdev
from pipeline.io import read_json, write_json

METRICS = ('mae', 'rmse', 'r_squared')


def describe(values):
    """SD across independent upstream initializations, never flattened 5x5."""
    if not values or any(value is None for value in values):
        return {'mean': None, 'sample_sd': None, 'n_upstream_seeds': len(values)}
    return {'mean': mean(values), 'sample_sd': stdev(values) if len(values) > 1 else None,
            'n_upstream_seeds': len(values)}


def summarize_records(config, records):
    upstream = config['upstream']['seeds']
    downstream = config['refiner']['seeds']
    variants = config['upstream']['variants']
    recipes = config['refiner']['recipes']
    index = {}
    expected = set()
    for variant in variants:
        allowed = [r for r in recipes if variant == 'multi_task' or r in ('embedding', 'embedding_capacity')]
        for up_seed in upstream:
            expected.add((variant, up_seed, 'native', None))
            for recipe in allowed:
                expected.update((variant, up_seed, recipe, seed) for seed in downstream)
    for record in records:
        key = (record['variant'], record['upstream_seed'], record['recipe'], record['seed'])
        if key in index:
            raise ValueError(f'duplicate evaluation record: {key}')
        index[key] = record
    if set(index) != expected:
        raise ValueError(f'incomplete evaluation matrix: missing={expected-set(index)}, unexpected={set(index)-expected}')
    groups = []
    for variant in variants:
        allowed = ['native', *[r for r in recipes if variant == 'multi_task' or r in ('embedding', 'embedding_capacity')]]
        for recipe in allowed:
            rows = []
            for up_seed in upstream:
                seeds = [None] if recipe == 'native' else downstream
                selected = [index[(variant, up_seed, recipe, seed)] for seed in seeds]
                rows.append({'upstream_seed': up_seed, 'n_downstream_seeds': len(seeds),
                             'metrics': {metric: mean([r['metrics'][metric] for r in selected])
                                         if all(r['metrics'][metric] is not None for r in selected) else None
                                         for metric in METRICS}})
            groups.append({'variant': variant, 'recipe': recipe, 'per_upstream': rows,
                           'metrics': {metric: describe([r['metrics'][metric] for r in rows]) for metric in METRICS}})
    specs = [('mt_minus_matched_main_only_native_mae_eV', 'multi_task', 'native', 'mt_main_only', 'native'),
             ('mt_minus_st_native_mae_eV', 'multi_task', 'native', 'single_task', 'native'),
             ('mt_embedding_minus_native_mae_eV', 'multi_task', 'embedding', 'multi_task', 'native'),
             ('mt_aux_minus_embedding_capacity_mae_eV', 'multi_task', 'embedding_aux', 'multi_task', 'embedding_capacity'),
             ('mt_aux_minus_embedding_mae_eV', 'multi_task', 'embedding_aux', 'multi_task', 'embedding'),
             ('mt_aux_minus_shuffle_mae_eV', 'multi_task', 'embedding_aux', 'multi_task', 'embedding_aux_shuffle'),
             ('mt_aux_minus_hidden_mae_eV', 'multi_task', 'embedding_aux', 'multi_task', 'embedding_hidden')]
    comparisons = {}
    for name, lhs_variant, lhs_recipe, rhs_variant, rhs_recipe in specs:
        if lhs_variant not in variants or rhs_variant not in variants:
            continue
        if any(recipe != 'native' and recipe not in recipes for recipe in (lhs_recipe, rhs_recipe)):
            continue
        paired = []
        for up_seed in upstream:
            seeds = [None] if lhs_recipe == rhs_recipe == 'native' else downstream
            deltas = []
            for seed in seeds:
                lhs = index[(lhs_variant, up_seed, lhs_recipe, None if lhs_recipe == 'native' else seed)]
                rhs = index[(rhs_variant, up_seed, rhs_recipe, None if rhs_recipe == 'native' else seed)]
                deltas.append({'downstream_seed': seed, 'mae_delta_eV': lhs['metrics']['mae']-rhs['metrics']['mae']})
            paired.append({'upstream_seed': up_seed, 'mean_delta_eV': mean([r['mae_delta_eV'] for r in deltas]),
                           'paired_downstream': deltas})
        comparisons[name] = {**describe([r['mean_delta_eV'] for r in paired]), 'per_upstream': paired,
                             'negative_upstream_seeds': sum(r['mean_delta_eV'] < 0 for r in paired),
                             'direction': 'negative MAE difference means improvement'}
    return groups, comparisons


def analyze(context):
    evaluation = read_json(context.output_dir / 'evaluation.json')
    if evaluation['identity'] != context.identity:
        raise ValueError('evaluation identity mismatch')
    groups, comparisons = summarize_records(context.config, evaluation['records'])
    summary = context.output_dir / 'summary.json'
    count = sum(context.config['data']['sizes'].values())
    write_json(summary, {'identity': context.identity, 'sample_count': count, 'groups': groups,
                         'comparisons': comparisons, 'records': evaluation['records'],
                         'aggregation': 'average downstream seeds within each upstream seed, then mean and sample SD (ddof=1) across upstream seeds',
                         'single_seed_sd': None, 'confidence_intervals': 'not computed',
                         'auxiliary_information': 'deterministic frozen computation; no new Shannon information'})
    report = context.output_dir / 'results_zh.md'
    lines = ['# MP CrystalTransformer 冻结读出结果', '',
             f"{count} 个材料；上游 seeds={context.config['upstream']['seeds']}，下游 seeds={context.config['refiner']['seeds']}。", '',
             '先在每个上游 seed 内平均下游 seed，再在上游 seed 间报告均值和样本 SD（ddof=1）。',
             '25 个下游组合不视为 25 个独立上游重复；单 seed 时 SD 留空。', '',
             '| 上游 | 读出 | Y MAE mean (eV) | upstream SD (eV) | RMSE mean (eV) | R² mean |',
             '|---|---|---:|---:|---:|---:|']
    def fmt(value):
        return '' if value is None else f'{value:.6f}'
    for group in groups:
        m = group['metrics']
        lines.append(f"| {group['variant']} | {group['recipe']} | {fmt(m['mae']['mean'])} | {fmt(m['mae']['sample_sd'])} | {fmt(m['rmse']['mean'])} | {fmt(m['r_squared']['mean'])} |")
    lines.extend(['', '## 配对 MAE 差值', '', '差值为前者减后者，负值表示改善；逐 seed 结果保存在 summary.json。', '',
                  '| 对比 | mean delta (eV) | upstream SD (eV) | 改善的上游 seeds |', '|---|---:|---:|---:|'])
    for name, comparison in comparisons.items():
        lines.append(f"| {name} | {fmt(comparison['mean'])} | {fmt(comparison['sample_sd'])} | {comparison['negative_upstream_seeds']}/{comparison['n_upstream_seeds']} |")
    lines.extend(['', '均值和 SD 是锁定划分下的初始化重复统计，不能代表所有材料分布、划分或对称性设置的不确定性。',
                  '本表不执行显著性检验，也不据此自动选择主辅任务。'])
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return [summary, report]