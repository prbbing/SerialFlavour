"""Small I/O/CLI helpers; scientific preparation and drawing live in each figure script."""
import argparse
from pathlib import Path
import shutil

from pipeline_common import PREPARED, FIGURE, DATA, SEEDS, checked_child, read_json, write_json, source_record, sha, validate_result_source, figure_filename


def prepare_one(figure_id, kind, prepare, data_dir=PREPARED):
    data_dir = Path(data_dir).resolve()
    payload, sources = prepare()
    records = [source_record(path) for path in dict.fromkeys(sources)]
    catalog_path = data_dir/'catalog.json'
    catalog = read_json(catalog_path) if catalog_path.exists() else {'schema_version': 1, 'figures': []}
    if catalog['schema_version'] != 1:
        raise ValueError('Unsupported prepared data schema')
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = 'pipeline_source.svg' if kind == 'pipeline' else figure_id+'.json'
    path = checked_child(data_dir, filename)
    if kind == 'pipeline':
        path.write_bytes(payload)
    else:
        write_json(path, dict(schema_version=1, kind=kind, figure=figure_filename(figure_id), payload=payload))
    entry = dict(id=figure_id, kind=kind, data_file=filename, output=figure_filename(figure_id), sha256=sha(path), sources=records)
    entries = {e['id']: e for e in catalog['figures']}
    entries[figure_id] = entry
    write_json(catalog_path, dict(schema_version=1, upstream_seeds=SEEDS,
               scope=f'{len(entries)} prepared figures; each has an independent script.', figures=list(entries.values())))
    print(f'Prepared {figure_id}', flush=True)
    return entry


def checked_input(figure_id, kind, data_dir=PREPARED):
    data_dir = Path(data_dir).resolve()
    catalog_path = data_dir/'catalog.json'
    if not catalog_path.exists():
        raise FileNotFoundError(f'Prepared data is missing: {catalog_path}. Run this figure script with --prepare-only first.')
    catalog = read_json(catalog_path)
    if catalog['schema_version'] != 1:
        raise ValueError('Unsupported prepared data schema')
    found = [e for e in catalog['figures'] if e['id'] == figure_id]
    if len(found) != 1:
        raise ValueError(f'No unique prepared entry for {figure_id}; run its script with --prepare-only.')
    entry = found[0]
    for record in entry['sources']:
        validate_result_source(record['path'])
    if entry['kind'] != kind or entry['output'] != figure_filename(figure_id):
        raise ValueError(f'Prepared catalog entry disagrees with this script: {figure_id}')
    source = checked_child(data_dir, entry['data_file'])
    if sha(source) != entry['sha256']:
        raise ValueError(f'Prepared data hash changed: {source}; rerun this script with --prepare-only.')
    return entry, source


def render_one(figure_id, kind, draw, data_dir=PREPARED, output_dir=FIGURE, copy_to=None):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or output_dir.is_relative_to(data_dir):
        raise ValueError('Keep generated images outside the prepared input directory.')
    entry, source = checked_input(figure_id, kind, data_dir)
    if kind != 'pipeline':
        envelope = read_json(source)
        if envelope['schema_version'] != 1 or envelope['kind'] != kind or envelope['figure'] != entry['output']:
            raise ValueError(f'Prepared data disagrees with this script: {source}')
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = checked_child(output_dir, entry['output'])
    removed = []
    if kind == 'pipeline':
        draw(source.read_text(encoding='utf-8'), output_dir)
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from paper_style import publication_style
        plt.rcdefaults()
        fig, options = draw(envelope['payload'])
        try:
            removed = publication_style(fig, entry['output'])
            fig.savefig(destination, **options)
        finally:
            plt.close(fig)
    report = dict(file=entry['output'], input=entry['data_file'], input_sha256=sha(source),
                  output_sha256=sha(destination), removed_for_caption=removed)
    reports = DATA/'render_reports' if output_dir == FIGURE.resolve() else output_dir/'render_reports'
    write_json(reports/(figure_id+'.json'), report)
    if copy_to:
        copy_to = Path(copy_to)
        copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, copy_to/entry['output'])
    print(f'Rendered {entry["output"]}', flush=True)
    return report


def run_cli(figure_id, kind, prepare, draw):
    parser = argparse.ArgumentParser(description=f'Independent figure: {figure_id}. Default: prepare data, then draw.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='Only update this figure\'s prepared input.')
    mode.add_argument('--plot-only', action='store_true', help='Only draw this figure from its saved input.')
    parser.add_argument('--data-dir', type=Path, default=PREPARED)
    parser.add_argument('--output-dir', type=Path, default=FIGURE)
    parser.add_argument('--copy-to', type=Path)
    args = parser.parse_args()
    if not args.plot_only:
        prepare_one(figure_id, kind, prepare, args.data_dir)
    if not args.prepare_only:
        render_one(figure_id, kind, draw, args.data_dir, args.output_dir, args.copy_to)
