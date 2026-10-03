"""Download/verify pinned offline assets; does not initialize an experiment or train."""
import argparse
from pathlib import Path
import sys
DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DOMAIN_ROOT))
from pipeline.context import load_context
from pipeline.io import sha256_file
from experiments.nlp_massive_xlm.data import (PINNED_FILES, ARCHIVE_URL, TOKENIZER_REVISION,
    BASE_WEIGHTS_SHA256, BASE_CONFIG_SHA256, pretrained_directory, fetch)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DOMAIN_ROOT / 'experiments/nlp_massive_xlm/config/cluster_full.json'))
    parser.add_argument('--data-root', help='override only asset destination, not experiment config')
    parser.add_argument('--verify-only', action='store_true', help='no network, no writes')
    args = parser.parse_args()
    context = load_context(args.config, DOMAIN_ROOT)
    if context.config['dataset'] != 'nlp_massive_xlm':
        parser.error('requires nlp_massive_xlm config')
    if args.data_root:
        context.config = {**context.config, 'data_root': args.data_root}
    sources = []
    for name, expected in PINNED_FILES.items():
        url = ARCHIVE_URL if name.endswith('tar.gz') else f'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/{TOKENIZER_REVISION}/{name.split("/")[-1]}'
        sources.append((url, context.data_dir / 'raw' / name, expected, 100_000_000))
    for name, expected, maximum in [('config.json', BASE_CONFIG_SHA256, 100_000),
                                    ('model.safetensors', BASE_WEIGHTS_SHA256, 2_000_000_000)]:
        sources.append((f'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/{TOKENIZER_REVISION}/{name}',
                        pretrained_directory(context) / name, expected, maximum))
    for url, path, expected, maximum in sources:
        if args.verify_only:
            if not path.is_file() or path.stat().st_size > maximum or sha256_file(path) != expected:
                raise ValueError(f'offline asset missing/corrupt: {path}')
        else:
            fetch(url, path, expected, max_bytes=maximum)
        print(f'verified {path}', flush=True)

if __name__ == '__main__':
    main()
