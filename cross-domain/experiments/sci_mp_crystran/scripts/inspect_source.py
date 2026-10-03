"""Download a fixed MP snapshot; inspect its schema before adapter preparation."""
import gzip, hashlib, json, urllib.request
from pathlib import Path
root = Path('/mnt/d/hep_analysis/gn2_study/dataset_ex/sci_mp_crystran')
root.mkdir(parents=True, exist_ok=True)
path = root / 'mp_all_20181018.json.gz'
expected = '1f3de2dc7c68959647240921b841293fce918ea1708e6f3760ba35a9cdfe0500'
if not path.exists():
    temp = path.with_suffix('.part')
    with urllib.request.urlopen('https://ndownloader.figshare.com/files/13309250', timeout=60) as response, temp.open('wb') as stream:
        print('HTTP bytes:', response.headers.get('Content-Length'), flush=True)
        count = 0
        while chunk := response.read(1024*1024):
            count += len(chunk)
            if count > 500_000_000: raise ValueError('download exceeds 500 MB smoke limit')
            stream.write(chunk)
            if count % (20*1024*1024) == 0: print('downloaded bytes', count, flush=True)
    if hashlib.file_digest(temp.open('rb'), 'sha256').hexdigest() != expected: raise ValueError('SHA256 mismatch')
    temp.replace(path)
print('verified bytes', path.stat().st_size, 'SHA256', hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(), flush=True)
import ijson
with gzip.open(path, 'rb') as stream:
    print('columns', next(ijson.items(stream, 'columns')))
with gzip.open(path, 'rb') as stream:
    row = next(ijson.items(stream, 'data.item', use_float=True))
    print('first row:', str(row)[:2200])