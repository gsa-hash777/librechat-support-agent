"""Download revision-pinned public ONNX data files; never execute remote Python."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

SERVICE = Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]


def checksum(path, algorithm='sha256', git_blob=False):
    result = hashlib.new(algorithm)
    if git_blob:
        result.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verified(path, entry):
    if not path.exists() or path.stat().st_size != entry['size']:
        return False
    if entry.get('sha256'):
        return checksum(path) == entry['sha256']
    return checksum(path, 'sha1', git_blob=True) == entry['git_blob_sha1']


def download(url, partial, entry):
    # A dropped CDN stream may end without raising. Resume only an explicit
    # matching Content-Range; never append an ignored range/full response.
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= entry['size']:
            if verified(partial, entry):
                return
            offset = 0
        request = Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urlopen(request, timeout=60) as source:
                append = offset and source.status == 206 and source.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
                if source.status == 206 and not append and offset:
                    raise RuntimeError('Unexpected model download byte range.')
                with partial.open('ab' if append else 'wb') as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
            if verified(partial, entry):
                return
        except (OSError, TimeoutError):
            pass
        print(json.dumps({'download_retry': attempt + 1, 'received_bytes': partial.stat().st_size if partial.exists() else 0}), flush=True)
    raise RuntimeError('Model download did not pass its pinned integrity check after retries.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.local/models/support-retrieval')
    parser.add_argument('--lock', type=Path, default=SERVICE / 'evaluation/retrieval_models.json')
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding='utf-8'))
    manifest = {'prepared_at_utc':datetime.now(timezone.utc).isoformat(), 'models':{}}
    for role, model in lock['models'].items():
        recorded = {**model, 'files':{}}
        folder = args.output / role
        folder.mkdir(parents=True, exist_ok=True)
        for local, entry in model['files'].items():
            assert Path(local).name == local
            path = folder / local
            if not verified(path, entry):
                if path.exists():
                    raise RuntimeError('Existing model file differs from lock; choose a clean output directory.')
                url = f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{entry["remote"]}'
                partial = path.with_suffix(path.suffix + '.part')
                began = time.monotonic()
                download(url, partial, entry)
                if not verified(partial, entry):
                    raise RuntimeError('Model integrity check failed: ' + role + '/' + local)
                partial.replace(path)
                print(json.dumps({'downloaded':role + '/' + local,'bytes':path.stat().st_size,'seconds':round(time.monotonic()-began,2)}),flush=True)
            recorded['files'][local] = {**entry, 'sha256':checksum(path)}
        manifest['models'][role] = recorded
    previous_path = args.output / 'manifest.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        if previous.get('models') == manifest['models']:
            # Stable provenance permits safe vector-index reuse after reruns.
            manifest['prepared_at_utc'] = previous['prepared_at_utc']
    payload = json.dumps(manifest,ensure_ascii=False,indent=2)
    (args.output / 'manifest.json').write_text(payload,encoding='utf-8')
    print(json.dumps({'status':'ready','model_dir':str(args.output),'remote_code_executed':False}))


if __name__ == '__main__':
    main()
