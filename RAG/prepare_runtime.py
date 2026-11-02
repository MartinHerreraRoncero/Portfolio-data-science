"""Restore the exact runtime snapshot without downloading or running models."""
import gzip
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'runtime/manifest.json').read_text(encoding='utf-8'))
    for asset in manifest:
        destination = (root / asset['path']).resolve()
        destination.relative_to(root)
        data = gzip.decompress((root / 'runtime' / (asset['path'] + '.gz')).read_bytes())
        if len(data) != asset['bytes'] or hashlib.sha256(data).hexdigest() != asset['sha256']:
            raise ValueError('Runtime asset integrity failure: ' + asset['path'])
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != asset['sha256']:
                raise FileExistsError('Different local asset: ' + str(destination))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(f"Runtime ready: {len(manifest)} assets verified.")


if __name__ == '__main__':
    main()
