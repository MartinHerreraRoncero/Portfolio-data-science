from pathlib import Path
import difflib, hashlib, json, zipfile
root = Path(__file__).resolve().parents[2]
archive = zipfile.ZipFile(root/'.orchestrator/recovery/table-grounding-before-final-senior-four.zip')
owned = set(archive.namelist())
before = json.loads((root/'.orchestrator/reviews/table-grounding-current-hashes.json').read_text(encoding='utf-8-sig'))
for name, digest in before.items():
    if name not in owned and hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Unexpected change outside correction scope: '+name)
patch, binding = [], {}
for name in sorted(owned):
    old, new = archive.read(name), (root/name).read_bytes()
    patch.extend(difflib.unified_diff(old.decode('utf-8-sig').splitlines(True),new.decode('utf-8-sig').splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    binding[name] = {'before':hashlib.sha256(old).hexdigest(),'after':hashlib.sha256(new).hexdigest()}
target = root/'.orchestrator/reviews/table-grounding-final-senior-fixes.patch'
target.write_text(''.join(patch),encoding='utf-8')
binding['patch_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
(target.with_suffix('.json')).write_text(json.dumps(binding,indent=2),encoding='utf-8')
print('Correction patch lines:', len(patch), 'sha256:', binding['patch_sha256'])
