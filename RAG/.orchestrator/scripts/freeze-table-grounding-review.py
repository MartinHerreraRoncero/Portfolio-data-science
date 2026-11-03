from pathlib import Path
import difflib, hashlib, json, zipfile
root = Path(__file__).resolve().parents[2]
archive = zipfile.ZipFile(root/'.orchestrator/recovery/table-grounding-before.zip')
old = {n: archive.read(n) for n in archive.namelist() if n.endswith('.py')}
current = {p.relative_to(root).as_posix(): p.read_bytes() for folder in ('rag_pensiones','experiments','tests') for p in (root/folder).glob('*.py')}
allowed = {'rag_pensiones/table_evidence.py','rag_pensiones/numeric_guard.py','experiments/table_grounded.py','experiments/compare_mixed_single27.py','experiments/compare_table_grounding.py','tests/test_table_grounding.py','tests/test_table_grounded_pipeline.py','tests/test_table_grounding_runner.py'}
changed = sorted(n for n in set(old)|set(current) if old.get(n) != current.get(n))
unexpected = [n for n in changed if n not in allowed]
if unexpected: raise RuntimeError('Unauthorized source changes: '+repr(unexpected))
review = root/'.orchestrator/reviews'
review.mkdir(exist_ok=True)
hashes = {n: hashlib.sha256(current[n]).hexdigest() for n in changed}
for label, paths in [('core', changed[:0]+[n for n in changed if n.startswith('rag_pensiones/') or n=='tests/test_table_grounding.py']), ('pipeline',[n for n in changed if n in {'experiments/table_grounded.py','experiments/compare_mixed_single27.py','tests/test_table_grounded_pipeline.py'}]), ('runner',[n for n in changed if 'runner' in n or n=='experiments/compare_table_grounding.py']), ('full',changed)]:
    diff = ''.join(line for n in paths for line in difflib.unified_diff(old.get(n,b'').decode('utf-8-sig').splitlines(True),current.get(n,b'').decode('utf-8-sig').splitlines(True),fromfile='a/'+n,tofile='b/'+n))
    target = review/('table-grounding-'+label+'-current.patch')
    target.write_text(diff,encoding='utf-8')
    print(label, len(diff.splitlines()), hashlib.sha256(target.read_bytes()).hexdigest())
(review/'table-grounding-current-hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
