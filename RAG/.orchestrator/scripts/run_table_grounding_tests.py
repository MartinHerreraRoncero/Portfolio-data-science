import hashlib, json, os, subprocess, sys, uuid
from pathlib import Path
from datetime import datetime, timezone
root = Path(__file__).resolve().parents[2]
os.chdir(root)
os.environ.update(TEMP=str(root/'tmp'), TMP=str(root/'tmp'), PYTHONIOENCODING='utf-8')
folder = root/'evaluation/table_grounding_2026-10-27/test_runs'
folder.mkdir(parents=True, exist_ok=True)
run_id = uuid.uuid4().hex
paths = list((root/'rag_pensiones').glob('*table*.py')) + [root/'rag_pensiones/numeric_guard.py',root/'experiments/compare_mixed_single27.py'] + list((root/'experiments').glob('*table_ground*.py')) + list((root/'tests').glob('test_table_ground*.py'))
hashes = lambda: {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
before = hashes()
log = folder/(run_id+'.log')
record = dict(status='running', run_id=run_id, log=str(log), source_before=before)
latest = folder.parent/'latest_tests.json'
latest.write_text(json.dumps(record, indent=2), encoding='utf-8')
with log.open('w', encoding='utf-8') as stream:
    result = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_table_ground*.py','-v'], stdout=stream, stderr=subprocess.STDOUT, cwd=root)
record.update(status='completed',exit_code=result.returncode,source_after=hashes(),source_unchanged=before==hashes(),finished_utc=datetime.now(timezone.utc).isoformat())
latest.write_text(json.dumps(record, indent=2), encoding='utf-8')
(folder/(run_id+'.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
print('\n'.join(log.read_text(encoding='utf-8').splitlines()[-8:]))
sys.exit(result.returncode)
