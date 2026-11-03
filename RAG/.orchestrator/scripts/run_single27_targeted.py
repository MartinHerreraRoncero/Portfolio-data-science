import hashlib, json, os, subprocess, sys, uuid
from pathlib import Path
from datetime import datetime, timezone
root = Path(__file__).resolve().parents[2]
os.chdir(root)
os.environ.update(TEMP=str(root / 'tmp'), TMP=str(root / 'tmp'), PYTHONIOENCODING='utf-8')
runs = root / 'evaluation/mixed_single27_2026-10-27/gemini_test_runs'
runs.mkdir(parents=True, exist_ok=True)
run_id = uuid.uuid4().hex
log_path = runs / (run_id + '.log')
latest = runs.parent / 'latest_gemini_tests.json'
paths = ('experiments/compare_mixed_single27.py', 'tests/test_mixed_single27.py')
before = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
latest.write_text(json.dumps(dict(status='running', run_id=run_id, log=str(log_path), source_before=before), indent=2), encoding='utf-8')
with log_path.open('w', encoding='utf-8') as log:
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_mixed_single27.py', '-v'], cwd=root, stdout=log, stderr=subprocess.STDOUT)
after = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
record = dict(status='completed', run_id=run_id, exit_code=result.returncode, log=str(log_path), source_before=before, source_after=after, source_unchanged=before == after, finished_utc=datetime.now(timezone.utc).isoformat())
(runs / (run_id + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
latest.write_text(json.dumps(record, indent=2), encoding='utf-8')
print('\n'.join(log_path.read_text(encoding='utf-8').splitlines()[-6:]), flush=True)
sys.exit(result.returncode)
