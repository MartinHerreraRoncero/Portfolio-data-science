import json, os, sys, uuid
from pathlib import Path
root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0, str(root))
from experiments.continuous_quality import check
folder = root / 'evaluation/continuous_quality_2026-10-27/control_runs' / uuid.uuid4().hex
folder.mkdir(parents=True)
records = {}
for arm in ('mixed', 'all27'):
    try:
        report = check(root / 'evaluation/mixed_single27_2026-10-27/paired_results_v2.json', root / 'evaluation/continuous_quality_2026-10-27/reference.json', folder / (arm + '.json'), arm=arm)
        failed = [c['case_id'] for c in report['case_results'] if not c['whole_correct']]
        expected = report['overall_pass'] is (arm == 'mixed') and (arm == 'mixed' or set(failed) == {'b08', 'n03'})
        records[arm] = dict(expected=expected, overall_pass=report['overall_pass'], failed=failed)
    except Exception as exc:
        records[arm] = dict(expected=False, error=f'{type(exc).__name__}: {exc}')
result = dict(output_dir=str(folder), controls=records, passed=all(r['expected'] for r in records.values()))
(folder / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
(folder.parent.parent / 'latest_controls.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False), flush=True)
sys.exit(0 if result['passed'] else 1)
