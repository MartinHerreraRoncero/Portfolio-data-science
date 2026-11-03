$ErrorActionPreference='Continue'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Set-Location -LiteralPath $root
$env:TEMP=Join-Path $root 'tmp'; $env:TMP=$env:TEMP; $env:PYTHONIOENCODING='utf-8'
$runs=Join-Path $root 'evaluation/mixed_single27_2026-10-27/gemini_test_runs'
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$runId=[guid]::NewGuid().ToString('N'); $log=Join-Path $runs ($runId+'.log')
$latest=Join-Path $root 'evaluation/mixed_single27_2026-10-27/latest_gemini_tests.json'
$sourceBefore=@{}; foreach($file in @('experiments/compare_mixed_single27.py','tests/test_mixed_single27.py')){$sourceBefore[$file]=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash}
@{status='running';run_id=$runId;log=$log;source_before=$sourceBefore}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $latest -Encoding UTF8
& python -m unittest discover -s tests -p test_mixed_single27.py -v *> $log
$testExit=$LASTEXITCODE; if($null -eq $testExit){$testExit=127}
$sourceAfter=@{}; foreach($file in $sourceBefore.Keys){$sourceAfter[$file]=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash}
$result=@{status='completed';run_id=$runId;exit_code=$testExit;log=$log;source_before=$sourceBefore;source_after=$sourceAfter;finished_utc=(Get-Date).ToUniversalTime().ToString('o')}
$result|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $runs ($runId+'.json')) -Encoding UTF8
$result|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $latest -Encoding UTF8
Get-Content -LiteralPath $log -Tail 6
exit $testExit
