[CmdletBinding()]
param([switch]$Apply)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$settings=Join-Path $env:USERPROFILE '.gemini/antigravity-cli/settings.json'
$cfg=Get-Content -LiteralPath $settings -Raw|ConvertFrom-Json
$before=$cfg.permissions|ConvertTo-Json -Depth 30 -Compress
$paths=@('rag_pensiones/grounding.py','experiments/compare_system.py','experiments/preserved_q3_verifier.py','experiments/query_preserved.py','experiments/evaluate_mixed_verifier.py','tests/test_mixed_verifier.py','tests/test_mixed_pipeline.py','evaluation/mixed_verifier_2026-10-26/README_RUNNER.md')
$rules=@($paths|ForEach-Object {'write_file(D:/Portfolio-data-science/RAG/'+$_+')'})
$rules+=@('command(python -m unittest discover -s tests -p test_mixed_verifier.py -v)','command(python -m unittest discover -s tests -p test_mixed_pipeline.py -v)','command(python -m unittest discover -s tests -q)')
$missing=@($rules|Where-Object {$_ -notin $cfg.permissions.allow})
if(-not $Apply){$missing;return}
$backup=Join-Path $root ('.orchestrator/recovery/mixed-permissions-'+[guid]::NewGuid().ToString('N')+'.json')
Set-Content -LiteralPath $backup -Value $before -Encoding UTF8
$cfg.permissions.allow=@($cfg.permissions.allow)+$missing
$cfg|ConvertTo-Json -Depth 40|Set-Content -LiteralPath $settings -Encoding UTF8
$after=Get-Content -LiteralPath $settings -Raw|ConvertFrom-Json
$old=$before|ConvertFrom-Json
if(($old.deny|ConvertTo-Json -Compress) -cne ($after.permissions.deny|ConvertTo-Json -Compress)){throw 'Deny rules changed'}
if(@($rules|Where-Object {$_ -notin $after.permissions.allow}).Count){throw 'Permission verification failed'}
[pscustomobject]@{added=$missing;backup=$backup;deny_preserved=$true;global_preset_changed=$false}|ConvertTo-Json -Depth 4
