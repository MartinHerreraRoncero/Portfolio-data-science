[CmdletBinding()]
param([switch]$Apply)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$settings=Join-Path $env:USERPROFILE '.gemini/antigravity-cli/settings.json'
$cfg=Get-Content -LiteralPath $settings -Raw|ConvertFrom-Json
$before=$cfg.permissions|ConvertTo-Json -Depth 30 -Compress
$paths=@('experiments/reanchor_layout_catalog.py','experiments/compare_layout_answers.py','tests/test_layout_catalog.py','tests/test_layout_answer_effect.py','evaluation/layout_answer_effect_2026-10-27/README.md')
$rules=@($paths|ForEach-Object {'write_file(D:/Portfolio-data-science/RAG/'+$_+')'})
$rules+=@('command(C:\Users\marti\AppData\Local\Programs\Python\Python311\python.exe .orchestrator/scripts/run_layout_answer_effect_tests.py)')
$missing=@($rules|Where-Object {$_ -notin $cfg.permissions.allow})
if(-not $Apply){$missing;return}
$backup=Join-Path $root ('.orchestrator/recovery/layout-effect-permissions-'+[guid]::NewGuid().ToString('N')+'.json')
Set-Content -LiteralPath $backup -Value $before -Encoding UTF8
$cfg.permissions.allow=@($cfg.permissions.allow)+$missing
$cfg|ConvertTo-Json -Depth 40|Set-Content -LiteralPath $settings -Encoding UTF8
$after=Get-Content -LiteralPath $settings -Raw|ConvertFrom-Json
$old=$before|ConvertFrom-Json
if(($old.deny|ConvertTo-Json -Compress) -cne ($after.permissions.deny|ConvertTo-Json -Compress)){throw 'Deny rules changed'}
if(@($rules|Where-Object {$_ -notin $after.permissions.allow}).Count){throw 'Permission verification failed'}
[pscustomobject]@{added=$missing;backup=$backup;deny_preserved=$true;global_preset_changed=$false}|ConvertTo-Json -Depth 4
