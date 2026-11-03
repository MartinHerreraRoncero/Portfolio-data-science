[CmdletBinding()]
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$settings = Join-Path $env:USERPROFILE '.gemini/antigravity-cli/settings.json'
$cfg = Get-Content -LiteralPath $settings -Raw | ConvertFrom-Json
$before = $cfg.permissions | ConvertTo-Json -Depth 30 -Compress
$rules = @('command(python -m unittest discover -s tests -q)', 'command(python -m unittest discover -s tests -p test_verifier_capacity.py -v)', 'write_file(D:/Portfolio-data-science/RAG/experiments/compare_verifier_capacity.py)', 'write_file(D:/Portfolio-data-science/RAG/tests/test_verifier_capacity.py)', 'write_file(D:/Portfolio-data-science/RAG/evaluation/verifier_capacity_2026-10-26/README_RUNNER.md)')
$rules += 'command(git grep -n "api/show")'
$rules += 'command(git status)'
$rules += 'command(git status --short)'
$missing = @($rules | Where-Object { $_ -notin $cfg.permissions.allow })
if (-not $Apply) { $missing; return }
$backup = Join-Path $root ('.orchestrator/recovery/verifier-permissions-' + [guid]::NewGuid().ToString('N') + '.json')
Set-Content -LiteralPath $backup -Value $before -Encoding UTF8
$cfg.permissions.allow = @($cfg.permissions.allow) + $missing
$cfg | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $settings -Encoding UTF8
$after = Get-Content -LiteralPath $settings -Raw | ConvertFrom-Json
$old = $before | ConvertFrom-Json
if (($old.deny | ConvertTo-Json -Compress) -cne ($after.permissions.deny | ConvertTo-Json -Compress)) { throw 'Deny rules changed unexpectedly' }
if (@($rules | Where-Object { $_ -notin $after.permissions.allow }).Count) { throw 'Permission verification failed' }
[pscustomobject]@{added=$missing;previous_permissions_backup=$backup;deny_rules_preserved=$true;global_preset_changed=$false} | ConvertTo-Json -Depth 5
