param([string]$RepoRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'

function Read-Report([string]$RelativePath) {
    Get-Content -Raw -LiteralPath (Join-Path $RepoRoot $RelativePath) | ConvertFrom-Json
}
function Get-Median($Values) {
    $sorted = @($Values | Sort-Object)
    if (-not $sorted.Count) { return $null }
    $mid = [int][Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$mid] }
    return ($sorted[$mid - 1] + $sorted[$mid]) / 2
}
function Get-Rows($Report, [bool]$Sequence = $false) {
    foreach ($run in $Report.runs) {
        if ($Sequence) {
            foreach ($step in $run.sequence.steps) {
                [pscustomobject]@{key="$($run.repeat)/$($run.branch)/$($run.sequence.id)/$($step.id)"; branch=$run.branch; repeat=$run.repeat; case=$step}
            }
        } else {
            [pscustomobject]@{key="$($run.fixture)/$($run.repeat)/$($run.branch)/$($run.case.id)"; branch=$run.branch; repeat=$run.repeat; case=$run.case}
        }
    }
}
function Normalize-CorrectionIdentity([string]$Text) {
    # Only the random original fact UUID inside a correction source tag changes.
    # Do not normalize numbers, dates, hashes of replacement text, or wording.
    return $Text -replace '(?<=source=correction:)[a-f0-9]{32}', '<original-fact-uuid>'
}
function Compare-Contexts($Old, $New, [bool]$Sequence=$false, [string]$Branch='') {
    $newRows = @{}
    foreach ($row in Get-Rows $New $Sequence) { $newRows[$row.key] = $row.case }
    $exact = $normalized = $checked = 0
    $differences = @()
    foreach ($row in Get-Rows $Old $Sequence) {
        if ($Branch -and $row.branch -ne $Branch) { continue }
        $checked++
        if (-not $newRows.ContainsKey($row.key)) { $differences += $row.key; continue }
        $oldContext = @($row.case.memory_contexts | Select-Object -First 1)
        $newContext = @($newRows[$row.key].memory_contexts | Select-Object -First 1)
        $oldText = if ($oldContext.Count) { [string]$oldContext[0].rendered } else { '' }
        $newText = if ($newContext.Count) { [string]$newContext[0].rendered } else { '' }
        if ($oldText -ceq $newText) { $exact++ }
        if ((Normalize-CorrectionIdentity $oldText) -ceq (Normalize-CorrectionIdentity $newText)) { $normalized++ }
        else { $differences += $row.key }
    }
    return @{checked=$checked; exact_equal=$exact; correction_uuid_normalized_equal=$normalized; differences=$differences; comparison='first_memory_context_only_not_model_reply_equivalence'}
}
function Live-Cost($Report, [bool]$Sequence=$false) {
    $rows = @(Get-Rows $Report $Sequence)
    $result = @{}
    foreach ($branch in 'raw','enhanced') {
        $subset = @($rows | Where-Object branch -eq $branch)
        $calls = @($subset.case.model_calls)
        $repeats = @{}
        foreach ($rep in 1..3) { $repeats[[string]$rep] = Get-Median @($subset | Where-Object repeat -eq $rep | ForEach-Object {$_.case.seconds}) }
        $result[$branch] = @{executions=$subset.Count; calls=$calls.Count;
            reported_tokens=($calls.metrics_delta.total_tokens_chat | Measure-Object -Sum).Sum;
            median_seconds=(Get-Median @($subset.case.seconds)); repeat_medians=$repeats;
            injected_chars=($subset.case.injected_chars | Measure-Object -Sum).Sum;
            execution_failures=@($subset | Where-Object {$_.case.execution_failure_layer}).Count;
            evidence_leaks=@($subset | Where-Object {$_.case.evidence_leaks.Count}).Count}
    }
    return $result
}

$currentB = Read-Report 'docs/MEMORY_STAGE_B_20261002_054418/paired.json'
$currentE = Read-Report 'docs/MEMORY_STAGE_E_20261002_053558/paired.json'
$currentC = Read-Report 'docs/MEMORY_STAGE_C_20261002_053706/paired.json'
$currentD = Read-Report 'docs/MEMORY_STAGE_D_20261002_053706/paired.json'
$currentSequence = Read-Report 'docs/MEMORY_STAGE_E_SEQUENCE_20261002_053706/sequence.json'
$liveE = Read-Report 'docs/MEMORY_STAGE_E_20261002_035145/paired.json'
$liveNatural = Read-Report 'docs/MEMORY_STAGE_E_20261002_035454/paired.json'
$liveSequence = Read-Report 'docs/MEMORY_STAGE_E_SEQUENCE_20261002_035155/sequence.json'
$maintenance = Read-Report 'docs/MEMORY_STAGE_F_MAINTENANCE_20261002_053707/maintenance.json'
$audit = Read-Report 'docs/AGENT_FRESH_AUDIT_20261002_053631/environment.json'
[xml]$auditXml = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot 'docs/AGENT_FRESH_AUDIT_20261002_053631/pytest.xml')
$comparisons = @{
    b_pre_integration_local = Compare-Contexts (Read-Report 'docs/MEMORY_STAGE_B_20261002_012048/paired.json') $currentB
    e_pre_patch_local = Compare-Contexts (Read-Report 'docs/MEMORY_STAGE_E_20261002_035446/paired.json') $currentE
    c_pre_patch_local = Compare-Contexts (Read-Report 'docs/MEMORY_STAGE_C_20261002_040119/paired.json') $currentC
    d_pre_patch_local = Compare-Contexts (Read-Report 'docs/MEMORY_STAGE_D_20261002_040129/paired.json') $currentD
    e_live_initial_context = Compare-Contexts $liveE $currentE
    natural_live_initial_context = Compare-Contexts $liveNatural $currentE
    sequence_enhanced_live_initial_context = Compare-Contexts $liveSequence $currentSequence $true 'enhanced'
    c_old_live_diagnostic = Compare-Contexts (Read-Report 'docs/MEMORY_STAGE_C_20261002_020255/paired.json') $currentC
    d_old_live_diagnostic = Compare-Contexts (Read-Report 'docs/MEMORY_STAGE_D_20261002_023909/paired.json') $currentD
}
$costBefore = Read-Report 'docs/MEMORY_STAGE_B_COST_20261002_011555.json'
$costFinal = Read-Report 'docs/MEMORY_STAGE_B_COST_20261002_054406.json'
$oldCostRows = @{}
foreach ($row in $costBefore.runs | Where-Object kind -eq 'measured') { $oldCostRows["$($row.candidate_count)/$($row.branch)/$($row.repeat)"] = $row }
$costDifferences = @()
foreach ($row in $costFinal.runs | Where-Object kind -eq 'measured') {
    $key = "$($row.candidate_count)/$($row.branch)/$($row.repeat)"
    $before = $oldCostRows[$key]
    if ($row.rendered_hash -ne $before.rendered_hash -or [string]::Join('|',$row.source_ids) -cne [string]::Join('|',$before.source_ids)) { $costDifferences += $key }
}
$hashDifferences = @()
foreach ($property in $currentSequence.source_snapshot_after.PSObject.Properties) {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $RepoRoot $property.Name)).Hash.ToLowerInvariant() -ne $property.Value) { $hashDifferences += $property.Name }
}
foreach ($property in $currentE.profile.protected_config_hashes.PSObject.Properties) {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $property.Name).Hash.ToLowerInvariant() -ne $property.Value) { $hashDifferences += $property.Name }
}
$checks = @{
    source_and_protected_hashes_current = -not $hashDifferences.Count
    audit_passed = $audit.exit_code -eq 0 -and [int]$auditXml.testsuites.testsuite.failures -eq 0 -and [int]$auditXml.testsuites.testsuite.errors -eq 0
    maintenance_passed = $maintenance.status -eq 'completed' -and $maintenance.repair_source_tables_unchanged -and $maintenance.repair_render_identical -and $maintenance.revoked_text_not_restored_to_delivery -and $maintenance.rollback_keeps_revocation
    final_retrieval_reports_stable = -not @(@($currentB,$currentC,$currentD,$currentE,$currentSequence) | Where-Object {$_.status -ne 'completed' -or -not $_.source_unchanged_during_run -or -not $_.protected_configs_unchanged}).Count
    final_b_cost_source_and_render_identical = -not $costDifferences.Count -and $costFinal.source_unchanged_during_run
    saved_profile_new_flags_off = -not ($currentE.profile.memory_policy.search_keys_enabled -or $currentE.profile.memory_policy.current_understanding_enabled -or $currentE.profile.memory_policy.evidence_package_enabled -or $currentE.profile.memory_policy.history_needs_enabled)
}
foreach ($name in 'b_pre_integration_local','e_pre_patch_local','c_pre_patch_local','d_pre_patch_local','e_live_initial_context','natural_live_initial_context','sequence_enhanced_live_initial_context') {
    $checks[$name] = $comparisons[$name].checked -gt 0 -and $comparisons[$name].checked -eq $comparisons[$name].correction_uuid_normalized_equal
}
$verification = @{generated_at=(Get-Date).ToString('o'); checks=$checks; comparisons=$comparisons; source_hash_differences=$hashDifferences;
    e_actual_reply_cost=(Live-Cost $liveE); natural_actual_reply_cost=(Live-Cost $liveNatural); sequence_actual_reply_cost=(Live-Cost $liveSequence $true);
    final_retrieval_summary=$currentE.summary; audit=$audit; junit_items=[int]$auditXml.testsuites.testsuite.tests;
    final_b_retrieval_summary=$currentB.summary; final_b_cost_summary=$costFinal.summary; b_cost_differences=$costDifferences;
    maintenance=@{source_unchanged_during_run=$maintenance.source_unchanged_during_run; cost_total=$maintenance.cost_total; phases=@($maintenance.phases | Select-Object name,seconds,cost)};
    limitations=@('Concurrent final retrieval runs are integrity checks, not isolated latency benchmarks.', 'First memory context equality does not prove equality of all model tool calls or replies.', 'Old C and D live cohorts are reported separately if prompt versions differ.', 'Reported tokens exclude missing usage on failed upstream calls.');
    acceptance_status=if (@($checks.Values | Where-Object {-not $_}).Count) {'review_required'} else {'engineering_verified_production_not_enabled'}}
$out = Join-Path $RepoRoot ('docs/MEMORY_UPGRADE_VERIFICATION_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
$verification | ConvertTo-Json -Depth 25 | Set-Content -LiteralPath $out -Encoding utf8
Write-Output ('REPORT ' + $out)
$verification | Select-Object acceptance_status,checks,comparisons,junit_items,source_hash_differences | ConvertTo-Json -Depth 5
