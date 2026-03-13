# Complete merge by taking all changes from remote
$logPath = ".cursor\debug.log"

function Write-DebugLog {
    param($sessionId, $runId, $hypothesisId, $location, $message, $data)
    $logEntry = @{
        sessionId = $sessionId
        runId = $runId
        hypothesisId = $hypothesisId
        location = $location
        message = $message
        data = $data
        timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    } | ConvertTo-Json -Compress
    Add-Content -Path $logPath -Value $logEntry
}

$sessionId = "debug-session"
$runId = "merge-completion"

# #region agent log
$gitStatus = git status --porcelain 2>&1 | Out-String
$currentBranch = git branch --show-current
$mergeInProgress = Test-Path ".git\MERGE_HEAD"

Write-DebugLog -sessionId $sessionId -runId $runId -hypothesisId "MERGE" -location "complete_merge.ps1:22" -message "Checking merge state before completion" -data @{
    gitStatus = $gitStatus
    currentBranch = $currentBranch
    mergeInProgress = $mergeInProgress
    coreLongpaths = (git config --get core.longpaths)
}
# #endregion

# Abort current merge if in progress
if ($mergeInProgress) {
    # #region agent log
    Write-Host "Aborting current merge..."
    $abortOutput = git merge --abort 2>&1 | Out-String
    $abortExitCode = $LASTEXITCODE
    
    Write-DebugLog -sessionId $sessionId -runId $runId -hypothesisId "MERGE" -location "complete_merge.ps1:35" -message "Merge abort result" -data @{
        exitCode = $abortExitCode
        output = $abortOutput
    }
    # #endregion
}

# #region agent log
Write-Host "Fetching latest from remote..."
$fetchOutput = git fetch origin main 2>&1 | Out-String
$fetchExitCode = $LASTEXITCODE

Write-DebugLog -sessionId $sessionId -runId $runId -hypothesisId "MERGE" -location "complete_merge.ps1:46" -message "Fetch result" -data @{
    exitCode = $fetchExitCode
    output = $fetchOutput
}
# #endregion

# #region agent log
Write-Host "Resetting to match remote (taking all remote changes)..."
$resetOutput = git reset --hard origin/main 2>&1 | Out-String
$resetExitCode = $LASTEXITCODE

Write-DebugLog -sessionId $sessionId -runId $runId -hypothesisId "MERGE" -location "complete_merge.ps1:55" -message "Reset to remote result" -data @{
    exitCode = $resetExitCode
    output = $resetOutput
}
# #endregion

# #region agent log
$finalStatus = git status --porcelain 2>&1 | Out-String
$finalBranch = git branch --show-current

Write-DebugLog -sessionId $sessionId -runId $runId -hypothesisId "MERGE" -location "complete_merge.ps1:64" -message "Final state after merge completion" -data @{
    gitStatus = $finalStatus
    currentBranch = $finalBranch
    mergeInProgress = (Test-Path ".git\MERGE_HEAD")
    success = ($resetExitCode -eq 0) -and -not (Test-Path ".git\MERGE_HEAD")
}
# #endregion

if ($resetExitCode -eq 0) {
    Write-Host "SUCCESS: Repository is now in sync with remote (all remote changes applied)"
} else {
    Write-Host "WARNING: Some issues occurred. Check output above."
}




