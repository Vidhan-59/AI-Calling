# =====================================================================
#  ShiftEaze AI Calling - start the server
#
#  Usage (from the ai-calling-agent folder, in PowerShell):
#      .\start.ps1
#
#  Options:
#      .\start.ps1 -Port 9000     run on a different port
#      .\start.ps1 -Reload        auto-restart when you edit a .py file
#      .\start.ps1 -SkipChecks    skip the pre-flight checks (fastest)
#      .\start.ps1 -Force         take over the port without asking
#
#  The actual checks live in preflight.py (stdlib-only, so it runs before
#  dependencies are installed). This script just creates the venv if needed,
#  colourises preflight's output, and launches uvicorn.
# =====================================================================

param(
    [int]$Port = 8000,
    [switch]$Reload,
    [switch]$SkipChecks,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " ShiftEaze AI Calling Agent" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------------
# 1. Virtual environment
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "1. Virtual environment" -ForegroundColor Cyan

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "  [WARN] No venv found. Creating one (takes a minute)..." -ForegroundColor Yellow
    python -m venv venv
    if (-not (Test-Path $python)) {
        Write-Host "  [FAIL] Could not create the venv. Is Python 3.10+ installed and on PATH?" -ForegroundColor Red
        Write-Host "         Check with:  python --version" -ForegroundColor Gray
        exit 1
    }
    Write-Host "  [OK]   venv created" -ForegroundColor Green
    Write-Host "  Installing dependencies..." -ForegroundColor Gray
    & $python -m pip install --disable-pip-version-check -q --upgrade pip
    & $python -m pip install --disable-pip-version-check -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] Some packages from requirements.txt failed to install." -ForegroundColor Yellow
        Write-Host "         The pre-flight check below will confirm whether the ones" -ForegroundColor Gray
        Write-Host "         this server actually needs are present." -ForegroundColor Gray
    } else {
        Write-Host "  [OK]   dependencies installed" -ForegroundColor Green
    }
} else {
    Write-Host "  [OK]   venv found" -ForegroundColor Green
}

# ---------------------------------------------------------------------
# 2. Pre-flight checks (dependencies, .env, scenario JSONs)
# ---------------------------------------------------------------------
function Invoke-Preflight {
    # preflight.py emits "LEVEL|message" lines; exit code 1 means a hard failure.
    $output = & $python "preflight.py" 2>&1
    $code = $LASTEXITCODE
    $needsInstall = @()

    foreach ($line in $output) {
        $text = [string]$line
        $parts = $text.Split("|", 2)
        if ($parts.Count -lt 2) {
            if ($text.Trim()) { Write-Host "         $text" -ForegroundColor Gray }
            continue
        }
        $level = $parts[0]
        $msg = $parts[1]
        switch ($level) {
            "OK"      { Write-Host "  [OK]   $msg" -ForegroundColor Green }
            "WARN"    { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
            "FAIL"    { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
            "NOTE"    { Write-Host "         $msg" -ForegroundColor Gray }
            "INSTALL" { $needsInstall = $msg.Split(",") }
            default   { Write-Host "         $text" -ForegroundColor Gray }
        }
    }
    return [PSCustomObject]@{ Code = $code; Install = $needsInstall }
}

if (-not $SkipChecks) {
    Write-Host ""
    Write-Host "2. Pre-flight checks" -ForegroundColor Cyan

    $result = Invoke-Preflight

    if ($result.Install.Count -gt 0) {
        Write-Host "  [WARN] Missing packages: $($result.Install -join ', ')" -ForegroundColor Yellow
        Write-Host "  Installing..." -ForegroundColor Gray
        & $python -m pip install --disable-pip-version-check -q $result.Install
        Write-Host "  [OK]   installed - re-checking" -ForegroundColor Green
        Write-Host ""
        $result = Invoke-Preflight
    }

    if ($result.Code -ne 0) {
        Write-Host ""
        Write-Host "  Fix the [FAIL] item above, then run .\start.ps1 again." -ForegroundColor Red
        Write-Host "  Setup walkthrough: START_HERE.md" -ForegroundColor Gray
        exit 1
    }
}

# ---------------------------------------------------------------------
# 3. Port availability
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "3. Port $Port" -ForegroundColor Cyan

$inUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($inUse) {
    $pid = $inUse[0].OwningProcess
    $owner = Get-Process -Id $pid -ErrorAction SilentlyContinue
    $name = if ($owner) { $owner.ProcessName } else { "unknown" }
    Write-Host "  [WARN] Already in use by $name (PID $pid)" -ForegroundColor Yellow

    $takeOver = $Force
    if (-not $Force) {
        $answer = Read-Host "  Stop it and take over the port? [y/N]"
        $takeOver = ($answer -eq "y")
    }

    if ($takeOver) {
        Stop-Process -Id $pid -Force
        Start-Sleep -Seconds 2
        Write-Host "  [OK]   stopped - port free" -ForegroundColor Green
    } else {
        Write-Host "  Leaving it alone. Use another port:  .\start.ps1 -Port 8001" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "  [OK]   free" -ForegroundColor Green
}

# ---------------------------------------------------------------------
# 4. Go
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host " Starting on http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Interactive API docs : http://127.0.0.1:$Port/docs"
Write-Host " Scenario list        : http://127.0.0.1:$Port/scenarios"
Write-Host " Postman collection   : postman\ShiftEaze_AI_Calling.postman_collection.json"
Write-Host ""
Write-Host " Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

$uvicornArgs = @("-u", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) { $uvicornArgs += "--reload" }

& $python @uvicornArgs
