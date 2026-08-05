# =====================================================================
#  ShiftEaze AI Calling - ONE-TIME setup for a fresh machine
#
#  Run this ONCE per new machine, after cloning the repo (see step 0 below).
#  After this finishes, use .\run-all.ps1 every day to actually start the
#  server - this script only gets a machine ready, it doesn't start anything
#  long-running itself.
#
#  What it does:
#    1. Checks Python 3.10+ and Git are on PATH
#    2. Creates the venv, installs requirements.txt + requirements-local.txt
#    3. Installs Ollama if missing (winget), makes sure it's running
#    4. Pulls the local LLM model (qwen2.5:7b-instruct by default)
#    5. Runs one real inference and checks Ollama actually used the GPU
#       (size_vram > 0) - this is the whole point of moving to this machine
#    6. Creates .env from .env.example if missing, and stops here so you can
#       fill in real secrets - .env is gitignored on purpose and was never
#       on GitHub, so it has to be set up by hand on every new machine
#    7. If .env already has real secrets, sets SCENARIO_LLM_BACKEND=local
#       automatically (worth it on a GPU - see LOCAL_AI_README.md)
#
#  Options:
#    -Model <name>     pull a different local model (default: qwen2.5:7b-instruct)
#    -SkipOllama       skip Ollama install/pull/GPU-check (Claude-only setup)
# =====================================================================

param(
    [string]$Model = "qwen2.5:7b-instruct",
    [switch]$SkipOllama
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Ok      ($m) { Write-Host "  [OK]   $m"   -ForegroundColor Green }
function Warn    ($m) { Write-Host "  [WARN] $m"   -ForegroundColor Yellow }
function Fail    ($m) { Write-Host "  [FAIL] $m"   -ForegroundColor Red }
function Info    ($m) { Write-Host "         $m"   -ForegroundColor Gray }
function Section ($m) { Write-Host ""; Write-Host $m -ForegroundColor Cyan }

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " ShiftEaze AI Calling - new machine setup" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------
Section "1. Prerequisites"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Fail "Python not found on PATH."
    Info "Install Python 3.10+ from https://python.org/downloads (check 'Add to PATH' during install)"
    Info "or:  winget install --id Python.Python.3.12"
    exit 1
}
$pyVersion = & python --version 2>&1
Ok "$pyVersion found"

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Fail "Git not found on PATH."
    Info "Install from https://git-scm.com/downloads or:  winget install --id Git.Git"
    exit 1
}
Ok "git found"

if (-not (Test-Path (Join-Path $PSScriptRoot "main.py"))) {
    Fail "main.py not found here - this script must run from inside the cloned ai-calling-agent folder."
    Info "Step 0, if you haven't done it yet:"
    Info "  git clone https://github.com/Vidhan-59/AI-Calling.git"
    Info "  cd AI-Calling"
    Info "  .\setup-new-machine.ps1"
    exit 1
}
Ok "running from $PSScriptRoot"

# ---------------------------------------------------------------------
# 2. Virtual environment + dependencies
# ---------------------------------------------------------------------
Section "2. Python environment"

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Info "creating venv..."
    python -m venv venv
    if (-not (Test-Path $python)) {
        Fail "Could not create the venv."
        exit 1
    }
}
Ok "venv ready"

Info "installing requirements.txt (this can take a few minutes on a fresh machine)..."
& $python -m pip install --disable-pip-version-check -q --upgrade pip
& $python -m pip install --disable-pip-version-check -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "requirements.txt install failed - see output above"; exit 1 }
Ok "requirements.txt installed"

if (-not $SkipOllama) {
    Info "installing requirements-local.txt (faster-whisper, piper-tts)..."
    & $python -m pip install --disable-pip-version-check -q -r requirements-local.txt
    if ($LASTEXITCODE -ne 0) { Fail "requirements-local.txt install failed - see output above"; exit 1 }
    Ok "requirements-local.txt installed"
}

# ---------------------------------------------------------------------
# 3. Ollama + GPU check
# ---------------------------------------------------------------------
if (-not $SkipOllama) {
    Section "3. Ollama + GPU"

    function Sync-PathFromRegistry {
        # A winget install updates PATH in the registry, but this already-running
        # PowerShell process keeps its own copy in memory - Get-Command won't see
        # a just-installed exe until $env:Path is refreshed from Machine+User.
        $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        $user = [System.Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = "$machine;$user"
    }

    function Find-OllamaExe {
        $cmd = Get-Command ollama -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        # winget's default install location if PATH still hasn't caught up.
        $fallback = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
        if (Test-Path $fallback) { return $fallback }
        return $null
    }

    $ollamaExe = Find-OllamaExe
    if (-not $ollamaExe) {
        Warn "Ollama not found - installing via winget..."
        winget install --id Ollama.Ollama --source winget --accept-package-agreements --accept-source-agreements
        Start-Sleep -Seconds 5
        Sync-PathFromRegistry
        $ollamaExe = Find-OllamaExe
        if (-not $ollamaExe) {
            Fail "Ollama install didn't complete. Install manually from https://ollama.com, then re-run this script."
            exit 1
        }
    }
    # Make sure the rest of this script (and this terminal session) can call
    # the bare `ollama` command, not just the resolved path.
    $ollamaDir = Split-Path $ollamaExe -Parent
    if ($env:Path -notlike "*$ollamaDir*") { $env:Path = "$env:Path;$ollamaDir" }
    Ok "ollama found ($ollamaExe)"

    # Ollama's Windows installer normally starts a background service on
    # 127.0.0.1:11434. If it's not up yet (e.g. just installed, tray app
    # hasn't launched), start it ourselves rather than failing here.
    $ollamaUp = $false
    try {
        Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop | Out-Null
        $ollamaUp = $true
    } catch {}

    if (-not $ollamaUp) {
        Info "starting ollama serve in the background..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 1
            try {
                Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop | Out-Null
                $ollamaUp = $true
                break
            } catch {}
        }
    }
    if (-not $ollamaUp) {
        Fail "Could not reach Ollama at http://127.0.0.1:11434 - is it installed correctly?"
        exit 1
    }
    Ok "ollama running"

    Info "pulling $Model (a few GB - one-time download)..."
    ollama pull $Model
    if ($LASTEXITCODE -ne 0) { Fail "ollama pull $Model failed"; exit 1 }
    Ok "$Model pulled"

    Info "running one real inference to confirm GPU offload..."
    try {
        Invoke-RestMethod "http://127.0.0.1:11434/api/generate" -Method Post -TimeoutSec 120 -Body (
            @{ model = $Model; prompt = "hi"; stream = $false } | ConvertTo-Json
        ) -ContentType "application/json" | Out-Null
    } catch {
        Fail "Test inference failed: $($_.Exception.Message)"
        exit 1
    }

    $ps = Invoke-RestMethod "http://127.0.0.1:11434/api/ps" -TimeoutSec 10
    $loaded = $ps.models | Where-Object { $_.name -like "$Model*" } | Select-Object -First 1
    if ($loaded -and $loaded.size_vram -gt 0) {
        $vramGB = [math]::Round($loaded.size_vram / 1GB, 2)
        Ok "GPU offload confirmed - $vramGB GB in VRAM"
        Info "Expect ~1-3s per turn now instead of the 33-60s measured on CPU-only hardware."
    } elseif ($loaded) {
        Warn "Model loaded but size_vram is 0 - running on CPU, not GPU."
        Info "Check: nvidia-smi shows the GPU, and your NVIDIA driver is current."
        Info "Ollama auto-detects CUDA - no config needed if the driver is right, so this"
        Info "usually means an outdated/missing NVIDIA driver rather than an Ollama setting."
    } else {
        Warn "Could not confirm GPU usage - 'ollama ps' showed no loaded model."
    }
}

# ---------------------------------------------------------------------
# 4. .env
# ---------------------------------------------------------------------
Section "4. .env (secrets - never came from GitHub, .env is gitignored on purpose)"

$envPath = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item ".env.example" $envPath
    Warn ".env created from .env.example - it has PLACEHOLDER values, not real secrets."
    Info "Edit .env now and fill in:"
    Info "  ANTHROPIC_API_KEY   - https://console.anthropic.com/settings/keys"
    Info "  TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER - https://console.twilio.com"
    Info "Easiest: securely copy the real .env from your other machine (USB / private transfer)"
    Info "instead of retyping - just never commit it or paste it anywhere public."
    Info ""
    Info "Re-run this script after editing .env, or just run .\run-all.ps1 directly."
} else {
    $envContent = Get-Content $envPath -Raw
    if ($envContent -match "your_anthropic_api_key_here|your_twilio_account_sid") {
        Warn ".env exists but still has placeholder values - fill in real secrets before running the server."
    } else {
        Ok ".env has real values"

        if (-not $SkipOllama) {
            # Worth it on a GPU: 1-3s/turn instead of 33-60s. See LOCAL_AI_README.md.
            if ($envContent -match '(?m)^\s*SCENARIO_LLM_BACKEND\s*=\s*claude\s*$') {
                $envContent = $envContent -replace '(?m)^\s*SCENARIO_LLM_BACKEND\s*=\s*claude\s*$', 'SCENARIO_LLM_BACKEND=local'
                Set-Content -Path $envPath -Value $envContent -Encoding UTF8 -NoNewline
                Ok "SCENARIO_LLM_BACKEND set to 'local' (this machine has a GPU - see LOCAL_AI_README.md)"
            } elseif ($envContent -match '(?m)^\s*SCENARIO_LLM_BACKEND\s*=\s*local\s*$') {
                Ok "SCENARIO_LLM_BACKEND already 'local'"
            } else {
                "`nSCENARIO_LLM_BACKEND=local" | Add-Content -Path $envPath -Encoding UTF8
                Ok "SCENARIO_LLM_BACKEND=local added"
            }
        }
    }
}

# ---------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host " Setup complete" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Next step:  .\run-all.ps1"
Write-Host ""
