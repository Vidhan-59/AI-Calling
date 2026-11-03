# Start the OmniVoice TTS service.
#
#   .\tts_service\run-tts.ps1                 # port 8100, GPU
#   .\tts_service\run-tts.ps1 -Port 8101
#   .\tts_service\run-tts.ps1 -Mock           # no GPU/model: tone generator, for wiring tests
#
# Settings come from the TTS_* environment variables (see tts_service/settings.py)
# and, if present, from the repo's .env (TTS_SERVICE_TOKEN etc.).
# First start downloads the model from Hugging Face (one time) and loads it
# onto the GPU - the service answers /health with "loading" until then.

param(
    [int]$Port = 8100,
    [string]$BindHost = "127.0.0.1",
    [switch]$Mock
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The venv lives outside the repo by default (torch + CUDA is ~6 GB; the project
# drive may be small). Override with TTS_VENV.
$venv = if ($env:TTS_VENV) { $env:TTS_VENV } else { Join-Path $env:USERPROFILE ".cache\ai-calling\tts-venv" }
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $python)) { $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $python)) {
    Write-Host "TTS venv missing. Create it first - see tts_service/README.md" -ForegroundColor Red
    exit 1
}

# Pick up TTS_* lines from .env without overriding what's already set.
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*(TTS_[A-Z_]+)\s*=\s*(.*?)\s*$' -and -not [Environment]::GetEnvironmentVariable($Matches[1])) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
        }
    }
}
# The app-side URL/token/timeouts share a prefix with the service's own
# settings; only the token is meaningful to both, the rest are harmless here.
if ($Mock) { $env:TTS_ENGINE = "mock" }
# Defaults that keep the big/user data off a small project drive: a model
# folder fetched once into ~/.cache/ai-calling/omnivoice-model (else the
# service downloads from Hugging Face), and the voice library next to it.
$cache = Join-Path $env:USERPROFILE ".cache\ai-calling"
if (-not $env:TTS_MODEL -and (Test-Path (Join-Path $cache "omnivoice-model\config.json"))) {
    $env:TTS_MODEL = Join-Path $cache "omnivoice-model"
}
if (-not $env:TTS_VOICES_DIR) { $env:TTS_VOICES_DIR = Join-Path $cache "voices" }
# Hugging Face downloads: quiet progress bars in logs.
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

Write-Host "OmniVoice TTS service on http://${BindHost}:$Port  (engine: $(if ($env:TTS_ENGINE) { $env:TTS_ENGINE } else { 'omnivoice' }))" -ForegroundColor Cyan
& $python -m uvicorn tts_service.app:app --host $BindHost --port $Port --log-level info
