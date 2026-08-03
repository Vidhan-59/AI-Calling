# # =====================================================================
# #  ShiftEaze AI Calling - ONE COMMAND TO RUN EVERYTHING
# #
# #      .\run-all.ps1
# #
# #  Does the whole sequence you would otherwise do by hand:
# #    1. creates the venv / installs dependencies if needed
# #    2. frees port 8000 (reliably - see note below)
# #    3. pre-flight checks (dependencies, API key, scenario JSONs)
# #    4. starts the API server
# #    5. starts a public tunnel and reads back its URL
# #    6. writes that URL into .env as NGROK_URL
# #    7. verifies all three legs Twilio needs (HTTPS, TwiML, WebSocket)
# #    8. watches the tunnel and re-creates it automatically if it dies
# #
# #  The server is started BEFORE the tunnel on purpose. localhost.run probes the
# #  backend when the tunnel is created; if nothing is listening on the port yet
# #  it marks the tunnel dead and serves a permanent 503. Server first, tunnel
# #  second. (This is only possible because NGROK_URL is re-read per call by
# #  public_url.py rather than captured at boot.)
# #
# #  Ctrl+C stops the server AND the tunnel together.
# #
# #  Options:
# #    -Port 8001            different port
# #    -Tunnel ngrok         use ngrok instead of the SSH tunnel
# #    -Tunnel none          server only, no tunnel (text testing)
# #    -NoAutoRecover        warn instead of re-creating a dead tunnel
# #    -SkipChecks           skip pre-flight checks
# #
# #  Why kills are done via CIM/Stop-Process and not pkill: on Windows,
# #  `pkill` from Git Bash does not reliably kill these processes, so a stale
# #  server keeps holding the port, the new one silently fails to bind, and you
# #  end up debugging a server that is not the one running. That cost a wasted
# #  test call once already.
# # =====================================================================

# param(
#     [int]$Port = 8000,
#     [ValidateSet('ssh', 'ngrok', 'none')][string]$Tunnel = 'ssh',
#     [switch]$NoAutoRecover,
#     [switch]$SkipChecks
# )

# $ErrorActionPreference = "Stop"
# Set-Location -Path $PSScriptRoot

# $python     = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
# $serverLog  = Join-Path $PSScriptRoot "server.log"
# $tunnelLog  = Join-Path $PSScriptRoot "tunnel.log"
# $envPath    = Join-Path $PSScriptRoot ".env"

# $script:ServerProc = $null
# $script:TunnelProc = $null
# $script:PublicUrl  = $null

# function Say     ($m) { Write-Host $m }
# function Ok      ($m) { Write-Host "  [OK]   $m"   -ForegroundColor Green }
# function Warn    ($m) { Write-Host "  [WARN] $m"   -ForegroundColor Yellow }
# function Fail    ($m) { Write-Host "  [FAIL] $m"   -ForegroundColor Red }
# function Info    ($m) { Write-Host "         $m"   -ForegroundColor Gray }
# function Section ($m) { Write-Host ""; Write-Host $m -ForegroundColor Cyan }

# # ---------------------------------------------------------------------
# # helpers
# # ---------------------------------------------------------------------

# function Stop-UvicornProcesses {
#     $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
#              Where-Object { $_.CommandLine -like '*uvicorn*main:app*' }
#     foreach ($p in $procs) {
#         try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
#     }
#     return ($procs | Measure-Object).Count
# }

# function Stop-TunnelProcesses {
#     # SSH reverse tunnels to localhost.run, plus any ngrok we started.
#     $ssh = Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'" -ErrorAction SilentlyContinue |
#            Where-Object { $_.CommandLine -like '*localhost.run*' }
#     foreach ($p in $ssh) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {} }
#     Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
# }

# function Test-UrlOk ([string]$url, [int]$timeoutSec = 10) {
#     try {
#         $r = Invoke-WebRequest -Uri $url -TimeoutSec $timeoutSec -UseBasicParsing -ErrorAction Stop
#         return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400)
#     } catch { return $false }
# }

# function Set-EnvPublicUrl ([string]$url) {
#     # Rewrite only the NGROK_URL line, preserving everything else in .env.
#     if (-not (Test-Path $envPath)) {
#         if (Test-Path ".env.example") { Copy-Item ".env.example" $envPath }
#         else { Fail ".env is missing and there is no .env.example to copy."; exit 1 }
#     }
#     $lines = Get-Content $envPath -Encoding UTF8
#     if ($lines -match '^\s*NGROK_URL\s*=') {
#         $lines = $lines | ForEach-Object {
#             if ($_ -match '^\s*NGROK_URL\s*=') { "NGROK_URL=$url" } else { $_ }
#         }
#     } else {
#         $lines += "NGROK_URL=$url"
#     }
#     Set-Content -Path $envPath -Value $lines -Encoding UTF8
# }

# function Start-SshTunnel ([int]$port) {
#     Remove-Item $tunnelLog -ErrorAction SilentlyContinue
#     $args = @(
#         "-o", "StrictHostKeyChecking=no",
#         "-o", "ServerAliveInterval=30",
#         "-o", "ServerAliveCountMax=3",
#         "-o", "ExitOnForwardFailure=yes",
#         "-R", "80:localhost:$port",
#         "nokey@localhost.run"
#     )
#     $proc = Start-Process -FilePath "ssh.exe" -ArgumentList $args `
#               -RedirectStandardOutput $tunnelLog -RedirectStandardError "$tunnelLog.err" `
#               -WindowStyle Hidden -PassThru

#     # localhost.run prints the assigned URL a few seconds after connecting.
#     for ($i = 0; $i -lt 25; $i++) {
#         Start-Sleep -Seconds 1
#         if (Test-Path $tunnelLog) {
#             $text = Get-Content $tunnelLog -Raw -ErrorAction SilentlyContinue
#             if ($text -match 'https://[a-z0-9]+\.lhr\.life') {
#                 return @{ Proc = $proc; Url = $Matches[0] }
#             }
#         }
#         if ($proc.HasExited) { break }
#     }
#     return @{ Proc = $proc; Url = $null }
# }

# function Start-NgrokTunnel ([int]$port) {
#     Remove-Item $tunnelLog -ErrorAction SilentlyContinue
#     $proc = Start-Process -FilePath "ngrok" -ArgumentList @("http", "$port", "--log=stdout") `
#               -RedirectStandardOutput $tunnelLog -RedirectStandardError "$tunnelLog.err" `
#               -WindowStyle Hidden -PassThru

#     for ($i = 0; $i -lt 20; $i++) {
#         Start-Sleep -Seconds 1
#         # ngrok's local API is the reliable place to read the public URL.
#         foreach ($apiPort in 4040, 4041) {
#             try {
#                 $t = Invoke-RestMethod "http://127.0.0.1:$apiPort/api/tunnels" -TimeoutSec 3 -ErrorAction Stop
#                 $https = $t.tunnels | Where-Object { $_.public_url -like "https://*" } | Select-Object -First 1
#                 if ($https) { return @{ Proc = $proc; Url = $https.public_url } }
#             } catch {}
#         }
#         if ($proc.HasExited) { break }
#     }
#     return @{ Proc = $proc; Url = $null }
# }

# function New-Tunnel ([int]$port, [string]$kind) {
#     if ($kind -eq 'ngrok') { return Start-NgrokTunnel $port }
#     return Start-SshTunnel $port
# }

# function Stop-Everything {
#     Say ""
#     Section "Shutting down"
#     if ($script:ServerProc -and -not $script:ServerProc.HasExited) {
#         try { Stop-Process -Id $script:ServerProc.Id -Force -ErrorAction Stop; Ok "server stopped" } catch {}
#     }
#     Stop-UvicornProcesses | Out-Null
#     Stop-TunnelProcesses
#     Ok "tunnel stopped"
#     Say ""
# }

# # ---------------------------------------------------------------------
# # main
# # ---------------------------------------------------------------------

# Say ""
# Write-Host "=====================================================" -ForegroundColor Cyan
# Write-Host " ShiftEaze AI Calling - full stack" -ForegroundColor Cyan
# Write-Host "=====================================================" -ForegroundColor Cyan

# try {
#     # -- 1. venv -------------------------------------------------------
#     Section "1. Virtual environment"
#     if (-not (Test-Path $python)) {
#         Warn "No venv - creating it (takes a minute)"
#         python -m venv venv
#         if (-not (Test-Path $python)) {
#             Fail "Could not create the venv. Is Python 3.10+ installed and on PATH?"
#             exit 1
#         }
#         & $python -m pip install --disable-pip-version-check -q --upgrade pip
#         & $python -m pip install --disable-pip-version-check -q -r requirements.txt
#         Ok "venv created, dependencies installed"
#     } else {
#         Ok "venv found"
#     }

#     # -- 2. free the port ---------------------------------------------
#     Section "2. Freeing port $Port"
#     $killed = Stop-UvicornProcesses
#     Stop-TunnelProcesses
#     if ($killed -gt 0) { Ok "stopped $killed old server process(es)" } else { Ok "nothing to clean up" }
#     Start-Sleep -Seconds 2

#     $held = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
#     if ($held) {
#         Fail "Port $Port is still held by PID $($held[0].OwningProcess) (not one of ours)."
#         Info "Use a different port:  .\run-all.ps1 -Port 8001"
#         exit 1
#     }
#     Ok "port $Port free"

#     # -- 3. pre-flight ------------------------------------------------
#     # Runs before the tunnel exists, so the NGROK_URL line preflight prints
#     # would show the previous (dead) value - filtered out here because this
#     # script sets it authoritatively in step 6.
#     if (-not $SkipChecks) {
#         Section "3. Pre-flight checks"
#         $output = & $python "preflight.py" 2>&1
#         $preflightCode = $LASTEXITCODE
#         $install = @()
#         foreach ($line in $output) {
#             $parts = ([string]$line).Split("|", 2)
#             if ($parts.Count -lt 2) { continue }
#             if ($parts[1] -like "*NGROK_URL*") { continue }
#             switch ($parts[0]) {
#                 "OK"      { Ok   $parts[1] }
#                 "WARN"    { Warn $parts[1] }
#                 "FAIL"    { Fail $parts[1] }
#                 "NOTE"    { Info $parts[1] }
#                 "INSTALL" { $install = $parts[1].Split(",") }
#             }
#         }
#         if ($install.Count -gt 0) {
#             Warn "installing missing packages: $($install -join ', ')"
#             & $python -m pip install --disable-pip-version-check -q $install
#             Ok "installed"
#             $preflightCode = 0
#         }
#         if ($preflightCode -ne 0) {
#             Fail "Fix the item above, then run .\run-all.ps1 again"
#             Info "Setup walkthrough: START_HERE.md"
#             exit 1
#         }
#     }

#     # -- 4. server ----------------------------------------------------
#     Section "4. API server"
#     Remove-Item $serverLog -ErrorAction SilentlyContinue
#     $script:ServerProc = Start-Process -FilePath $python `
#         -ArgumentList @("-u", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$Port") `
#         -RedirectStandardOutput $serverLog -RedirectStandardError "$serverLog.err" `
#         -WindowStyle Hidden -PassThru

#     $up = $false
#     for ($i = 0; $i -lt 25; $i++) {
#         Start-Sleep -Seconds 1
#         if (Test-UrlOk "http://127.0.0.1:$Port/" 3) { $up = $true; break }
#         if ($script:ServerProc.HasExited) { break }
#     }
#     if (-not $up) {
#         Fail "Server did not come up."
#         foreach ($f in "$serverLog.err", $serverLog) {
#             if (Test-Path $f) { Info "--- $f ---"; Get-Content $f -Tail 15 | ForEach-Object { Info $_ } }
#         }
#         exit 1
#     }
#     Ok "server listening on http://127.0.0.1:$Port (PID $($script:ServerProc.Id))"

#     # -- 5. tunnel (AFTER the server, see header note) ----------------
#     if ($Tunnel -ne 'none') {
#         Section "5. Public tunnel ($Tunnel)"
#         $t = New-Tunnel $Port $Tunnel
#         $script:TunnelProc = $t.Proc
#         if (-not $t.Url) {
#             Fail "Could not obtain a tunnel URL."
#             if (Test-Path "$tunnelLog.err") {
#                 Info "tunnel error output:"
#                 Get-Content "$tunnelLog.err" -Tail 6 | ForEach-Object { Info $_ }
#             }
#             if ($Tunnel -eq 'ssh') { Info "Try:  .\run-all.ps1 -Tunnel ngrok" }
#             else { Info "Try:  .\run-all.ps1 -Tunnel ssh" }
#             Info "Or run without a tunnel (text testing only):  .\run-all.ps1 -Tunnel none"
#             exit 1
#         }
#         $script:PublicUrl = $t.Url
#         Ok "tunnel up: $($t.Url)"

#         # -- 6. wire it in ---------------------------------------------
#         Set-EnvPublicUrl $t.Url
#         Ok "NGROK_URL written to .env (re-read per call - no restart needed)"
#     } else {
#         Section "5. Public tunnel"
#         Warn "skipped (-Tunnel none) - text testing works, real calls will be refused"
#     }

#     # -- 7. verify the three legs Twilio needs ------------------------
#     if ($script:PublicUrl) {
#         Section "7. Verifying what Twilio needs"

#         # Retry: a freshly created tunnel can take a few seconds to route.
#         $httpsOk = $false
#         for ($i = 0; $i -lt 6; $i++) {
#             if (Test-UrlOk "$($script:PublicUrl)/" 10) { $httpsOk = $true; break }
#             Start-Sleep -Seconds 3
#         }
#         if ($httpsOk) { Ok "HTTPS reachable   $($script:PublicUrl)/" }
#         else { Warn "HTTPS check failed - a real call will be refused by the guard" }

#         $twimlUrl = "$($script:PublicUrl)/scenarios/voice?ref=selftest&scenario_id=shift_confirm&language=hi&attempt=1"
#         try {
#             $r = Invoke-WebRequest -Uri $twimlUrl -Method Post -TimeoutSec 12 -UseBasicParsing -ErrorAction Stop
#             if ($r.Content -match 'ConversationRelay') { Ok "TwiML endpoint returns valid ConversationRelay XML" }
#             else { Warn "TwiML endpoint answered but without a ConversationRelay element" }
#         } catch { Warn "TwiML endpoint check failed: $($_.Exception.Message)" }

#         $wsUri = ($script:PublicUrl -replace '^https://', 'wss://') + "/scenarios/media-stream"
#         try {
#             $ws = New-Object System.Net.WebSockets.ClientWebSocket
#             $cts = New-Object System.Threading.CancellationTokenSource
#             $cts.CancelAfter(12000)
#             $ws.ConnectAsync([Uri]$wsUri, $cts.Token).Wait()
#             if ($ws.State -eq 'Open') { Ok "WebSocket leg open   $wsUri" } else { Warn "WebSocket state: $($ws.State)" }
#             $ws.Dispose()
#         } catch { Warn "WebSocket check failed: $($_.Exception.InnerException.Message)" }
#     }

#     # -- 8. ready ------------------------------------------------------
#     Say ""
#     Write-Host "=====================================================" -ForegroundColor Green
#     Write-Host " READY" -ForegroundColor Green
#     Write-Host "=====================================================" -ForegroundColor Green
#     Say ""
#     Say " Local API      : http://127.0.0.1:$Port"
#     Say " API docs       : http://127.0.0.1:$Port/docs"
#     Say " Scenarios      : http://127.0.0.1:$Port/scenarios"
#     if ($script:PublicUrl) {
#         Say " Public URL     : $($script:PublicUrl)   (already saved to .env)"
#         Say ""
#         Say " You can place a real call now - Postman folder '4. Place a real call'"
#     } else {
#         Say ""
#         Say " No tunnel: use Postman folder '3. Test without a phone'"
#     }
#     Say ""
#     Say " Press Ctrl+C to stop the server and the tunnel together." -ForegroundColor Gray
#     Say ""

#     # -- 9. watch ------------------------------------------------------
#     $tick = 0
#     while ($true) {
#         Start-Sleep -Seconds 5
#         $tick += 5

#         if ($script:ServerProc.HasExited) {
#             Fail "Server exited unexpectedly."
#             foreach ($f in "$serverLog.err", $serverLog) {
#                 if (Test-Path $f) { Info "--- $f ---"; Get-Content $f -Tail 15 | ForEach-Object { Info $_ } }
#             }
#             break
#         }

#         # Check the tunnel every 30s. Free tunnels expire on their own, and a
#         # dead one is invisible until a call fails - so surface it here.
#         if ($script:PublicUrl -and ($tick % 30 -eq 0)) {
#             if (-not (Test-UrlOk $script:PublicUrl 10)) {
#                 Warn "$(Get-Date -Format 'HH:mm:ss')  tunnel is down: $($script:PublicUrl)"

#                 if ($NoAutoRecover) {
#                     Info "Re-run .\run-all.ps1 to get a fresh tunnel (-NoAutoRecover is set)."
#                 } else {
#                     Info "re-creating it..."
#                     Stop-TunnelProcesses
#                     Start-Sleep -Seconds 2
#                     $t2 = New-Tunnel $Port $Tunnel
#                     if ($t2.Url) {
#                         $script:TunnelProc = $t2.Proc
#                         $script:PublicUrl = $t2.Url
#                         Set-EnvPublicUrl $t2.Url
#                         Ok "new tunnel: $($t2.Url)  (.env updated; no server restart needed)"
#                     } else {
#                         Warn "could not re-create the tunnel - will retry in 30s"
#                     }
#                 }
#             }
#         }
#     }
# }
# finally {
#     Stop-Everything
# }




















# =====================================================================
#  ShiftEaze AI Calling - ONE COMMAND TO RUN EVERYTHING
#
#      .\run-all.ps1
#
#  Does the whole sequence you would otherwise do by hand:
#    1. creates the venv / installs dependencies if needed
#    2. frees port 8000 (reliably - see note below)
#    3. pre-flight checks (dependencies, API key, scenario JSONs)
#    4. starts the API server
#    5. starts a public tunnel and reads back its URL
#    6. writes that URL into .env as NGROK_URL
#    7. verifies all three legs Twilio needs before accepting calls
#    8. watches the tunnel and re-creates it automatically if it dies
#
#  The server is started BEFORE the tunnel on purpose. localhost.run probes the
#  backend when the tunnel is created; if nothing is listening on the port yet
#  it marks the tunnel dead and serves a permanent 503. Server first, tunnel
#  second. (This is only possible because NGROK_URL is re-read per call by
#  public_url.py rather than captured at boot.)
#
#  Ctrl+C stops the server AND the tunnel together.
#
#  Options:
#    -Port 8001            different port
#    -Tunnel ngrok         use ngrok instead of the SSH tunnel
#    -Tunnel none          server only, no tunnel (text testing)
#    -NoAutoRecover        warn instead of re-creating a dead tunnel
#    -SkipChecks           skip pre-flight checks
#
#  Why kills are done via CIM/Stop-Process and not pkill: on Windows,
#  `pkill` from Git Bash does not reliably kill these processes, so a stale
#  server keeps holding the port, the new one silently fails to bind, and you
#  end up debugging a server that is not the one running. That cost a wasted
#  test call once already.
# =====================================================================

param(
    [int]$Port = 8000,
    [ValidateSet('ssh', 'ngrok', 'none')][string]$Tunnel = 'ssh',
    [switch]$NoAutoRecover,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python     = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$serverLog  = Join-Path $PSScriptRoot "server.log"
$tunnelLog  = Join-Path $PSScriptRoot "tunnel.log"
$envPath    = Join-Path $PSScriptRoot ".env"

$script:ServerProc = $null
$script:TunnelProc = $null
$script:PublicUrl  = $null

function Say     ($m) { Write-Host $m }
function Ok      ($m) { Write-Host "  [OK]   $m"   -ForegroundColor Green }
function Warn    ($m) { Write-Host "  [WARN] $m"   -ForegroundColor Yellow }
function Fail    ($m) { Write-Host "  [FAIL] $m"   -ForegroundColor Red }
function Info    ($m) { Write-Host "         $m"   -ForegroundColor Gray }
function Section ($m) { Write-Host ""; Write-Host $m -ForegroundColor Cyan }

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

function Stop-UvicornProcesses {
    $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -like '*uvicorn*main:app*' }
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
    }
    return ($procs | Measure-Object).Count
}

function Stop-TunnelProcesses {
    # SSH reverse tunnels to localhost.run, plus any ngrok we started.
    $ssh = Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'" -ErrorAction SilentlyContinue |
           Where-Object { $_.CommandLine -like '*localhost.run*' }
    foreach ($p in $ssh) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {} }
    Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Test-UrlOk ([string]$url, [int]$timeoutSec = 10) {
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec $timeoutSec -UseBasicParsing -ErrorAction Stop
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400)
    } catch { return $false }
}

function Set-EnvPublicUrl ([string]$url) {
    # Rewrite only the NGROK_URL line, preserving everything else in .env.
    if (-not (Test-Path $envPath)) {
        if (Test-Path ".env.example") { Copy-Item ".env.example" $envPath }
        else { Fail ".env is missing and there is no .env.example to copy."; exit 1 }
    }
    $lines = Get-Content $envPath -Encoding UTF8
    if ($lines -match '^\s*NGROK_URL\s*=') {
        $lines = $lines | ForEach-Object {
            if ($_ -match '^\s*NGROK_URL\s*=') { "NGROK_URL=$url" } else { $_ }
        }
    } else {
        $lines += "NGROK_URL=$url"
    }
    Set-Content -Path $envPath -Value $lines -Encoding UTF8
}

function Start-SshTunnel ([int]$port) {
    Remove-Item $tunnelLog -ErrorAction SilentlyContinue
    Remove-Item "$tunnelLog.err" -ErrorAction SilentlyContinue

    # Keep the SSH session alive and fail the process if the reverse
    # forwarding cannot actually be established.
    $args = @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "TCPKeepAlive=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=15",
        "-R", "80:127.0.0.1:$port",
        "nokey@localhost.run"
    )

    $proc = Start-Process -FilePath "ssh.exe" -ArgumentList $args `
              -RedirectStandardOutput $tunnelLog -RedirectStandardError "$tunnelLog.err" `
              -WindowStyle Hidden -PassThru

    # localhost.run prints the assigned URL after the SSH connection is ready.
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1

        if ($proc.HasExited) { break }

        if (Test-Path $tunnelLog) {
            $logText = Get-Content $tunnelLog -Raw -ErrorAction SilentlyContinue
            if ($logText -match '(https://[a-z0-9-]+\.lhr\.life)') {
                return @{ Proc = $proc; Url = $Matches[1] }
            }
        }
    }

    return @{ Proc = $proc; Url = $null }
}

function Start-NgrokTunnel ([int]$port) {
    Remove-Item $tunnelLog -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath "ngrok" -ArgumentList @("http", "$port", "--log=stdout") `
              -RedirectStandardOutput $tunnelLog -RedirectStandardError "$tunnelLog.err" `
              -WindowStyle Hidden -PassThru

    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 1
        # ngrok's local API is the reliable place to read the public URL.
        foreach ($apiPort in 4040, 4041) {
            try {
                $t = Invoke-RestMethod "http://127.0.0.1:$apiPort/api/tunnels" -TimeoutSec 3 -ErrorAction Stop
                $https = $t.tunnels | Where-Object { $_.public_url -like "https://*" } | Select-Object -First 1
                if ($https) { return @{ Proc = $proc; Url = $https.public_url } }
            } catch {}
        }
        if ($proc.HasExited) { break }
    }
    return @{ Proc = $proc; Url = $null }
}

function Test-PublicTunnel ([string]$url, [int]$port) {
    if ([string]::IsNullOrWhiteSpace($url)) { return $false }

    # 1) Public HTTPS must actually reach our local server.
    $httpsOk = $false
    for ($i = 0; $i -lt 5; $i++) {
        if (Test-UrlOk $url 10) {
            $httpsOk = $true
            break
        }
        Start-Sleep -Seconds 2
    }

    if (-not $httpsOk) {
        Warn "Public URL is not reachable: $url"
        return $false
    }

    # 2) Twilio must be able to fetch the TwiML endpoint.
    $twimlUrl = "$url/scenarios/voice?ref=selftest&scenario_id=shift_confirm&language=hi&attempt=1"
    try {
        $r = Invoke-WebRequest -Uri $twimlUrl -Method Post -TimeoutSec 15 `
             -UseBasicParsing -ErrorAction Stop

        if ($r.StatusCode -lt 200 -or $r.StatusCode -ge 400) {
            Warn "TwiML endpoint returned HTTP $($r.StatusCode)"
            return $false
        }

        if ($r.Content -notmatch 'ConversationRelay') {
            Warn "TwiML endpoint responded but ConversationRelay XML was not found"
            return $false
        }
    } catch {
        Warn "TwiML endpoint is not reachable: $($_.Exception.Message)"
        return $false
    }

    # 3) The WebSocket route must be reachable through the same tunnel.
    $wsUri = ($url -replace '^https://', 'wss://') + "/scenarios/media-stream"
    $ws = $null
    $cts = $null

    try {
        $ws = New-Object System.Net.WebSockets.ClientWebSocket
        $cts = New-Object System.Threading.CancellationTokenSource
        $cts.CancelAfter(12000)
        $ws.ConnectAsync([Uri]$wsUri, $cts.Token).Wait()

        if ($ws.State -ne 'Open') {
            Warn "WebSocket did not open. State: $($ws.State)"
            return $false
        }
    } catch {
        $msg = $_.Exception.Message
        if ($_.Exception.InnerException) { $msg = $_.Exception.InnerException.Message }
        Warn "WebSocket check failed: $msg"
        return $false
    } finally {
        if ($ws) { $ws.Dispose() }
        if ($cts) { $cts.Dispose() }
    }

    return $true
}

function Start-VerifiedTunnel ([int]$port, [string]$kind) {
    # A tunnel URL can be printed by localhost.run before the public endpoint
    # is actually usable. Never publish it to .env until all checks pass.
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        Info "starting $kind tunnel (attempt $attempt/4)..."

        Stop-TunnelProcesses
        Start-Sleep -Seconds 2

        $t = New-Tunnel $port $kind

        if (-not $t.Url) {
            Warn "No tunnel URL received."
            if ($t.Proc -and $t.Proc.HasExited) {
                Info "tunnel process exited."
            }
            continue
        }

        Info "received tunnel URL: $($t.Url)"

        if (Test-PublicTunnel $t.Url $port) {
            return $t
        }

        Warn "Tunnel URL was assigned but failed end-to-end validation."
        Stop-TunnelProcesses
        Start-Sleep -Seconds 2
    }

    return @{ Proc = $null; Url = $null }
}

function New-Tunnel ([int]$port, [string]$kind) {
    if ($kind -eq 'ngrok') { return Start-NgrokTunnel $port }
    return Start-SshTunnel $port
}

function Stop-Everything {
    Say ""
    Section "Shutting down"
    if ($script:ServerProc -and -not $script:ServerProc.HasExited) {
        try { Stop-Process -Id $script:ServerProc.Id -Force -ErrorAction Stop; Ok "server stopped" } catch {}
    }
    Stop-UvicornProcesses | Out-Null
    Stop-TunnelProcesses
    Ok "tunnel stopped"
    Say ""
}

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

Say ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " ShiftEaze AI Calling - full stack" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

try {
    # -- 1. venv -------------------------------------------------------
    Section "1. Virtual environment"
    if (-not (Test-Path $python)) {
        Warn "No venv - creating it (takes a minute)"
        python -m venv venv
        if (-not (Test-Path $python)) {
            Fail "Could not create the venv. Is Python 3.10+ installed and on PATH?"
            exit 1
        }
        & $python -m pip install --disable-pip-version-check -q --upgrade pip
        & $python -m pip install --disable-pip-version-check -q -r requirements.txt
        Ok "venv created, dependencies installed"
    } else {
        Ok "venv found"
    }

    # -- 2. free the port ---------------------------------------------
    Section "2. Freeing port $Port"
    $killed = Stop-UvicornProcesses
    Stop-TunnelProcesses
    if ($killed -gt 0) { Ok "stopped $killed old server process(es)" } else { Ok "nothing to clean up" }
    Start-Sleep -Seconds 2

    $held = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($held) {
        Fail "Port $Port is still held by PID $($held[0].OwningProcess) (not one of ours)."
        Info "Use a different port:  .\run-all.ps1 -Port 8001"
        exit 1
    }
    Ok "port $Port free"

    # -- 3. pre-flight ------------------------------------------------
    # Runs before the tunnel exists, so the NGROK_URL line preflight prints
    # would show the previous (dead) value - filtered out here because this
    # script sets it authoritatively in step 6.
    if (-not $SkipChecks) {
        Section "3. Pre-flight checks"
        $output = & $python "preflight.py" 2>&1
        $preflightCode = $LASTEXITCODE
        $install = @()
        foreach ($line in $output) {
            $parts = ([string]$line).Split("|", 2)
            if ($parts.Count -lt 2) { continue }
            if ($parts[1] -like "*NGROK_URL*") { continue }
            switch ($parts[0]) {
                "OK"      { Ok   $parts[1] }
                "WARN"    { Warn $parts[1] }
                "FAIL"    { Fail $parts[1] }
                "NOTE"    { Info $parts[1] }
                "INSTALL" { $install = $parts[1].Split(",") }
            }
        }
        if ($install.Count -gt 0) {
            Warn "installing missing packages: $($install -join ', ')"
            & $python -m pip install --disable-pip-version-check -q $install
            Ok "installed"
            $preflightCode = 0
        }
        if ($preflightCode -ne 0) {
            Fail "Fix the item above, then run .\run-all.ps1 again"
            Info "Setup walkthrough: START_HERE.md"
            exit 1
        }
    }

    # -- 4. server ----------------------------------------------------
    Section "4. API server"
    Remove-Item $serverLog -ErrorAction SilentlyContinue
    $script:ServerProc = Start-Process -FilePath $python `
        -ArgumentList @("-u", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$Port") `
        -RedirectStandardOutput $serverLog -RedirectStandardError "$serverLog.err" `
        -WindowStyle Hidden -PassThru

    $up = $false
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Seconds 1
        if (Test-UrlOk "http://127.0.0.1:$Port/" 3) { $up = $true; break }
        if ($script:ServerProc.HasExited) { break }
    }
    if (-not $up) {
        Fail "Server did not come up."
        foreach ($f in "$serverLog.err", $serverLog) {
            if (Test-Path $f) { Info "--- $f ---"; Get-Content $f -Tail 15 | ForEach-Object { Info $_ } }
        }
        exit 1
    }
    Ok "server listening on http://127.0.0.1:$Port (PID $($script:ServerProc.Id))"

    # -- 5. tunnel (AFTER the server, see header note) ----------------
    if ($Tunnel -ne 'none') {
        Section "5. Public tunnel ($Tunnel)"
        $t = Start-VerifiedTunnel $Port $Tunnel
        $script:TunnelProc = $t.Proc

        if (-not $t.Url) {
            Fail "Could not establish a verified public tunnel."
            if (Test-Path "$tunnelLog.err") {
                Info "tunnel error output:"
                Get-Content "$tunnelLog.err" -Tail 10 | ForEach-Object { Info $_ }
            }
            if ($Tunnel -eq 'ssh') { Info "Try:  .\run-all.ps1 -Tunnel ngrok" }
            else { Info "Try:  .\run-all.ps1 -Tunnel ssh" }
            Info "Or run without a tunnel (text testing only):  .\run-all.ps1 -Tunnel none"
            exit 1
        }

        $script:PublicUrl = $t.Url
        Ok "verified tunnel up: $($t.Url)"

        # Only write a URL to .env after the tunnel has passed the same
        # checks Twilio depends on. This prevents a dead lhr.life URL from
        # being stored and used for the next real call.
        Set-EnvPublicUrl $t.Url
        Ok "NGROK_URL written to .env (re-read per call - no restart needed)"
    } else {
        Section "5. Public tunnel"
        Warn "skipped (-Tunnel none) - text testing works, real calls will be refused"
    }

    # -- 7. final tunnel assertion -------------------------------------
    if ($script:PublicUrl) {
        Section "7. Final Twilio connectivity check"
        if (Test-PublicTunnel $script:PublicUrl $Port) {
            Ok "HTTPS + TwiML + WebSocket all reachable"
        } else {
            Fail "Public tunnel failed the final Twilio connectivity check."
            Fail "The URL will NOT be trusted for real calls."
            exit 1
        }
    }

    # -- 8. ready ------------------------------------------------------
    Say ""
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host " READY" -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Say ""
    Say " Local API      : http://127.0.0.1:$Port"
    Say " API docs       : http://127.0.0.1:$Port/docs"
    Say " Scenarios      : http://127.0.0.1:$Port/scenarios"
    if ($script:PublicUrl) {
        Say " Public URL     : $($script:PublicUrl)   (already saved to .env)"
        Say ""
        Say " You can place a real call now - Postman folder '4. Place a real call'"
    } else {
        Say ""
        Say " No tunnel: use Postman folder '3. Test without a phone'"
    }
    Say ""
    Say " Press Ctrl+C to stop the server and the tunnel together." -ForegroundColor Gray
    Say ""

    # -- 9. watch ------------------------------------------------------
    $tick = 0
    while ($true) {
        Start-Sleep -Seconds 5
        $tick += 5

        if ($script:ServerProc.HasExited) {
            Fail "Server exited unexpectedly."
            foreach ($f in "$serverLog.err", $serverLog) {
                if (Test-Path $f) { Info "--- $f ---"; Get-Content $f -Tail 15 | ForEach-Object { Info $_ } }
            }
            break
        }

        # Check the tunnel every 30s. Free tunnels expire on their own, and a
        # dead one is invisible until a call fails - so surface it here.
        if ($script:PublicUrl -and ($tick % 30 -eq 0)) {
            $tunnelProcessDead = $script:TunnelProc -and $script:TunnelProc.HasExited
            $tunnelReachable = Test-UrlOk $script:PublicUrl 10

            if ($tunnelProcessDead -or -not $tunnelReachable) {
                Warn "$(Get-Date -Format 'HH:mm:ss')  tunnel is down: $($script:PublicUrl)"

                if ($NoAutoRecover) {
                    Info "Re-run .\run-all.ps1 to get a fresh tunnel (-NoAutoRecover is set)."
                } else {
                    Info "re-creating it..."
                    Stop-TunnelProcesses
                    Start-Sleep -Seconds 2
                    $t2 = Start-VerifiedTunnel $Port $Tunnel
                    if ($t2.Url) {
                        $script:TunnelProc = $t2.Proc
                        $script:PublicUrl = $t2.Url
                        Set-EnvPublicUrl $t2.Url
                        Ok "verified new tunnel: $($t2.Url)  (.env updated; no server restart needed)"
                    } else {
                        Warn "could not re-create a verified tunnel - will retry in 30s"
                    }
                }
            }
        }
    }
}
finally {
    Stop-Everything
}