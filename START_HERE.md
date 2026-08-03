# START HERE — running the ShiftEaze AI Calling agent

Everything you need to start the server and make a call, every time.

- **First time ever?** Do §1 (once), then §2.
- **Just want to run it again?** Skip to §2 — it's two commands.
- **Making a real phone call?** §2 then §5.

---

## 1. One-time setup

You only ever do this once on this machine.

### 1.1 Check Python

```powershell
python --version
```

Needs **3.10 or newer**. If it's missing, install from [python.org](https://www.python.org/downloads/) and tick *"Add Python to PATH"* during install.

### 1.2 Allow PowerShell scripts to run

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Answer `Y`. Without this, Windows blocks `start.ps1` and the venv activation script.

### 1.3 Set your Anthropic API key

```powershell
cd "C:\Users\Vidha\OneDrive\Desktop\Veracity\AI-Calling\ai-calling-agent"
Copy-Item .env.example .env -ErrorAction SilentlyContinue
notepad .env
```

Set this one line and save:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

> Twilio and `NGROK_URL` are **only** needed for real phone calls (§5). Everything else — all five scenarios, all three languages, every branch — works with just this key.

That's it. `start.ps1` creates the virtual environment and installs dependencies for you on first run.

---

## 2. Start the server (every time)

```powershell
cd "C:\Users\Vidha\OneDrive\Desktop\Veracity\AI-Calling\ai-calling-agent"
.\start.ps1
```

You'll see pre-flight checks, then:

```
=====================================================
 Starting on http://127.0.0.1:8000
=====================================================

 Interactive API docs : http://127.0.0.1:8000/docs
 Scenario list        : http://127.0.0.1:8000/scenarios

 Press Ctrl+C to stop.
```

**Leave this window open** — closing it stops the server. Open a second PowerShell window (or use Postman) to send requests.

### Options

| Command | What it does |
|---|---|
| `.\start.ps1` | Normal start on port 8000 |
| `.\start.ps1 -Reload` | Auto-restarts when you edit a `.py` file (use while developing) |
| `.\start.ps1 -Port 8001` | Different port — useful if 8000 is taken |
| `.\start.ps1 -SkipChecks` | Skip pre-flight checks (fastest start) |

### If you'd rather not use the script

```powershell
cd "C:\Users\Vidha\OneDrive\Desktop\Veracity\AI-Calling\ai-calling-agent"
.\venv\Scripts\Activate.ps1
uvicorn main:app --port 8000
```

### Confirm it's alive

Open <http://127.0.0.1:8000/scenarios> in a browser — you should see all five scenarios as JSON. Or <http://127.0.0.1:8000/docs> for the interactive API explorer.

---

## 3. Import the Postman collection (once)

1. Open Postman → **Import** (top-left)
2. Drag in `postman\ShiftEaze_AI_Calling.postman_collection.json`
3. Done — a **ShiftEaze AI Calling** collection appears in the sidebar

### Set your values once

Click the collection name → **Variables** tab → edit the **Current value** column:

| Variable | Set it to |
|---|---|
| `base_url` | `http://127.0.0.1:8000` (already correct for local) |
| `to_phone_number` | The number to call, e.g. `+919426005911` |
| `employee_name` | `Amit` |
| `terminal` | `CT3` |
| `supervisor_name` | `Suresh` |

**Save** (Ctrl+S). Every request now picks these up — you don't edit request bodies to change the phone number or name.

### The collection is organised in the order you should use it

| Folder | Use it for |
|---|---|
| **0. Health** | Confirm the server is up |
| **1. Discovery** | See the five scenarios and their branches |
| **2. Preview** | Proof-read the exact spoken lines — **do this before calling anyone** |
| **3. Test without a phone** | Hold full conversations over HTTP, no Twilio needed |
| **4. Place a real call** | One request per scenario |
| **5. Outcomes** | Read back what happened |
| **6. Generic agent** | The non-ShiftEaze free-form agent |

---

## 4. Test a scenario without spending a phone call

This is the fastest way to check behaviour, and it needs nothing but your Anthropic key.

In Postman: **3. Test without a phone** → **Turn 1 — Hindi, operator says नहीं** → **Send**.

You get back the branch the AI matched, its confidence, the scripted reply, and the outcome code:

```json
{
  "opening_line": "नमस्ते Amit, मैं ShiftEaze की ओर से …",
  "operator_said": "नहीं",
  "agent_reply": "कृपया अपनी अनुपस्थिति का कारण बताएँ।",
  "matched_branch": "no",
  "outcome": "awaiting_absence_reason",
  "call_ended": false,
  "classification": { "confidence": 0.98, "detected_language": "hi" }
}
```

Then send **Turn 2** to continue the same conversation and watch the reason get captured verbatim.

**Worth running:** the **Negation trap** request. It sends `"haan nahi aa paunga yaar, tabiyat kharab hai"` — which opens with *haan* (yes) but means no. A keyword-matching IVR records the exact opposite answer; the classifier gets it right and, because the reason was volunteered in the same breath, skips the redundant "tell us why" question.

The **Batch** requests run every branch of a scenario in one go — use them after editing any scenario JSON. Open Postman's **Console** (bottom-left) to see the branch-by-branch log.

---

## 5. Make a real phone call

### 5.1 Fill in Twilio + a public tunnel

Real calls need Twilio to reach your laptop, which needs a public HTTPS URL.

Start a tunnel in its **own** PowerShell window and leave it open:

```powershell
ngrok http 8000
```

Copy the `https://xxxx.ngrok-free.app` line it prints.

### 5.2 Put it in `.env`

```powershell
notepad .env
```

```env
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
NGROK_URL=https://xxxx.ngrok-free.app
```

### 5.3 Restart the server

`.env` is read at boot, so: **Ctrl+C** in the server window, then `.\start.ps1` again.

> ⚠️ **The tunnel URL changes every time you restart ngrok** on the free tier. Each restart means: copy new URL → update `.env` → restart the server. This is the single most common reason a call rings and then goes silent.

### 5.4 Preview first, then call

1. **2. Preview** → pick your scenario → **Send**. Read the `opening_line`. Check `missing_facts` is `[]`.
2. **4. Place a real call** → same scenario → **Send**.

The response confirms exactly what the operator will hear:

```json
{
  "status": "success",
  "call_sid": "CAxxxxxxxx",
  "voice": "hi-IN-Neural2-B",
  "will_say": "नमस्ते Amit, मैं ShiftEaze की ओर से …"
}
```

`call_sid` is saved automatically, so **5. Outcomes → Get one scenario call** works right after with no copy-paste.

### 5.5 Watch it happen

The server window logs every step live:

```
[CAxxx] setup: shift_confirm/hi attempt=1
[CAxxx] "नहीं आ सकता भाई" -> branch=no outcome=awaiting_absence_reason end=False
[CAxxx] "तबीयत खराब है" -> branch=no outcome=absent_with_reason end=True
```

---

## 6. Editing what the agent says

All five scripts live in `scenarios\*.json` — one file per case, all three languages inside.

1. Edit the file (e.g. change a `reply`, add a phrase to `keywords`)
2. In Postman: **1. Discovery → Reload scenarios from disk** → **Send**
3. Next call uses it — **no server restart**

If the JSON is malformed you get a `400` naming the exact problem, so a broken script can't half-load.

The `reply` text is spoken **verbatim** — the AI picks which branch applies but never writes the words. That's what keeps compliance-sensitive wording exactly as your team approved it.

---

## 7. Two rules for writing facts

The facts you send go straight into a text-to-speech engine.

1. **Never send `14:00`.** Send `"दोपहर 2"` / `"બપોરે 2"` / `"2 in the afternoon"`. TTS reads `14:00` aloud as "fourteen hundred".
2. **The Hindi/Gujarati scripts already contain बजे / વાગ્યે.** Send `"सुबह 8"`, **not** `"सुबह 8 बजे"` — otherwise the operator hears *"subah 8 baje baje"*.

`POST /scenarios/preview` exists to catch exactly this.

---

## 8. Stopping and restarting

| To do this | Do that |
|---|---|
| Stop the server | **Ctrl+C** in the server window |
| Stop a stuck server (reliable) | `Get-CimInstance Win32_Process -Filter "Name='python.exe'" \| Where-Object { $_.CommandLine -like '*uvicorn*' } \| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }` |
| Check nothing is still holding the port | `Get-NetTCPConnection -LocalPort 8000 -State Listen` — no output means free |
| Restart after editing `.env` | Ctrl+C, then `.\start.ps1` |
| Restart after editing a scenario JSON | Not needed — `POST /scenarios/reload` |
| Restart after editing a `.py` file | Ctrl+C then `.\start.ps1`, or use `-Reload` to automate it |

---

## 9. Troubleshooting

### First thing to check when a real call misbehaves: Twilio's own error log

Twilio records exactly why a call failed, and it's far more useful than guessing:

```powershell
$sid = (Select-String -Path .env -Pattern '^TWILIO_ACCOUNT_SID=' ).Line.Split('=')[1].Trim()
$tok = (Select-String -Path .env -Pattern '^TWILIO_AUTH_TOKEN=' ).Line.Split('=')[1].Trim()
$pair = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${sid}:${tok}"))
(Invoke-RestMethod "https://monitor.twilio.com/v1/Alerts?PageSize=5" `
  -Headers @{ Authorization = "Basic $pair" }).alerts |
  Select-Object date_created, error_code, alert_text | Format-List
```

| Error code | Means |
|---|---|
| **11200** | Twilio couldn't fetch your TwiML — stale tunnel URL, or server not restarted after editing `.env`. The `alert_text` shows the exact URL and HTTP status it got. |
| **12100 / 12200** | Twilio fetched the TwiML but couldn't parse/validate it. |
| **31xxx** | ConversationRelay / media problem. |


| Symptom | Cause and fix |
|---|---|
| `start.ps1 cannot be loaded because running scripts is disabled` | Run §1.2 |
| `[FAIL] ANTHROPIC_API_KEY is not set` | Set it in `.env` (§1.3), then re-run |
| `Port 8000 is already in use` | The script offers to stop it — answer `y`. Or `.\start.ps1 -Port 8001` |
| Postman: `Could not send request` | Server isn't running. Check the server window, run **0. Health** |
| `400 Missing required facts` | Run the matching **2. Preview** request — its `missing_facts` names them |
| Hindi/Gujarati shows as `?????` | You used `curl` in Git Bash — it mangles non-ASCII bodies. Use Postman or PowerShell's `Invoke-RestMethod` |
| Real call: `500 NGROK_URL is not configured` | §5.2 — and restart the server after editing `.env` |
| Real call: `502 Not placing the call: your public URL … is unreachable` | Working as intended — your tunnel is dead. Restart it, update `.env`, restart the server. This guard exists so a dead tunnel can't burn a call. |
| **"Press any key to continue"** before the agent speaks | Normal on a Twilio **trial** account — Twilio prepends that to every outbound call. Upgrade the account to remove it. |
| **"We're sorry, an application error occurred"** then hangup | Twilio couldn't fetch the TwiML — almost always a stale tunnel URL in `.env`, or the server wasn't restarted after updating it. Confirm with the Twilio alerts check below (look for error **11200**). |
| Server seems to ignore your `.env` edit | An older server process is still holding port 8000, so your restart silently failed to bind. Kill it properly (see §8) and start again — `pkill` from Git Bash does **not** reliably kill Windows processes. |
| `ModuleNotFoundError` | `.\start.ps1` installs what's needed; if it persists, run `.\venv\Scripts\python.exe -m pip install -r requirements.txt` |

---

## Files worth knowing about

```
start.ps1                  Start the server - this is your entry point
preflight.py               The checks start.ps1 runs (deps, .env, scenario JSONs)
.env                       Your keys - never commit this
scenarios\*.json           The five call scripts, all 3 languages - edit these
postman\ShiftEaze_AI_Calling.postman_collection.json    Import into Postman
START_HERE.md              This file
POSTMAN_GUIDE.md           Every endpoint with full request bodies
README.md                  Architecture and how it works
```

You can also run the checks on their own, without starting anything:

```powershell
.\venv\Scripts\python.exe preflight.py
```

---

## Where to go next

- **[POSTMAN_GUIDE.md](POSTMAN_GUIDE.md)** — every endpoint, full request bodies, outcome codes, and the current status of the telephony leg
- **[README.md](README.md)** — architecture: why Twilio ConversationRelay instead of an audio-native API, and how the AI classification layer works
