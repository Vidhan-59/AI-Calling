# ShiftEaze AI Calling — Postman Guide

How to drive the five ShiftEaze calling scenarios in English, Hindi and Gujarati.

Base URL for local runs: `http://127.0.0.1:8000`

> ### 📦 Ready-made collection — import this instead of typing requests by hand
>
> **`postman/ShiftEaze_AI_Calling.postman_collection.json`** — 35 pre-built requests covering
> every endpoint below, organised in the order you should use them, with all five scenarios
> in all three languages, assertions, and `call_sid` auto-captured after placing a call.
>
> Postman → **Import** → drag the file in. Then set your values once under the collection's
> **Variables** tab (`to_phone_number`, `employee_name`, `terminal`, `supervisor_name`) and
> every request picks them up.
>
> **To start the server:** see **[START_HERE.md](START_HERE.md)** — `.\start.ps1` and you're up.

> **Postman setup for Hindi/Gujarati:** in the request **Body** tab choose **raw → JSON**. Postman sends UTF‑8 correctly, so Devanagari and Gujarati text in the body works as‑is. (Do **not** use `curl` in Git Bash on Windows for these — it mangles non‑ASCII request bodies into `?????`. Postman, Insomnia, and PowerShell's `Invoke-RestMethod` are all fine.)

---

## The five scenarios

| `scenario_id` | Case | Type | Branches the AI can pick |
|---|---|---|---|
| `shift_confirm` | 01 Shift Confirmation Reminder | Conversational | `yes` `no` `repeat` `dispute` `already_confirmed` `supervisor` `wrong_number` |
| `punch_in_remind` | 02 Punch‑In Reminder | Reactive IVR | `acknowledged` `already_punched` `technical_issue` `supervisor` `wrong_number` |
| `absence_priority` | 03 Absence Prediction Priority Call | Conversational | `yes` `no` `tentative` `leave_filed` `why_calling` `supervisor` `wrong_number` |
| `ot_request` | 04 Overtime Request | Conversational | `yes` `no` `asks_pay` `asks_hours` `recent_ot` `supervisor` `wrong_number` |
| `punch_out_remind` | 05 Punch‑Out Reminder | Reactive IVR | `still_working` `already_punched` `technical_issue` `supervisor` `wrong_number` |

`language` is `en`, `hi`, or `gu` on every request. Every scenario also has a `silence` path (no answer / unintelligible) and honours `attempt: 1` or `2` for the follow‑up wording from the question script.

---

## Which facts each scenario requires

The agent may **only** speak facts you send. A call is rejected with `400` if a required fact is missing, rather than letting an operator hear `your duty is on {date}`.

| `scenario_id` | Required in `facts` | Also used if present |
|---|---|---|
| `shift_confirm` | `employee_name`, `date`, `time`, `shift_name`, `terminal` | `supervisor_name` |
| `punch_in_remind` | `employee_name`, `time`, `terminal` | `supervisor_name` |
| `absence_priority` | `employee_name`, `date`, `time`, `terminal` | `supervisor_name` |
| `ot_request` | `employee_name`, `terminal`, `shortfall_count`, `duration` | `supervisor_name` |
| `punch_out_remind` | `employee_name`, `terminal` | `supervisor_name` |

### ⚠️ Write facts the way they should be *spoken*, not the way they're stored

These strings go straight into a text‑to‑speech engine. Two rules that matter:

1. **Don't send `14:00`.** Send `"do baje dopahar"` / `"दोपहर 2"` / `"બપોરે 2"`. A TTS engine reads `14:00` as "fourteen hundred", which operators don't parse.
2. **The Hindi/Gujarati templates already contain the word for "o'clock"** (`बजे` / `વાગ્યે`). So send `"सुबह 8"`, **not** `"सुबह 8 बजे"` — otherwise the operator hears *"subah 8 baje baje"*. Use `POST /scenarios/preview` (below) to catch exactly this before dialling anyone.

---

## 1. See what's available

**GET** `{{base}}/scenarios`

Returns all five scenarios with their languages, branch keys and required facts. No body.

**GET** `{{base}}/scenarios/shift_confirm`

The full knowledge base for one scenario — every script line in all three languages, the keyword hints, the outcome codes, and the guardrail facts the AI is working from.

---

## 2. Preview the exact spoken lines (no call placed) — **do this first**

**POST** `{{base}}/scenarios/preview`

```json
{
  "scenario_id": "shift_confirm",
  "language": "hi",
  "attempt": 1,
  "facts": {
    "employee_name": "Amit",
    "date": "2 अगस्त",
    "time": "सुबह 8",
    "shift_name": "Day",
    "terminal": "CT3",
    "supervisor_name": "Suresh"
  }
}
```

You get back the opening line, the silence close, and every branch's reply exactly as the operator would hear it, plus the `voice` / `tts_language` / `transcription_language` that call will use and any `missing_facts`. Proof‑read the Hindi/Gujarati here before spending a real call on it.

---

## 3. Have a conversation with no phone, no Twilio, no tunnel

This is the fastest way to test behaviour, and it costs nothing but Claude tokens.

**POST** `{{base}}/scenarios/test-turn`

```json
{
  "scenario_id": "shift_confirm",
  "language": "hi",
  "session_id": "my-test-1",
  "facts": {
    "employee_name": "Amit",
    "date": "2 अगस्त",
    "time": "सुबह 8",
    "shift_name": "Day",
    "terminal": "CT3",
    "supervisor_name": "Suresh"
  },
  "message": "नहीं"
}
```

Response tells you the branch the AI matched, its confidence, the detected language, the scripted reply, the outcome code, and whether the call would end:

```json
{
  "session_id": "my-test-1",
  "opening_line": "नमस्ते Amit, मैं ShiftEaze की ओर से …",
  "operator_said": "नहीं",
  "agent_reply": "कृपया अपनी अनुपस्थिति का कारण बताएँ।",
  "matched_branch": "no",
  "outcome": "awaiting_absence_reason",
  "call_ended": false,
  "collected": {},
  "classification": { "confidence": 0.98, "detected_language": "hi", "rationale": "…" }
}
```

**Continue the same conversation** by re‑sending with the *same* `session_id` and the next utterance (you can drop `facts` — the session remembers them):

```json
{
  "scenario_id": "shift_confirm",
  "language": "hi",
  "session_id": "my-test-1",
  "message": "मेरे पिताजी की तबीयत खराब है, हॉस्पिटल जाना है"
}
```

→ `matched_branch: "no"`, `call_ended: true`, `outcome: "absent_with_reason"`, and the reason captured verbatim in `collected.absence_reason`.

**Simulate silence / an unintelligible answer:** send `"message": ""`.

### Test every branch at once

**POST** `{{base}}/scenarios/test-batch`

```json
{
  "scenario_id": "ot_request",
  "language": "gu",
  "facts": {
    "employee_name": "Kiran",
    "terminal": "CT3",
    "shortfall_count": "3",
    "duration": "8 કલાક",
    "supervisor_name": "Suresh"
  },
  "utterances": [
    "હા ભાઈ આવી જઈશ",
    "ના યાર આજે નહીં આવી શકું",
    "કેટલા પૈસા મળશે એ કહો પહેલા",
    "કેટલા કલાક કામ કરવાનું છે?",
    "મેં ગઈકાલે પણ ઓટી કરી હતી ભાઈ"
  ]
}
```

Each utterance runs as its own fresh conversation, so one request exercises one branch per line. Run this after editing any scenario JSON.

---

## 4. Place a real phone call

**Prerequisites:** `ANTHROPIC_API_KEY`, `TWILIO_*` and a real public `NGROK_URL` in `.env`, and the server restarted after editing `.env`. See the caveat in §7 before relying on this.

**POST** `{{base}}/scenarios/call`

```json
{
  "scenario_id": "shift_confirm",
  "language": "hi",
  "attempt": 1,
  "to_phone_number": "+919426005911",
  "rgid": "RG12345",
  "facts": {
    "employee_name": "Amit",
    "date": "2 अगस्त",
    "time": "सुबह 8",
    "shift_name": "Day",
    "terminal": "CT3",
    "supervisor_name": "Suresh"
  }
}
```

Response:

```json
{
  "status": "success",
  "call_sid": "CAxxxxxxxx",
  "scenario_id": "shift_confirm",
  "language": "hi",
  "voice": "hi-IN-Neural2-B",
  "will_say": "नमस्ते Amit, मैं ShiftEaze की ओर से …",
  "track_at": "/scenario-calls/CAxxxxxxxx"
}
```

`to_phone_number` must be E.164 — include the country code (`+91…` for India).

---

## 5. Read the outcome

**GET** `{{base}}/scenario-calls` — all placed calls, newest first
**GET** `{{base}}/scenario-calls?scenario_id=ot_request` — filtered
**GET** `{{base}}/scenario-calls/CAxxxxxxxx` — one call in full

The single‑call response carries the facts it dialled with, `matched_branch`, `outcome`, anything `collected` (e.g. the absence reason), the `detected_language` the operator actually spoke, plus the verbatim transcript with per‑turn sentiment and an `escalation_recommended` flag.

### Outcome codes

| Scenario | Outcomes |
|---|---|
| `shift_confirm` | `confirmed` · `absent_with_reason` · `disputed_flagged_to_planner` · `confirmed_self_reported_verify` · `escalated_supervisor_notified` · `data_mismatch_flagged` · `no_response` |
| `punch_in_remind` | `acknowledged_pending_verification` · `operator_claims_done_recheck_flagged` · `technical_issue_escalated` · `escalated_supervisor_notified` · `data_mismatch_flagged` · `no_response_reminder_logged` |
| `absence_priority` | `confirmed` · `declined_replacement_initiated` · `tentative_replacement_on_standby` · `leave_already_filed_verify` · `escalated_supervisor_notified` · `data_mismatch_flagged` · `no_response_treated_as_decline` |
| `ot_request` | `ot_accepted` · `ot_declined` · `awaiting_response_flag_ot_hours_check` · `escalated_supervisor_notified` · `data_mismatch_flagged` · `no_response_treated_as_decline` |
| `punch_out_remind` | `ot_extension_logged` · `operator_claims_done_recheck_flagged` · `technical_issue_escalated` · `escalated_supervisor_notified` · `data_mismatch_flagged` · `no_response_reminder_logged` |

---

## 6. Editing the scripts (the "knowledge base")

Everything the agent says lives in `scenarios/*.json` — one file per case, all three languages inside. Each branch has:

- `description` + `keywords` → what the AI matches on (keywords are **hints**, not required matches)
- `reply` → the exact words spoken. **The model never writes this**, so a script change can't be paraphrased away by the AI.
- `outcome` → the code written to the DB / sent onward to Odoo
- `end_call` → whether the call closes or the agent re‑asks
- `collect` + `after_collect_reply` → for the two‑step reason capture

Each scenario also has a `knowledge_base` array — the guardrails (no wage discussion, no leave approval, never reveal another employee's data, never mention consequences, never invent shift facts). These are given to the classifier as context.

After editing a JSON:

**POST** `{{base}}/scenarios/reload` → re‑reads from disk, no server restart. Returns `400` with the specific problem if a file is malformed, so you can't half‑load a broken script.

---

## 7. Verification status — ✅ working end-to-end on a real phone call

**Confirmed on a live Hindi call** to `+919426005911` (call SID `CA1b3840…`, 32s, 5 turns):

```
setup: shift_confirm/hi attempt=1
  operator: "नहीं!"                        -> branch=no   → agent asks for the reason
  operator: "उसे दिन मेरे घर पर शादी है।"    -> branch=no   → outcome=absent_with_reason
```

Stored outcome:

```json
{
  "outcome": "absent_with_reason",
  "matched_branch": "no",
  "collected": { "absence_reason": "उसे दिन मेरे घर पर शादी है।" }
}
```

The full chain works: Twilio dials → fetches the Hindi TwiML → opens the ConversationRelay WebSocket → Google STT transcribes Devanagari → Claude classifies the branch → the exact scripted Hindi reply is spoken back → the reason is captured verbatim → the outcome is persisted.

**Also verified against the live Claude API** via `/scenarios/test-turn` and `/scenarios/test-batch`: all 5 scenarios, all 3 languages, every branch, two-step reason capture, silence handling, and fact substitution. Classification handles code-mixed speech and negation traps — `"haan nahi aa paunga yaar, tabiyat kharab hai"` (opens with *haan*/yes but means no) is correctly classified `no`, with the reason auto-extracted so the redundant "tell us why" question is skipped.

### Two things about real calls worth knowing

**1. "Press any key to continue" is Twilio's trial-account message.** It's prepended to every outbound call on a trial account, before our agent speaks. Upgrading the Twilio account removes it. It is not coming from this code.

**2. The tunnel URL is the fragile part — and the #1 cause of a failed call.** Free tunnel URLs change on every restart and expire on their own. If `NGROK_URL` in `.env` is stale, Twilio dials, fails to fetch the TwiML, and the operator hears *"we're sorry, an application error occurred"* before being hung up on. Twilio logs this as error **11200**.

`POST /scenarios/call` now **pre-checks that the URL is reachable** and refuses with a `502` explaining the problem rather than placing a doomed call. If you see that 502: restart the tunnel, update `.env`, and **restart the server** (`.env` is only read at boot).

> ⚠️ On Windows, `pkill` from Git Bash does **not** reliably kill the server — a stale process keeps holding port 8000 and your restart silently fails to bind, so the old (stale) `NGROK_URL` stays live. Use the PowerShell command in §8 of [START_HERE.md](START_HERE.md) and confirm the port is free before restarting. This exact trap caused one failed call during testing.

### Voice IDs, if you ever need to change them

The scenario JSONs use `hi-IN-Neural2-B`, `gu-IN-Wavenet-B`, `en-IN-Neural2-B` with `ttsProvider="Google"`. Gujarati has **no** Neural2 voice in Google TTS — Wavenet is the best available tier. To change any of them, edit `voice` / `tts_language` / `transcription_language` in the scenario JSON and `POST /scenarios/reload` — no code change, no restart.

`debug="true"` is set on the `<ConversationRelay>` element, so Twilio reports handshake detail in the Console call log. The server also logs every setup, utterance, matched branch and outcome to stdout.

---

## Quick reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/scenarios` | List all five scenarios |
| GET | `/scenarios/{id}` | One scenario's full knowledge base |
| POST | `/scenarios/reload` | Re‑read `scenarios/*.json` after editing |
| POST | `/scenarios/preview` | Render spoken lines, no call placed |
| POST | `/scenarios/test-turn` | One conversational turn over HTTP |
| POST | `/scenarios/test-batch` | Many utterances, one request |
| POST | `/scenarios/call` | Place a real phone call |
| GET/POST | `/scenarios/voice` | Twilio webhook (returns TwiML) — not called by you |
| WS | `/scenarios/media-stream` | ConversationRelay socket — not called by you |
| GET | `/scenario-calls` | Outcomes of placed calls |
| GET | `/scenario-calls/{call_sid}` | One call: outcome + transcript + sentiment |
