# AI Calling Agent

A real-time voice AI system that connects **Twilio Voice** to **Claude (Anthropic)** to hold natural, two-way phone conversations. Callers talk, Claude listens (via live sentiment analysis) and replies out loud, and every turn is stored so you can review transcripts, sentiment, and escalation flags afterward.

> ### 🇮🇳 ShiftEaze scripted scenario calling — see **[POSTMAN_GUIDE.md](POSTMAN_GUIDE.md)**
>
> On top of the generic agent described below, this repo implements the **five ShiftEaze
> calling scenarios** (Shift Confirmation, Punch-In, Absence Prediction, Overtime Request,
> Punch-Out) in **English, Hindi and Gujarati**, using the exact scripts from the
> character-profile and question-script documents.
>
> The scripts live as editable knowledge bases in [scenarios/](scenarios/) — one JSON per case,
> all three languages inside. Claude classifies what the operator said onto one scripted
> branch (handling code-mixed Hinglish/Gujlish, paraphrase and negation), then speaks the
> **exact scripted reply** for that branch. The model chooses the branch; it never writes the
> words — which is what makes it safe for HR-sensitive calls while still sounding like a
> conversation rather than a keypad IVR.
>
> **Start here:**
> - **[START_HERE.md](START_HERE.md)** — how to start the server (`.\start.ps1`), every time
> - **[postman/ShiftEaze_AI_Calling.postman_collection.json](postman/)** — 35 ready-made requests, import straight into Postman
> - **[POSTMAN_GUIDE.md](POSTMAN_GUIDE.md)** — every endpoint with full request bodies and outcome codes
>
> Quickest path: `.\start.ps1`, then `POST /scenarios/preview` to proof-read the lines, then
> `POST /scenarios/test-turn` to hold a full conversation over HTTP with no phone, Twilio or
> tunnel involved.

> **Heads-up if you're new to this repo:** there are currently **two things living in this folder**:
> 1. **This simple agent** (`main.py`, `routers/`, `services/agent_manager.py`, `services/claude_client.py`, `services/conversation_store.py`, `simple_db.py`, `schemas.py`) — a working, runnable Claude + Twilio voice agent. This README is about this one.
> 2. **A much larger in-progress rebuild** (`configs/`, `services/flow/`, `services/pipeline/`, `services/telephony/`, `services/stt/`, `services/tts/`, `services/agents/`, `services/dispatch/`, `services/guardrails/`, `migrations/`, `evals/`, `workers/`, `config.py`, `db.py`, `models.py`) targeting a much bigger platform (Postgres, Redis, Pipecat, multi-agent routing, Odoo integration, Indian telephony providers). That work is **not wired up or runnable yet** — most of its providers aren't configured, and it depends on infrastructure (Postgres, Redis) that isn't running locally.
>
> The two share `config.py` (it deliberately keeps backwards-compatible aliases like `ANTHROPIC_API_KEY` for this simple agent) and `schemas.py` (holds this agent's Pydantic request/response models), but otherwise don't interfere with each other. If you only care about "make a phone call talk to Claude," you're in the right place — everything below works today with just an Anthropic key and Twilio credentials.

---

## Why Claude instead of OpenAI, and how this actually works

The original version of this repo used OpenAI's **Realtime API**, which does audio-in/audio-out directly over a WebSocket — the whole point of that API is that OpenAI handles speech recognition and speech synthesis internally.

**Claude has no equivalent realtime speech API.** Claude only understands text. So instead of streaming raw audio to Claude, this version uses **Twilio ConversationRelay**, which does the speech-to-text and text-to-speech *for you* and talks to your server in plain text over a WebSocket. That means:

```
Caller speaks
     │
     ▼
Twilio ConversationRelay  ──speech-to-text──►  transcribed text
     │                                              │
     │                                              ▼
     │                                   Your FastAPI server:
     │                                     1. Sentiment-classify the utterance (Claude Haiku 4.5)
     │                                     2. Build conversation history + agent's system prompt
     │                                     3. Ask Claude for a reply (streamed)
     │                                              │
     ◄──────────────text-to-speech────────────────  │
     ▼
Caller hears the reply
```

Everything your server ever sees or produces is **plain text** — no audio codecs, no buffering, no realtime audio plumbing to write yourself. This is also simpler to run locally and cheaper to operate than the audio-native approach.

The two-way, sentiment-aware behavior you asked for works like this:
- Every caller utterance is transcribed, **sentiment-classified** (positive/neutral/negative, an emotion word, and an `escalate` flag), and stored.
- If the caller sounds negative or the escalate flag is set, a short note (e.g. `[Caller tone: frustrated, sentiment=negative]`) is prepended to what Claude sees for that turn — so Claude's *next reply* is written with that context in mind (more empathetic, offers to escalate, etc.).
- The full conversation history (not just the latest message) is sent to Claude on every turn, so it remembers everything said earlier in the call.

---

## Architecture

```
┌─────────────┐   ConversationRelay   ┌──────────────┐    HTTPS     ┌─────────────┐
│   Twilio    │ ◄──── WebSocket ─────► │  FastAPI     │ ◄──────────► │   Claude    │
│   Voice     │      (plain text)      │  Server      │              │  (Anthropic)│
└─────────────┘                        └──────┬───────┘              └─────────────┘
                                               │
                                        SQLite (simple_db.py)
                                     agents / calls / messages
```

- **Dynamic, multi-agent, prompt-based "training"** — there's no fine-tuning involved (Claude's API doesn't support that). Instead, each **agent** is a row in the database: a name, a system prompt, a welcome greeting, a voice, a Claude model, and a "knowledge base" (a list of facts/FAQs that get woven into the prompt). Editing an agent's prompt via the API changes its behavior on the very next call — that's the "training from the prompt itself" mechanism. You can create as many agents as you want (e.g. one for sales, one for support, one for appointment reminders) and route different phone numbers or calls to different agents.
- **Storage** is a small SQLite file at `data/legacy_app.db`, managed by `simple_db.py` — no Postgres/Redis/Docker required to run this.

---

## Prerequisites

- Python 3.10+ (this repo was set up and tested with Python 3.14 on Windows)
- An [Anthropic API key](https://console.anthropic.com/settings/keys)
- A [Twilio](https://www.twilio.com/try-twilio) account (Account SID, Auth Token, a phone number) — only needed for real phone calls, **not** for local testing (see Step 5 below)
- [ngrok](https://ngrok.com/download) (or similar) — only needed for real phone calls, so Twilio can reach your local server

---

## Step-by-step setup

All commands below are PowerShell, run from `ai-calling-agent/`.

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` appear at the start of your prompt. If PowerShell blocks the activation script with an execution-policy error, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, the `anthropic` SDK, Twilio's SDK, and `pydantic-settings` (needed because `config.py` is now shared with the larger rebuild described above).

### 3. Configure your `.env` file

```powershell
Copy-Item .env.example .env
```

Then open `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...           # required - get one at console.anthropic.com
TWILIO_ACCOUNT_SID=AC...               # only needed for real phone calls
TWILIO_AUTH_TOKEN=...                  # only needed for real phone calls
TWILIO_PHONE_NUMBER=+1...              # only needed for real phone calls
NGROK_URL=https://xxxx.ngrok-free.app  # only needed for real phone calls
```

`DEFAULT_CLAUDE_MODEL` defaults to `claude-opus-5` and can be left as-is, or changed to `claude-sonnet-5` / `claude-haiku-4-5` for a faster/cheaper conversational model (sentiment analysis always uses Haiku 4.5 regardless, since that's a simple classification task).

### 4. Start the server

```powershell
uvicorn main:app --port 8000
```

You should see `Uvicorn running on http://127.0.0.1:8000`. On first start, a SQLite file is created at `data/legacy_app.db` and seeded with one `default` agent (its prompt comes from `prompts/system_prompt.txt`).

Verify it's alive:

```powershell
curl http://127.0.0.1:8000/
# {"status":"healthy","message":"AI Voice Assistant is running!"}
```

### 5. Test it *without* a phone call, Twilio, or ngrok

This is the fastest way to check your agent's prompt, knowledge base, and sentiment-aware behavior — talk to it over plain HTTP:

```powershell
curl -X POST http://127.0.0.1:8000/chat/test `
  -H "Content-Type: application/json" `
  -d '{\"agent_id\": \"default\", \"session_id\": \"test1\", \"message\": \"Hi there!\"}'
```

`session_id` is any string you pick — reuse it across requests to continue the same conversation. Try sending a frustrated-sounding message and watch the `sentiment` block in the response, and notice the reply gets noticeably more empathetic.

### 6. Wire up a real phone call

1. Start ngrok in a separate terminal: `ngrok http 8000`, then copy the `https://...ngrok-free.app` URL into `NGROK_URL` in `.env` and restart uvicorn.
2. **Inbound calls:** in the [Twilio Console](https://console.twilio.com/), open your phone number's configuration and set "A call comes in" to a webhook pointing at `<your-ngrok-url>/outgoing-call` (HTTP POST).
3. **Outbound calls:** hit `/make-call`:

   ```powershell
   curl -X POST http://127.0.0.1:8000/make-call `
     -H "Content-Type: application/json" `
     -d '{\"to_phone_number\": \"+1234567890\", \"agent_id\": \"default\"}'
   ```

Either way, Twilio connects the call to `/media-stream` via ConversationRelay, and the conversation loop described above takes over.

> **Note:** ConversationRelay is a Twilio **public beta** product. If message field names change on Twilio's end, `routers/voice.py` is the one file to check/update — it's a single, well-commented WebSocket handler.

---

## Creating and editing agents (the "training" workflow)

Every agent is just a database row you manage over the API — no code changes, no redeploys, no fine-tuning.

**Create a new agent:**

```powershell
curl -X POST http://127.0.0.1:8000/agents `
  -H "Content-Type: application/json" `
  -d '{
    \"name\": \"Sales Bot\",
    \"system_prompt\": \"You are a friendly sales rep for Acme Corp. Keep replies short and warm.\",
    \"welcome_greeting\": \"Hi! Thanks for calling Acme, how can I help?\",
    \"knowledge_base\": [
      \"Acme ships in 3-5 business days\",
      \"Returns are accepted within 30 days of purchase\"
    ]
  }'
```

This returns an `id` (auto-generated from the name, e.g. `sales-bot`) — pass that as `agent_id` on `/make-call`, in the ConversationRelay webhook URL (`/outgoing-call?agent_id=sales-bot`), or on `/chat/test` to talk to that specific agent.

**Edit an agent live** (takes effect on the very next call/message — this is the whole point):

```powershell
curl -X PUT http://127.0.0.1:8000/agents/sales-bot `
  -H "Content-Type: application/json" `
  -d '{\"system_prompt\": \"You are now also allowed to discuss bulk order discounts.\"}'
```

**List / inspect / delete:**

```powershell
curl http://127.0.0.1:8000/agents
curl http://127.0.0.1:8000/agents/sales-bot
curl -X DELETE http://127.0.0.1:8000/agents/sales-bot
```

`knowledge_base` is a list of short facts/FAQs — think of it as lightweight, editable "training data." It gets woven into the agent's system prompt on every turn, so Claude can reference those facts naturally without you having to rewrite the whole prompt every time a fact changes.

---

## Reviewing calls, transcripts, and sentiment

```powershell
curl http://127.0.0.1:8000/calls                    # list recent calls (real + /chat/test sessions)
curl http://127.0.0.1:8000/calls/<call_sid>          # full transcript, per-message sentiment, escalation flag
```

Each message in a transcript carries `sentiment_label` (positive/neutral/negative), `sentiment_score` (-1.0 to 1.0), `emotion` (one word), and `escalate` (whether a human should probably step in). The call summary also reports `average_sentiment_score` and `escalation_recommended` for the whole conversation. Escalation flags are currently just logged to the server console (`routers/voice.py`) — wiring that up to a Slack alert, webhook, or live transfer is a natural next step if you need it.

---

## API reference

| Method    | Endpoint             | Description                                                  |
| --------- | --------------------- | ------------------------------------------------------------ |
| GET       | `/`                    | Health check                                                  |
| GET       | `/agents`              | List all agents                                                |
| POST      | `/agents`              | Create a new agent                                             |
| GET       | `/agents/{id}`         | Get one agent                                                  |
| PUT       | `/agents/{id}`         | Update an agent's prompt/voice/model/knowledge base            |
| DELETE    | `/agents/{id}`         | Delete an agent (not the `default` one)                        |
| POST      | `/make-call`           | Place an outbound call (`to_phone_number`, `agent_id`)          |
| POST/GET  | `/outgoing-call`       | Twilio webhook - returns ConversationRelay TwiML               |
| POST      | `/recording-status`    | Twilio recording status webhook                                |
| WebSocket | `/media-stream`        | ConversationRelay's text-in/text-out socket (the call loop)     |
| GET       | `/calls`               | List recent calls                                              |
| GET       | `/calls/{call_sid}`    | Full transcript + sentiment summary for one call                |
| POST      | `/chat/test`           | Talk to an agent over plain HTTP - no phone/Twilio needed        |

---

## Project layout (this agent's files only)

```
main.py                       FastAPI app, wires up the routers
config.py                     Shared settings (owned by the larger rebuild; keeps
                               backwards-compatible aliases this agent relies on)
schemas.py                    Pydantic request/response models for this agent's endpoints
simple_db.py                  This agent's own isolated SQLite store (data/legacy_app.db)
prompts/system_prompt.txt     Seed prompt for the built-in "default" agent (first run only)

services/
  claude_client.py            Anthropic API wrapper: streaming replies + sentiment classification
  agent_manager.py             Agent CRUD + system-prompt assembly (base prompt + knowledge base)
  conversation_store.py        Calls/messages persistence + summary/sentiment aggregation

routers/
  agents.py                    Agent CRUD endpoints
  calls.py                     /make-call, /outgoing-call TwiML, /calls, /chat/test
  voice.py                     The ConversationRelay WebSocket - the live call loop
```

---

## Troubleshooting

- **"Missing ANTHROPIC_API_KEY" error when calling `/chat/test` or making a call:** the server starts fine without a key (health check and agent CRUD both work), but anything that actually talks to Claude needs it set in `.env` — restart uvicorn after editing `.env`.
- **`ModuleNotFoundError` on startup:** make sure your venv is activated (`(venv)` in the prompt) and you ran `pip install -r requirements.txt` inside it.
- **Twilio can't reach your server:** `NGROK_URL` in `.env` must match your *current* ngrok URL exactly (it changes every time you restart ngrok on the free tier) — update `.env` and restart uvicorn after every ngrok restart.
- **A call connects but there's silence/no reply:** check the server console — sentiment classification and Claude replies are logged there, and any error (e.g. a bad model name) will show up as a Python traceback.

---

## Development

```powershell
pip install -r requirements-dev.txt
black .
isort .
flake8
mypy main.py
pytest
```

## License

MIT - see [LICENSE](LICENSE).

## ⚠️ Disclaimer

This project is not officially affiliated with Anthropic or Twilio. Use responsibly and in accordance with their terms of service.
