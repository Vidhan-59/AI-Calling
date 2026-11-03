# AI Agent Studio

Create voice agents, configure how they sound, and train them on your own
scripts, Q&A, example conversations and documents. Open `/studio` on the
server (for example `http://localhost:8000/studio`), enter the admin API key
(`ADMIN_API_KEY`, blank if unset), and use **+ Create New Agent**.

What exists today (phases P0-P2 of the Studio plan):

| Area | Where |
|---|---|
| Dashboard: totals, active/published/draft, calls, recent activity, agent cards with Edit / Duplicate / Delete | `static/agent_studio.html` |
| Agent workspace: Basic information, Voice studio, Personality and instructions, Script and knowledge training, progress, live voice preview | same |
| API | `routers/studio.py` (`/studio/...`) |
| Storage (agents, training, calls, messages, traces) | `services/studio/store.py`, tables `studio_*` in `data/legacy_app.db` |
| Prompt assembly (what the model actually receives) | `services/studio/prompt.py`, `GET /studio/agents/{id}/prompt` |
| Retrieval over Q&A, scripts, scenarios, documents (BM25, no extra service) | `services/studio/knowledge.py` |
| Audio-continuity diagnostics | `GET /studio/diagnostics/audio`, `/studio#/diagnostics` |

Also built: **Call now** (section 5 of the agent workspace) - see below.

Not built yet: the browser testing playground, the full real-time call screen,
the AI execution trace panel and call history. Phone calls already write the
transcript (`studio_messages`) and trace events (`studio_traces`) those screens
will read.

## Call now

Section 5 of the agent workspace phones ONE number with the saved agent and shows
the status and live transcript. Set the number in `.env`:

```
STUDIO_CALL_NUMBER=+919876543210
```

It is read on every call (no restart). It also needs the Twilio settings
(`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`) and a reachable
`NGROK_URL`; on a Twilio trial account the number must be a verified caller ID.
The page shows exactly which of these is missing and disables the button until
they are all set. The number is only ever shown masked (`+91•••••••210`).

How the call runs: Twilio calls the number, fetches `/studio/voice`, and connects a
media stream to `/studio/media` (`services/voice_pipeline/studio_pipeline.py`).
Per turn the caller's words are transcribed, the agent's Q&A / scripts / demo
conversations / documents are searched, and the saved instructions plus the
matches go to the LLM; the reply is spoken in the agent's OmniVoice voice and
speed. A question that is essentially one of the agent's Q&A questions is answered
with the stored answer word for word (names, numbers and times can't be altered by
the model); anything else goes through the LLM. The agent's opening greeting is
the "Opening greeting" field.

Settings: `STUDIO_REPLY_MAX_TOKENS` (220) and `STUDIO_MAX_CALL_SECS` (600).

Test without calling anyone:
`.env\Scripts\python.exe tools\studio_call_smoke_test.py`
(a fake Twilio socket against the real STT, retrieval, LLM and OmniVoice).

## What the voice controls really do

The Voice studio shows each control's real status, taken from what OmniVoice
exposes (`services/studio/voice_caps.py`):

* **Supported per call:** voice, language (hi/gu/en), speaking speed 0.5x-2.0x.
* **Fixed when a voice is created:** gender, age, pitch (5 levels), English accent.
  "Create a new voice" builds one through `/voices/design`; it is then cloned so it
  sounds the same on every call.
* **Wording only (via instructions):** speaking style and emotional tone change how
  the agent phrases replies, not the synthesized audio.
* **Not supported by OmniVoice:** stability, clarity, expressiveness, regional
  pronunciation. They are listed as unsupported rather than shown as sliders.

## Training is retrieval, not fine-tuning

For each caller question the agent is given the best-matching Q&A pairs, script
blocks, demo conversations and document passages (BM25 ranking over Hindi,
Gujarati and English words) inside its prompt. The model weights are never
changed. **Knowledge base -> Test what the agent would know** shows exactly what
would be retrieved for a question. Documents: PDF, DOCX, CSV, TXT, MD up to 8 MB;
web pages must be public (private and loopback addresses are refused).

## API (all need `X-API-Key` when `ADMIN_API_KEY` is set)

```
GET  /studio/meta                      enums, templates, voice capabilities, use cases
GET  /studio/stats
GET/POST /studio/agents                GET/PUT/DELETE /studio/agents/{id}
POST /studio/agents/{id}/duplicate     POST /studio/agents/{id}/status   (draft | published | active)
GET  /studio/agents/{id}/prompt?query=
GET/POST /studio/agents/{id}/training  PUT/DELETE /studio/training/{item}
POST /studio/agents/{id}/training/qa/import   (CSV / JSON, append | replace)
GET  /studio/agents/{id}/training/export?format=json|csv
POST /studio/agents/{id}/knowledge/document | /knowledge/url | /knowledge/search
GET  /studio/diagnostics/audio
GET  /studio/call-config        POST /studio/agents/{id}/call
GET  /studio/calls/{id}         POST /studio/calls/{id}/end
```

Voices, voice design and the audio preview reuse the Voice Studio endpoints
(`/voices`, `/voices/design`, `/voices/preview`); nothing is duplicated.

Publishing needs system instructions (or a role) and a conversation objective;
"active" also needs an attached use case.

## Environment

No new variables. Uses `ADMIN_API_KEY`, `TTS_SERVICE_URL` / `TTS_SERVICE_TOKEN`
(OmniVoice) and the existing `.env`. `pypdf` is required for PDF upload (already in
`requirements.txt`).

## Tests

```
.\venv\Scripts\python.exe -m pytest tests/integration/test_studio_api.py tests/unit/test_playout.py
```
