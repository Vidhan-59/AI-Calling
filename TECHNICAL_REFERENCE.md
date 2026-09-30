# Technical Reference — AI-Calling

A deep, file-and-line-level reference to everything in this repository: what
each module does, how the three call-handling systems actually work, exact
message/data contracts, and every known gap or bug. This is not a narrative
audit (see `AUDIT.md` for that, written 2026-08-20) and not a design
rationale doc (see `PHASE2_ARCHITECTURE_PROPOSAL.md` for that) — it's the
thing to open when you need to know exactly how something works or why a
line of code is the way it is.

**Repo state this document describes:** branch `main`, latest commit
`6590ecc`, **plus a second, uncommitted pass** sitting in the working tree
(sentiment on the rebuild call path, per-turn latency work, the
`/agent-voice/preview` dry-run endpoint, `routers/analytics.py`,
`services/auth.py`, and dialer-state reconciliation on startup). Every
section below describes the code as it exists on disk right now, and calls
out explicitly wherever a described piece is uncommitted.

---

## 0. The three systems in one repo

This repository contains **three independent call-handling systems** that
happen to share one FastAPI process, one Anthropic account, and (partially)
one database. They were built at different times, for different purposes,
and do not call into each other.

| | Track A | Track B | The Rebuild |
|---|---|---|---|
| **What it is** | Free-form generic agent, CRUD-defined | ShiftEaze's 5 scripted HR scenarios | Config-driven multi-agent flow-graph engine |
| **Reply source** | Claude writes the words, live | AI picks one pre-written line per branch | Claude writes the words, but confined to one node's objective |
| **Entry points** | `routers/agents.py`, `routers/calls.py`, `routers/voice.py` | `routers/scenarios.py`, `routers/voice_scenario.py` | `routers/campaigns.py`, `routers/agent_voice.py`, `routers/analytics.py` |
| **Storage** | `data/legacy_app.db` (raw `sqlite3`, `simple_db.py`) | same file, different tables | `data/app.db` (async SQLAlchemy, `models.py`) |
| **Telephony** | Twilio direct (`twilio.rest.Client`) | Twilio direct | `services/telephony/*` ABC (mock/twilio/exotel/plivo), only reachable via the dialer |
| **Status** | Live, proven on a real call (per `README.md`) | Live, proven on a real call (per `POSTMAN_GUIDE.md`) | Live-wired as of the Phase 2/3 work; **not yet smoke-tested against a real Twilio call** (no Twilio creds configured on this dev machine) |
| **"Training" mechanism** | Edit `system_prompt` + `knowledge_base` via CRUD | Edit `scenarios/*.json` | Edit `configs/agents/*.yaml` + `configs/flows/*.yaml` + `prompts/agents/<id>/persona.j2` — zero Python |

**Why three?** Per `simple_db.py`'s own docstring: Track A predates the
rebuild; Track B was built for a specific production need (ShiftEaze HR
calling at scale) using the same low-level pattern as Track A; the rebuild
is a from-scratch, config-driven generalization of Track B's ideas (any
number of agents, not five hardcoded ones) that isn't finished replacing
either older track yet. Nothing here is dead code by accident — Track A is
explicitly kept as "the `/chat/test` demo path," Track B is explicitly the
thing in real HR-calling use, and the rebuild is explicitly where new work
is happening.

---

## 1. Repository layout

```
main.py                     FastAPI app entrypoint, all startup wiring
config.py                   Central pydantic Settings + legacy module-level aliases
db.py                       Async SQLAlchemy engine/session + Redis client (rebuild)
models.py                   SQLAlchemy ORM models (rebuild data model)
schemas.py                  Pydantic request/response models (Track A + rebuild)
scenario_schemas.py         Pydantic models specific to Track B
simple_db.py                Raw-sqlite3 store for Track A + Track B (legacy_app.db)
public_url.py                Resolves the live ngrok URL fresh per request
preflight.py                 Stdlib-only pre-boot checks (run by start.ps1)
alembic.ini, migrations/     DB migrations for the rebuild's Postgres/SQLite schema
docker-compose.yml           Local Postgres 16 + Redis 7 for the rebuild
start.ps1 / run-all.ps1 / setup-new-machine.ps1   Windows operational scripts

routers/
  agents.py, calls.py, voice.py        Track A
  scenarios.py, voice_scenario.py      Track B
  campaigns.py, agent_voice.py, analytics.py   Rebuild

services/
  claude_client.py            All Anthropic API calls, every track
  agent_manager.py             Track A: agent CRUD + system-prompt assembly
  conversation_store.py        Track A/B shared: calls/messages read/write
  call_logger.py               Track A/B shared: per-call JSON transcript file
  scenario_engine.py            Track B: ScenarioSession state machine
  scenario_store.py             Track B: scenario_calls outcome storage
  agent_registry.py             Rebuild: YAML config loading + DB sync
  verbalize.py                  Hindi/Gujarati number/time-to-speech (unwired)
  auth.py                       Admin X-API-Key dependency (uncommitted)
  agents/conversation_agent.py  Rebuild: the per-turn decision loop
  flow/graph.py, resolver.py, slots.py     Rebuild: flow-graph engine
  guardrails/input.py, output.py, rules.py  Rebuild: 3-layer guardrail system
  tools/registry.py, hr_tools.py, roster_tools.py, transfer_tools.py, comms_tools.py
  dispatch/admission.py, dialer.py       Rebuild: compliance gate + call queue
  pipeline/chunker.py            Sentence-chunking for TTS (uncommitted)
  telephony/, stt/, tts/          Provider ABCs — built, mostly unwired
  integrations/odoo_client.py      HMAC-signed webhook delivery to Odoo

configs/agents/*.yaml, configs/flows/*.yaml   The rebuild's 5 agent definitions
prompts/*.j2, prompts/agents/<id>/persona.j2   Jinja2 prompt templates
scenarios/*.json                                Track B's 5 scripted scenarios

local_ai/                    Fully offline STT(faster-whisper)→LLM(Ollama)→TTS(Piper)
tests/unit/, tests/integration/, tests/local_ai/
data/                         Runtime SQLite files land here (empty in git)
workers/, evals/               Placeholder packages, no implementation yet
```

---

## 2. Track A — the legacy free-form agent

### 2.1 Agent CRUD and the "training" mechanism

`routers/agents.py` exposes plain CRUD at `/agents` (`GET/POST/PUT/DELETE`),
backed by `services/agent_manager.py` against `simple_db.py`'s `agents`
table. There is no ML training anywhere in this track — "training" an agent
means editing its `system_prompt` and `knowledge_base` list.

`agent_manager.build_system_prompt(agent)` is the entire mechanism: it
appends the agent's `knowledge_base` (a JSON-encoded list of strings) as a
bullet list under the instruction *"Use the following reference information
when relevant... Do not mention that you were given this list; just use it
naturally."* That assembled string becomes the Claude `system` parameter on
every turn.

`create_agent()` auto-slugifies `name` into an id if none is given, and
appends `-{6 hex chars}` on collision. `update_agent()` only touches fields
actually present in the request body (`exclude_unset`) and always bumps
`updated_at`.

**Prompt-injection guard on agent authoring itself**: `schemas.py`'s
`AgentCreateRequest`/`AgentUpdateRequest` run every `knowledge_base` entry
through `services/guardrails/input.contains_prompt_injection` and reject
(422) any that match — because knowledge base entries are woven verbatim
into every call's system prompt, so this is the CRM-free-text-becomes-
prompt-injection risk, caught at the API boundary rather than left unguarded.

### 2.2 SQLite schema (`simple_db.py`)

One file, `data/legacy_app.db`, single shared `sqlite3` connection guarded
by a `threading.Lock()`. Deliberately isolated from the rebuild's `db.py`/
`models.py`/`app.db` — the module's own docstring says this whole file can
be deleted once the rebuild replaces `/agents` and `/chat/test`.

```sql
CREATE TABLE agents (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, system_prompt TEXT NOT NULL,
  welcome_greeting TEXT NOT NULL, voice TEXT NOT NULL, model TEXT NOT NULL,
  effort TEXT NOT NULL DEFAULT 'medium',
  knowledge_base TEXT NOT NULL DEFAULT '[]',   -- JSON list of strings
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE calls (
  call_sid TEXT PRIMARY KEY, agent_id TEXT NOT NULL,
  from_number TEXT, to_number TEXT, direction TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'in_progress',
  started_at TEXT NOT NULL, ended_at TEXT
);

CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, call_sid TEXT NOT NULL,
  role TEXT NOT NULL, content TEXT NOT NULL,
  sentiment_label TEXT, sentiment_score REAL, emotion TEXT,
  escalate INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
-- plus scenario_calls (Track B, same file — see §3.6)
```

All timestamps are UTC ISO-8601 strings (`now_iso()`), not native SQLite
datetimes. `init_db()` seeds a `"default"` agent from `prompts/system_prompt.txt`
if the `agents` table is empty on first boot.

### 2.3 Call flow: `POST /make-call` → TwiML → `/media-stream`

1. `POST /make-call` (`routers/calls.py`) validates the agent exists and
   Twilio creds + `NGROK_URL` are configured, then calls
   `twilio.rest.Client(...).calls.create(url=f"{NGROK_URL}/outgoing-call?agent_id=...", record=True, recording_status_callback=...)`.
   **Note**: this reads `NGROK_URL` from the module-level alias frozen at
   import time (`config.NGROK_URL`), not the fresh-per-request
   `public_url.get_public_url()` that Track B uses — so Track A does not
   benefit from Track B's "swap tunnel without restarting the server" fix.
2. `GET/POST /outgoing-call` (Twilio's webhook) looks up the agent
   (`agent_id` query param, falls back to `"default"`), returns TwiML:
   ```xml
   <Response><Connect>
     <ConversationRelay url="wss://{host}/media-stream"
                         welcomeGreeting="{agent.welcome_greeting}"
                         voice="{agent.voice}" ttsProvider="Google" interruptible="true">
       <Parameter name="agentId" value="{agent.id}"/>
     </ConversationRelay>
   </Connect></Response>
   ```
   No `ttsLanguage`/`transcriptionLanguage`/`debug` attributes — Track A is
   English-only/simpler than Track B's TwiML.
3. `/media-stream` WebSocket (`routers/voice.py`) runs the turn loop — see
   §2.4.

### 2.4 ConversationRelay message contract (exact)

Inbound (Twilio → us):
- `{"type":"setup","callSid":...,"customParameters":{...}}`
- `{"type":"prompt","voicePrompt":"<transcript>","last":bool}` — if `last`
  is falsy, discarded (interim transcript, not final).
- `{"type":"interrupt", "utteranceUntilInterrupt":...}` — logged only.
- `{"type":"dtmf","digit":...}` — logged only, no action taken.

Outbound (us → Twilio): `{"type":"text","token":"<chunk>","last":bool}`,
streamed per chunk, then one final empty-token `last:true` sentinel.
`{"type":"end"}` only on the `RuntimeError` (missing API key) error path.

`customParameters` is read defensively both ways
(`data.get("customParameters") or data.get("custom_parameters")`) because
ConversationRelay is a Twilio public-beta product whose exact field names
the code's own docstring warns may change.

On `setup`, the welcome greeting — already spoken by Twilio directly from
the TwiML attribute, never over this socket — is still recorded into
`history` and both stores, specifically so Claude's first reply has context
that a greeting already happened.

### 2.5 Concurrent sentiment + reply generation (uncommitted)

**This is a just-changed piece of code, not yet on `main`.** Old behavior
was sentiment-then-reply, sequential (one full Claude round-trip of pure
added latency per turn). Current behavior in the `prompt` handler:

1. `sentiment_task = asyncio.create_task(claude_client.classify_sentiment(user_text))` — fired, not awaited.
2. `conversation_store.add_message(call_sid, "user", user_text)` — inserted
   **immediately, with no sentiment yet**, so transcript ordering stays
   correct regardless of which async task finishes first.
3. `generate_reply_stream(...)` streams the reply to Twilio *while*
   `sentiment_task` runs independently on the event loop — this is the
   actual concurrency; the two Claude calls are never awaited against each
   other.
4. Only after the reply is fully sent: `sentiment = await sentiment_task`
   (already resolved almost every time, since it's a small/fast Haiku call
   that started before the streaming reply even finished).
5. Backfill: `conversation_store.set_message_sentiment(...)` and
   `call_logger.set_turn_sentiment(...)` update the already-inserted
   row/log entry in place, keyed by the row id / turn seq captured at
   insertion time.

**Trade-off accepted**: the old code injected a `"[Caller tone: {emotion},
sentiment={sentiment}] "` prefix onto the user's turn in Claude's own
context when sentiment was negative/escalate — a small quality benefit
(more empathetic replies to visibly upset callers). That's gone now,
because it's exactly what forced the two calls to run sequentially. `/chat/test`
(§2.6) still does this, since it has no latency constraint.

`/agent-voice/media-stream` (the rebuild's WebSocket, §5.11) applies the
identical concurrent-sentiment pattern.

### 2.6 `POST /chat/test`

Plain-HTTP agent testing, no Twilio/ngrok/phone required — explicitly for
validating a prompt + knowledge base + sentiment behavior right after
setting `ANTHROPIC_API_KEY`. Sequential (not concurrent): classify sentiment
→ persist → **mutate the in-memory history's last entry** to prepend the
tone tag if negative/escalate (the stored DB row is unaffected, only what
Claude sees this one call) → non-streaming `generate_reply()` → persist
reply → return `{call_sid, reply, sentiment}`.

### 2.7 `services/conversation_store.py` + `services/call_logger.py`

`conversation_store.py` wraps `simple_db`'s `calls`/`messages` tables.
`get_call_summary(call_sid)` (powers `GET /calls/{sid}`) joins the call row
with every message and computes `average_sentiment_score` and
`escalation_recommended` (`any(m.escalate)`).

`call_logger.py` is a **parallel, file-based** audit log, independent of
SQLite — one JSON file per call at `data/call_logs/<sanitized_sid>.json`,
written incrementally per turn under a lock so a mid-call crash still
preserves whatever was captured. `log_turn()` now returns the turn's `seq`
number (uncommitted change) specifically so a caller can log a turn
immediately and backfill its sentiment later via the new
`set_turn_sentiment(call_sid, seq, sentiment)` — the file-log equivalent of
`conversation_store.set_message_sentiment`.

### 2.8 Quirks worth knowing

- `routers/voice.py`'s only error handling is a bare `except RuntimeError`
  (missing API key) around the whole socket — any other exception is
  **unhandled** and crashes the WebSocket handler ungracefully. Contrast
  Track B, which catches broadly.
- Its `finally` block unconditionally marks the call `"completed"`,
  regardless of how the loop actually exited (including an unhandled
  exception) — unlike Track B, which distinguishes a clean end from a
  dropped connection.
- `POST /make-call`'s `CallRequest.contact_id` field is accepted but never
  used — no contact lookup happens on this path at all (Track A has no
  concept of `models.Contact`/consent/DNC; only the rebuild's dialer does).

---

## 3. Track B — the ShiftEaze scripted scenario system

### 3.1 Five scenarios × three languages

`scenarios/*.json`, loaded/cached by `scenario_engine.load_scenarios()`:
`shift_confirm`, `absence_priority`, `ot_request`, `punch_in_remind`,
`punch_out_remind`. Every file defines `languages: {en, hi, gu}` always all
three, and every language block requires `opening`, `voice`,
`tts_language`, `transcription_language`, `branches`, `silence_reply`
(validated at load time — missing any of these raises).

Example branch (`shift_confirm.json`, English):
```json
"no": {
  "description": "Operator says they will not attend / cannot come.",
  "keywords": ["no", "can't come", "won't attend", "not coming"],
  "reply": "Please tell us the reason for your absence.",
  "collect": "absence_reason",
  "after_collect_reply": "Your information has been recorded. Thank you!",
  "outcome": "absent_with_reason",
  "end_call": true
}
```
Each scenario also carries `required_facts` (must be supplied or preview/
call is rejected), `max_attempts`, `no_answer_outcome`, and a
`knowledge_base` array of plain-English compliance facts (e.g. *"Salary,
wages, OT rates and payment questions must be deferred to HR — never
answered on this call"*) that documents intent inline but is **not
currently read anywhere in `scenario_engine.py`** — only `branches`
descriptions/keywords feed the classifier prompt.

### 3.2 How "the AI picks a line, never writes free text" is enforced

Not a prompt instruction alone — a structural guarantee:

1. `_classification_schema()` builds a JSON schema whose `branch` field is
   `"enum": [*branch_keys, "unclear"]` — a **closed enumeration**, no free
   -text reply field exists in the schema at all.
2. `classify_utterance()` calls `claude_client.classify_structured()` with
   `output_config={"format": {"type":"json_schema", "schema": schema}}` —
   Anthropic's structured-output mode constrains generation to that schema,
   so an out-of-vocabulary branch is not representable, not just discouraged.
3. Once classified, `ScenarioSession.handle_utterance()` looks up
   `lang_cfg["branches"][classification.branch]["reply"]` — the literal
   pre-written string from the JSON, with only `{placeholders}` substituted.
   The model's own generated text is used *only* for an audit-log
   `rationale` field and an optional `volunteered_reason` capture — never
   as spoken output.

Module docstring states the safety property directly: *"the model chooses
the branch, it never writes the words... That combination is what makes
this safe for HR-sensitive calls at 1,300+ operators."*

Negation and context-sensitivity are handled at the prompt level (not the
schema): *"Watch negation carefully: 'haan nahi aa paunga' means NO, not
yes"*, and — critically for the reason-collection flow — *"If the operator
is now giving a reason [after being asked], that is NOT a new 'no' — classify
as 'unclear' so the collection flow captures it verbatim."* Conversation
history fed to the classifier is capped to the last 6 turns.

### 3.3 The Gujarati-via-Hindi-STT workaround

Google Cloud STT (what Twilio ConversationRelay uses) has no `gu-IN`
recognition model — setting `transcriptionLanguage="gu-IN"` silently falls
back to `en-US` and produces garbage transcripts. The fix
(`routers/scenarios.py`):

```python
_TRANSCRIPTION_LANGUAGE_OVERRIDES = {"gu-IN": "hi-IN"}
```

applied only at TwiML-build time (`scenario_twiml()`), substituting `hi-IN`
for the `transcriptionLanguage` attribute while `ttsLanguage`/`voice` stay
native Gujarati (`gu-IN`/`gu-IN-Wavenet-B`) — **synthesis is unaffected,
only recognition is rerouted**. The scenario JSON itself still says
`"gu-IN"`; the override lives only in the router. The classifier prompt
carries the matching instruction: *"Gujarati speech may be transcribed
through a Hindi speech model... 'હા' (yes) may appear as 'हा'. Classify on
meaning."* — that's the actual mechanism making a Hindi-garbled Gujarati
transcript still classify correctly. `POST /scenarios/preview` returns the
**raw**, un-overridden `transcription_language` — the override applies only
on the real call path.

### 3.4 `POST /scenarios/preview` and `POST /scenarios/test-batch`

**`/preview`**: renders the opening line, silence reply, and every branch's
reply/outcome with the given `facts` substituted — no call, no DB write,
pure dry-run over `scenario_engine.render()`.

**`/test-batch`**: given a list of utterances, classifies each — **note:
each utterance starts a brand-new `ScenarioSession`**, so this is N
independent single-turn classifications, not one continuous conversation
("the quickest way to verify all branches still classify correctly after
editing a scenario JSON"). Response per utterance: `matched_branch`,
`confidence`, `agent_reply`, `outcome`, `call_ended`.

`POST /scenarios/test-turn` is the continuous-conversation variant, keyed
by `f"{scenario_id}:{language}:{session_id}"` in an in-memory dict — module
comment: *"Live phone calls keep their session in the WebSocket handler's
closure instead"* — this in-memory dict is for HTTP testing only.

### 3.5 Tunnel-reachability check

`_check_tunnel_reachable(base_url)` does a synchronous
`urllib.request.urlopen(base_url + "/", timeout=8)` right before dialing.
Exists because of a real production failure mode: a dead ngrok tunnel makes
Twilio dial the operator, fail to fetch TwiML, and play "an application
error occurred" before hanging up — indistinguishable from a broken agent
to the caller, with only a Twilio 11200 alert as a trace. This turns that
into an immediate HTTP 502 with an actionable message before the call is
even placed. `base_url` comes from `public_url.get_public_url()`, which
re-parses `.env`'s `NGROK_URL=` line fresh on every call (not cached at
import time) so a tunnel swap doesn't require a server restart.

### 3.6 `scenario_calls` outcome storage

Same physical file as Track A (`simple_db.py`), separate table:
`call_sid, scenario_id, language, to_number, employee_name, rgid,
facts_json, attempt, status, outcome, matched_branch, collected_json,
detected_language, started_at, ended_at`. Written twice per real call: once
at `POST /scenarios/call` time with a synthetic `call_sid=f"pending:{ref}"`
(the real Twilio SID isn't known yet), and again in the WebSocket's `setup`
handler once it is — re-keyed onto the real SID. **The `pending:*` row is
never deleted**, so a stray row persists per completed call.

`GET /scenario-calls/{call_sid}` joins this row with the shared
`conversation_store` transcript and the file-based call log into one
response.

### 3.7 `routers/voice_scenario.py` — specifics beyond Track A's loop

- Every scripted reply is sent as **one** `{"type":"text","last":true}`
  message (not token-streamed — it's a static string, not an LLM stream).
- Local-LLM-backend interaction (explicit comment): *"Sentiment runs
  concurrently with classification on Claude — cloud capacity means the
  second call doesn't steal time from the first. That breaks on the local
  backend: Ollama on a fixed-core CPU can't truly parallelize two
  inferences, so 'concurrent' just doubles the real wait. Sentiment isn't
  load-bearing for the scenario flow, so skip it outright on local."*
  Implemented as `sentiment_task` only created `if utterance and not
  is_local_backend`.
- If `handle_utterance()` raises, falls back to the scripted `on_silence()`
  close rather than improvising an unscripted line on a compliance call.
- `finally` distinguishes a clean end from a dropped connection
  (`status="disconnected"`, `outcome` falls back through session/scenario
  defaults to a hardcoded `"call_dropped"`).

### 3.8 `ScenarioSession` state machine (`scenario_engine.py`)

Safety caps: `MIN_UTTERANCE_CHARS=2` (shorter = noise), `MAX_REPROMPTS_PER_ATTEMPT=2`,
`MAX_TOTAL_TURNS=10`. `render(template, facts)` deliberately isn't
`str.format` — a missing key must never raise mid-call; an unmatched
`{placeholder}` is left literally visible rather than silently dropped, so
a misconfigured scenario is obvious in a transcript instead of silently
losing information.

`handle_utterance()` decision order: already finished → no-op. Too-short
utterance → `on_silence()`. Turn cap exceeded → `on_silence()`. **Mid-flow
collection** (a prior "no" set `self.collecting`): the next utterance is
captured **verbatim, unclassified** — *"because my wife is in hospital"
must not be re-read as a new branch.* Otherwise classify; `"unclear"`
reprompts up to the cap then gives up. A branch with `collect` set either
asks the follow-up question (and waits) or, if the operator already
volunteered the reason in the same utterance, skips straight to
`after_collect_reply` without asking again.

### 3.9 Local LLM backend switch

`SCENARIO_LLM_BACKEND=local` (default `"claude"`) swaps
`classify_utterance()`'s `classify_fn` to `local_ai.llm_engine.classify_structured`
— "mirrors claude_client.classify_structured's exact signature so this is a
same-shape swap, not a forked code path." See §7 for the local pipeline
itself.

---

## 4. The Rebuild — config-driven multi-agent flow-graph system

### 4.1 Mental model

A deterministic **flow graph of objectives** (YAML), where the LLM never
sees the graph — only the current node's objective text, rendered into a
system prompt by a template pipeline. Transition conditions are a small
closed grammar (`==`,`!=`,`>=`,`<=`,`>`,`<` against a flattened
slots+counters context), not `eval()`, specifically because the YAML is
meant to be editable by someone who owns a calling scenario, not a
Python engineer.

### 4.2 `services/flow/graph.py` — the FlowGraph engine

`Transition(when: str, goto: str)`. `goto` is either a node id in the same
graph or the sentinel `"HANDOFF:<agent_id>"` — the graph itself does
nothing special with that string; `conversation_agent.py` is what inspects
the prefix. `matches(context)`: `"always"` short-circuits True; otherwise
scans for the first comparator substring in `when`, splits once, looks up
the left side in `context`, parses the right side as a literal
(`true`/`false`→bool, quoted→str, else `int()`, else raw string), compares.
**Edge case**: an unfilled slot (`context[key] is None`) compared with an
ordering operator (`>`,`<`,`>=`,`<=`) is coerced to `0` before comparing —
not treated as "missing," just as zero.

`FlowNode(id, objective, collect=[], tools=[], transitions=[],
on_max_reprompts=None)`. **`on_max_reprompts` is defined but never read
anywhere** — reprompt-limit handling in the shipped `shift_confirm.yaml`
flow is expressed as ordinary transitions guarded by `reprompts >= N`
instead. Looks like a field defined ahead of an unbuilt feature.

`FlowGraph.from_yaml()` validates only that `entry` and every declared
`terminal_nodes` id exist among the parsed nodes — transition `goto`
targets and `when` expressions are **not** validated at load time; a typo'd
`goto` only surfaces when that transition is actually evaluated at runtime.
`next_node_id(current_node_id, context)` is **first-match-wins** over that
node's transitions; returns `None` if none match, which the caller reads as
"stay here, reprompt."

### 4.3 Prompt assembly (`services/flow/resolver.py` + `prompts/*.j2`)

Jinja env: `autoescape=False` (output is spoken text, not HTML),
`trim_blocks=True`, `lstrip_blocks=True`. A missing template file is caught
and silently rendered as `""` — no error, just an empty section (a real
silent-failure risk if a persona file is mistyped).

`render_system_prompt()` assembles, in order, joined by blank lines,
skipping empty sections:

1. `prompts/base_persona.j2` — identifies the caller/organization/context,
   states it's a live phone call with background noise, frames "one
   objective at a time, not a general assistant."
2. `prompts/voice_output_rules.j2` — hard formatting constraints: 1–2
   sentences, ≤25 words (except reading roster details), exactly one
   question per turn, no markdown/bullets/emoji, numbers/times/dates spelled
   the way they're spoken ("do baje dopahar" not "14:00"), never spell shift
   codes as letters, unclear-transcript handling (ask once, then move on or
   offer SMS — never a third time).
3. `prompts/language_policy.j2` — reply in the resolved language name; a
   fixed set of terms (`shift, roster, punch-in, punch-out, overtime, OT,
   terminal, crane, gate`) stay in English regardless of language; a
   Gujarati-specific code-switching block; immediate permanent language
   switch if the caller asks.
4. `prompts/guardrails.j2` — "ABSOLUTE RULES — these override your
   objective." Most rules render conditionally on membership in the agent's
   `guardrails:` list; several are **unconditional regardless of that
   list**: roster facts must come only from `lookup_roster` tool output
   ("This is the most important rule on this list"), RGID read only as
   last-4-digits, never claim to be human, call `transfer_to_human` if
   asked for a person/distressed/off-objective.
5. `prompts/agents/<id>/persona.j2` — agent-specific framing (see §4.7 for
   `shift_confirm`'s).
6. The node block (`prompts/node_block.j2`): `CURRENT OBJECTIVE`, then
   conditionally `INFORMATION YOU STILL NEED` (missing slots),
   `WHAT YOU HAVE CONFIRMED SO FAR` (filled slots), available tools this
   turn, supervisor guidance if present, and a rephrasing instruction if
   this node has been reprompted before.

**Gap found**: `guardrails.j2` has conditional blocks for
`no_wage_discussion`, `no_leave_approval`, `no_third_party_employee_data`,
`no_disciplinary_threat`, `no_medical_advice` — but `shift_confirm.yaml`
also lists `disclose_ai_and_recording` and `hallucination_guard` in its
`guardrails:`. Neither has a corresponding `{% if %}` block in
`guardrails.j2`: `hallucination_guard`'s enforcement is purely mechanical
(§4.5), but `disclose_ai_and_recording` has **no implementation anywhere**
found in the prompt templates or `services/guardrails/rules.py` — it's
asserted in config but not wired into the prompt or the output check.

### 4.4 `SlotState` (`services/flow/slots.py`)

`current_node_id, filled: dict, reprompt_counts: dict, turn_count,
handoff_count, roster_available: bool`. `roster_available` is set True the
first time `lookup_roster` succeeds this call, and is what
`hallucination_guard` checks before trusting any stated shift/date/time.
`context()` flattens `filled` plus `reprompts` (count for the *current*
node) and `turn_count` — this flattened dict is exactly what
`FlowGraph.next_node_id` evaluates transitions against, so `"reprompts >= 2"`
and a slot name live in the same namespace by design.

### 4.5 Guardrails — three layers, and a confirmed bug

Per `services/guardrails/rules.py`'s docstring: prompt instruction
(`guardrails.j2`) → fast regex classifier (this layer) → post-call QA audit
(`services/agents/qa_agent.py` — **referenced but does not exist in the
repo**, so layer 3 is not built). "Treat this layer as a fast, imperfect
net, not the only net."

**Full `GUARDRAIL_CATALOG`** (`services/guardrails/rules.py`):

| id | severity | deflection |
|---|---|---|
| `no_wage_discussion` | BLOCK | "Payment ke baare mein HR team aapko batayegi..." |
| `no_leave_approval` | BLOCK | "Main aapka reason record kar raha hoon, Roster Planner review karenge." |
| `no_third_party_employee_data` | BLOCK | "Main sirf aapki shift ke baare mein baat kar sakta hoon." |
| `no_disciplinary_threat` | BLOCK | "Main sirf yaad dilane ke liye call kar raha hoon..." |
| `no_medical_advice` | WARN | "Aap safety officer ya doctor se zaroor baat kariye." |
| `rgid_full_number_leak` | BLOCK | "Aapki ID ke aakhri char digit maine confirm kar liye hain." |
| `claims_to_be_human` | BLOCK | "Main ShiftEaze ka automated calling assistant hoon." |

`get_active_rules(guardrail_ids)` always includes `rgid_full_number_leak`
and `claims_to_be_human` regardless of the agent's own list, unions in the
rest, and — importantly — filters with `if i in GUARDRAIL_CATALOG`, so an
unknown id is silently skipped here.

**`hallucination_guard` is *not* a catalog entry.** It's synthesized inline
in `services/guardrails/output.py::check_output_text()`:

```python
if "hallucination_guard" in guardrail_ids and not roster_available:
    if _mentions_shift_details(text):   # "shift","terminal","baje","tareek","roster",...
        violations.append(GuardrailViolation(
            rule_id="hallucination_guard", severity=Severity.BLOCK,
            matched_text=text, description="...",
        ))
```

**Confirmed bug**: `enforce()` — called unconditionally from
`conversation_agent.run_turn` on every turn's final text — does:

```python
blocking = [v for v in violations if v.severity == Severity.BLOCK]
if not blocking:
    return text, violations
top: GuardrailRule = GUARDRAIL_CATALOG[blocking[0].rule_id]   # <-- KeyError
return top.deflection, violations
```

with a plain dict subscript, no `.get()`, no guard — unlike
`get_active_rules`'s `if i in GUARDRAIL_CATALOG` check. Any turn where (a)
`hallucination_guard` is in the agent's configured guardrails (it is, for
`shift_confirm`), (b) `roster_available` is still False (true for every
turn before `lookup_roster` first succeeds — e.g. the entire `greet` node),
and (c) the model's output text contains any of `_SHIFT_DETAIL_HINTS`
(which includes the plain Hindi words "baje"/"tareek" — not rare tokens)
will raise an **unhandled `KeyError: 'hallucination_guard'`**, crashing the
turn. This was caught in this pass while testing the new
`/agent-voice/preview` endpoint against `shift_confirm` with a fake LLM
that happened to say "Dhanyavaad, shift confirm ho gayi" before the roster
had been looked up. **Not fixed in this pass** — flagged here as the single
highest-priority correctness bug in the rebuild. Fix is either: add a
`hallucination_guard` entry to `GUARDRAIL_CATALOG`, or have `enforce()`
special-case that id the same way `check_output_text` does.

`services/guardrails/input.py::scan_input()` is a separate, simpler layer —
flat boolean flags (`prompt_injection_flagged`, `wrong_number_hint`) on
*caller* utterances (not agent output), checked once per turn in
`conversation_agent.run_turn` against the latest user message. Flagging
only, never blocking — the flow engine's structural design (no tool exists
that approves leave, transfers money, etc.) is what actually prevents harm;
the flag exists for QA/logging visibility.

### 4.6 Tool registry (`services/tools/registry.py`) and every tool

The registry is "the single choke point every tool call goes through" —
idempotency, timeouts, and `ToolInvocation` audit logging are enforced once
here, not per-tool.

`@tool(*, idempotent=False, terminal=False, timeout_ms=1200, requires_slots=())`
registers a function by name. `requires_slots` is stored but **never
checked anywhere in `invoke_tool`** — declared on `confirm_shift` but not
enforced; likely documentation-only or a not-yet-wired feature.

`invoke_tool(name, *, call_id, session, args)`:
1. If `idempotent`, checks for an existing `ToolInvocation` row with the
   same idempotency key (`sha256(call_id:tool_name:sorted_json_args)`) and
   `status=="ok"` — if found, returns the cached result **without
   re-invoking the function**.
2. Otherwise inserts a `pending` row first (so even a failed/timed-out call
   is logged), then `asyncio.wait_for(func(**kwargs), timeout=timeout_ms/1000)`.
3. `session`/`call_id` are injected into the call only if the target
   function's own signature declares those parameter names — so
   `lookup_roster(rgid, date)` (no such params) doesn't get them, while
   `confirm_shift(session, call_id, rgid, ...)` does.
4. Timeout → `ToolTimeoutError`, status `"timeout"`. Any other exception →
   re-raised unchanged, status `"error"`. Either way, `duration_ms` is
   always recorded in `finally`.

**Every registered tool:**

| tool | file | idempotent | terminal | touches |
|---|---|---|---|---|
| `lookup_roster(rgid, date)` | roster_tools.py | yes | no | Redis `roster:{rgid}:{date}` (pre-warmed snapshot — "a cold Odoo ORM call inside a live conversation is not acceptable within the tool timeout budget") |
| `confirm_shift(session, call_id, rgid, roster_line_id, intent, reason=None)` | hr_tools.py | yes | no | Odoo webhook `call.shift_confirmed` |
| `mark_absence_intent(...)` | hr_tools.py | yes | no | Odoo webhook `call.absence_intent` |
| `record_ot_response(...)` | hr_tools.py | yes | no | Odoo webhook `call.ot_response` |
| `log_attendance_exception(...)` | hr_tools.py | yes | no | Odoo webhook `call.attendance_exception` |
| `transfer_to_human(reason, target)` | transfer_tools.py | no | **yes** | nothing directly — thin, structured decision only |
| `transfer_to_agent(target_agent_id, reason, carry_slots, context_summary)` | transfer_tools.py | no | **yes** | nothing directly; only `carry_slots`+`context_summary` cross to the new agent, never the raw transcript |
| `end_call(outcome, notes)` | transfer_tools.py | no | **yes** | nothing directly |
| `schedule_callback(rgid, at, reason)` | comms_tools.py | yes | no | Redis `callback:{rgid}:{at}`, 7-day TTL |
| `send_sms_fallback(rgid, template_id, params)` | comms_tools.py | yes | no | `services.integrations.sms_client.send_sms` |

`to_claude_tool_schema(name)` derives a Claude tool-use schema straight from
the Python function's signature (type hints → JSON schema types; no
default = required), skipping `session`/`call_id` params. `RECORD_SLOT_TOOL`
(in `conversation_agent.py`) is a separate, hand-written synthetic tool —
not in this registry — offered on every turn so the model can record a
confirmed slot value in the same call as its spoken reply, avoiding a
second extraction pass.

### 4.7 Agent configs — worked example: `shift_confirm`

`configs/agents/shift_confirm.yaml` (abridged):
```yaml
id: shift_confirm
version: 1
languages: [hi, gu, en]
voice:
  hi: { provider: sarvam, voice_id: meera, speed: 1.0 }
  gu: { provider: azure, voice_id: gu-IN-DhwaniNeural, speed: 0.95 }
llm:
  conversation: { model: claude-haiku-4-5, max_tokens: 150, temperature: 0.4 }
tools: [lookup_roster, confirm_shift, mark_absence_intent, schedule_callback,
        send_sms_fallback, transfer_to_agent, transfer_to_human, end_call]
guardrails: [no_wage_discussion, no_leave_approval, no_third_party_employee_data,
             disclose_ai_and_recording, hallucination_guard]
limits: { max_duration_sec: 180, max_turns: 24, max_reprompts_per_slot: 2 }
retry: { attempts: 3, backoff: ["15m","2h","6h"], on_no_answer: sms_fallback }
calling_window: { start: "07:00", end: "21:00", tz: "Asia/Kolkata" }
```

`configs/flows/shift_confirm.yaml` — entry `greet`, 8 nodes, terminal
nodes `[wrong_person, schedule_callback_node, confirm_and_close,
close_decline]`. Flow: `greet` (collect `identity_verified`) → `state_shift`
(tools: `lookup_roster`; collect `shift_acknowledged`) → `ask_intent`
(collect `attendance_intent`; branches to `confirm_and_close` /
`capture_reason` / `probe_once` / `HANDOFF:ot_request`) → `capture_reason`
(tools: `mark_absence_intent`) → `close_decline`. `probe_once` and `greet`
both have a `reprompts >= N` transition to `schedule_callback_node` as their
give-up path.

**Acceptance test for this whole subsystem** (per `agent_registry.py`'s
docstring): adding a 6th agent requires exactly 3 new files — one
`configs/agents/<id>.yaml`, one `configs/flows/<id>.yaml`, one
`prompts/agents/<id>/persona.j2` — and **zero Python changes**. Config
validation is strict (`extra="forbid"` on every Pydantic model in
`agent_registry.py`) so a stray YAML key fails loudly at load time.

### 4.8 `conversation_agent.run_turn` — full turn lifecycle

1. Scan the latest user message for prompt injection (flag only, logged).
2. Resolve the current node, render the full system prompt (§4.3).
3. Build `tools = [RECORD_SLOT_TOOL] + [schema for t in node.tools]`.
4. Call the LLM (default `generate_conversation_turn`, non-streaming — see
   §4.9 for why).
5. Apply any `record_slot` calls to `SlotState.filled`.
6. If there are **real** tool calls: invoke each via `invoke_tool`
   (`lookup_roster` success sets `slot_state.roster_available = True` right
   here, before the guardrail check later in the same turn — so a turn that
   both looks up the roster and states shift details in its follow-up text
   will not trip `hallucination_guard`, since the flag is already True by
   the time `enforce()` runs); feed all tool results back in **one** resume
   round (no recursive tool-calling — "a tool that needs a second tool call
   is a flow-design smell, not something this loop should paper over"). A
   `ToolTimeoutError` becomes a tool-result telling the model to promise an
   SMS follow-up and explicitly "do not guess the missing information" —
   the timeout handling is itself a guardrail-reinforcing instruction fed
   back into the same turn. Any other tool exception similarly becomes an
   `is_error` tool-result rather than propagating — a broken tool degrades
   one turn's output, it doesn't crash the call.
7. Run the final text through `enforce_guardrails()` (§4.5 — this is where
   the `hallucination_guard` bug can fire).
8. If a terminal tool fired (`transfer_to_human`/`transfer_to_agent`/
   `end_call`), return immediately with that decision. Otherwise evaluate
   `flow_graph.next_node_id()`: `None` → `"reprompt"` (bump the reprompt
   counter); `"HANDOFF:x"` → `"handoff"` (not implemented downstream —
   §4.11 ends the call gracefully instead); else `"advance"` or
   `"advance_terminal"`.

`TurnOutcome` carries `text_to_speak`, `slot_state`, `guardrail_violations`,
`tool_calls` (with per-call status), `decision`, `decision_detail`,
`tokens_in`/`tokens_out`, `input_injection_flagged`.

### 4.9 Why `generate_conversation_turn` is non-streaming

`services/claude_client.py::generate_conversation_turn` is a plain
(non-streaming) `messages.create` call — the loop above needs to see
whether a `tool_use` block came back before it can decide whether to run a
resume round, which a token stream can't tell you mid-flight. The
docstring commits to a design that real sentence-level streaming to TTS
happens *downstream* of the final text, via `services/pipeline/chunker.py`
(§4.13) — that's the piece this pass actually built.

### 4.10 `services/agent_registry.py` — DB sync

`sync_configs_to_db()` (called from `db.init_db()`, which `main.py` now
calls on every boot — see §4.14) upserts every `configs/agents/*.yaml` into
`Agent`/`AgentVersion`/`Prompt`/`FlowGraph` rows, comparing
`persona_hash`/`flow_hash` (sha256 of the rendered persona + raw flow YAML)
before writing a new version row — a no-op if nothing changed. This is also
where config validation actually happens on every boot: `get_agent_config()`/
`get_flow_graph()` raise `ValueError` on a bad YAML, and nothing here
catches that — a broken agent config **fails server startup**, not the
first live call to it.

### 4.11 Compliance gate + dialer queue (`services/dispatch/*`)

**`admission.py::can_dial(contact, campaign, now=None)`** — the single
choke point every dial must pass, checked in this exact order:
1. `is_kill_switch_engaged()` — read fresh from Redis every time (never
   cached), so flipping the switch takes effect on the very next check, not
   after a restart.
2. `contact.dnc` → refuse, `reason_code="dnc"`.
3. `contact.consent_status != "granted"` → refuse, `reason_code="consent"`.
4. Outside `campaign.calling_window` (timezone-aware via `zoneinfo`,
   correctly handles an overnight-wrapping window like 22:00–06:00 even
   though no shipped agent config uses one yet) → refuse,
   `reason_code="calling_window"`.

`record_opt_out(session, contact, source)` sets `dnc=True` **and**
`consent_status="revoked"` together, specifically so no code path that only
checks one of the two fields can re-dial — plus a `ConsentRecord` audit row.
Because `can_dial` always reads the live `Contact` row, an opt-out recorded
here is picked up by the very next dispatch cycle automatically.

**`dialer.py::run_dispatch_cycle()`** — one pass, no owned event loop:
picks up to `batch_size` `pending`, due (`next_attempt_at <= now`)
`CallTask` rows ordered by `priority DESC, created_at ASC`, capped by
remaining concurrency capacity (`max_concurrent_calls - active_call_count()`).
For each: `can_dial()` gates it. `kill_switch` refusal leaves the task
`pending` untouched (transient, retry next cycle for free). `dnc`/`consent`/
`calling_window` refusal marks it `abandoned` (not retryable by waiting — a
human decision is needed) **without consuming a retry attempt**. On
admission, calls `provider.originate_call(...)`; on provider failure,
schedules a backoff retry (`_schedule_retry_or_fail`, reading the agent's
own `retry.backoff`/`retry.attempts`, falling back to a hardcoded
`["15m","2h","6h"]` schedule if the agent config can't be loaded) or marks
`failed` if attempts are exhausted. On success, creates a `Call` row,
marks the task `in_call`, increments the Redis concurrency counter. Does
**not commit** — the caller owns the transaction, only `session.flush()`.

**`handle_call_status_callback()`** — Twilio's status webhook lands here.
Maps `busy|no-answer|failed|canceled` → retry-or-fail via the same backoff
logic; `completed` → task `completed`; anything else (ringing/in-progress)
is a no-op. Always decrements the concurrency counter on any terminal
status via `mark_call_ended()`.

**`reconcile_active_call_count(session)`** (uncommitted) — recomputes the
Redis concurrency counter from `count(CallTask.state=="in_call")` instead
of trusting whatever Redis held across a restart. Closes a real leak: an
ungraceful shutdown (crash, `kill -9`, OOM) skips `mark_call_ended()`'s
decrement in `agent_voice.py`'s `finally`, permanently occupying a
concurrency slot until the process restarts — there was no self-healing
before this. Wired into `main.py`'s startup sequence, and must run **before**
the dialer loop starts (see §4.14 for why registration order matters here).

### 4.12 `routers/campaigns.py` — the REST surface

`POST/GET /contacts`, `POST /contacts/{id}/opt-out`, `POST/GET /campaigns`,
`GET /campaigns/{id}/tasks`, `POST/GET /kill-switch`, `POST /dialer/run-once`
(manual dispatch trigger, useful when `dialer_enabled=False`), `POST
/dialer/call-status` (Twilio's webhook). `create_campaign` validates
`agent_id` against `agent_registry.get_agent_config()` before creating any
`CallTask` rows, and de-dupes `contact_ids` while preserving order.

**Auth (uncommitted)**: every endpoint except `/dialer/call-status`
requires `X-API-Key` when `ADMIN_API_KEY` is configured
(`Depends(require_admin_key)`, `services/auth.py`) — the Twilio webhook is
deliberately excluded since Twilio cannot send a custom header; it's
protected only by the unguessable `CallSid`. Left unset (the default),
these endpoints run with **no authentication at all**, logged loudly once
per process.

### 4.13 `services/pipeline/chunker.py` (uncommitted)

```python
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")

def split_sentences(text: str) -> list[str]:
    ...
```

A deliberately simple heuristic (not NLP): splits on whitespace following a
Latin `.!?` or a Hindi/Gujarati danda (`।`/`॥`). `routers/agent_voice.py`'s
`prompt` handler now sends each sentence as its own
`{"type":"text","token":chunk,"last":bool}` message instead of the whole
(already-fully-generated) reply in one shot — this is the concrete
implementation of the design `generate_conversation_turn`'s docstring
already committed to (§4.9): TTS can start speaking the first sentence
while later ones are still being transmitted, rather than waiting for the
entire multi-sentence reply to cross the WebSocket first.

### 4.14 `routers/agent_voice.py` — the live call entrypoint

**`GET/POST /agent-voice/twiml?task_id=...`** — `task_id` is a `CallTask`
id; only `services/dispatch/dialer.py` should ever build this URL. Loads
contact/campaign context, loads the agent config + flow graph, runs **one**
`run_turn()` call against a synthetic bootstrap message
(`CALL_CONNECTED_SENTINEL = "[CALL_CONNECTED]"`) to get the opening line
(the rebuild's agents have no fixed scripted greeting the way Track B
does — the opening line is LLM-generated from the entry node's objective),
and returns TwiML with that text as `welcomeGreeting`.

**`WS /agent-voice/media-stream`** — on `setup`: requires a `taskId` custom
param (ends the call with `outcome:"config_error"` if missing/unknown);
finds or creates the `Call` row; creates an `AgentSpan`; **re-runs the
identical bootstrap `run_turn()` call** rather than threading the greeting
string through a TwiML `<Parameter>` — deliberately, because
`<Parameter>` values show up in Twilio's own debug console (an odd place
for a multi-sentence AI-generated greeting) and a long string is an awkward
XML-attribute fit. Cost: one extra Claude call per answered call, paid once.

On `prompt`: fires sentiment classification concurrently with `run_turn`
(§2.5's pattern, mirrored here), persists both turns via `_persist_turn`
(now accepting an optional `sentiment` dict, §4.15), sends the reply
sentence-chunked (§4.13), and handles the decision: `handoff` → ends
gracefully with `outcome:"handoff_not_implemented"` (multi-agent handoff
execution genuinely isn't built — flagged, not silently mishandled);
`end_call`/`transfer_to_human` → finalizes and ends; anything else → call
stays open.

`finally`: if the socket closed without a clean `ended` flag (dropped
connection), finalizes with `outcome_name="call_dropped"`,
`status="disconnected"` — so the `CallTask` doesn't stay stuck `in_call`
forever, permanently occupying a concurrency slot.

**`POST /agent-voice/preview` (uncommitted)** — dry-run a config-driven
agent through a scripted list of caller utterances, no Twilio, no
`CallTask`, no dialer. Runs real Claude calls through `run_turn`, against a
DB session that is **rolled back at the end, never committed**, so no
`Call`/`Turn`/`CallOutcome` rows are left behind. This is the "test a new
agent before dialing anyone" tool — mirrors Track B's `/scenarios/preview`.
**Caveat documented in its own docstring**: if the scripted utterances
drive the flow into a tool with a real external side effect (e.g. the Odoo
webhook in `hr_tools.py`) while `ODOO_WEBHOOK_URL` is configured, that HTTP
call still happens for real — rollback only undoes local DB writes, not
network requests already sent. Don't preview flows that reach terminal HR
tools against a live Odoo webhook.

`_finalize_call()` sets `Call.status`/`hangup_by`, writes one `CallOutcome`
row (`slots_json` from the final `SlotState.filled`), marks the `CallTask`
`completed`/`abandoned`, and always calls `mark_call_ended()` to free the
concurrency slot — **not idempotency-guarded against being called twice**
for the same call (e.g. an `end_call` decision followed immediately by the
socket disconnecting before the loop exits cleanly could in principle
double-insert a `CallOutcome` row; not observed in verification, not
structurally prevented either).

### 4.15 Data model (`models.py`) — the rebuild's tables

`Agent`/`AgentVersion`/`Prompt`/`FlowGraph` — the published, auditable
snapshot of `configs/agents/*.yaml` (§4.10). `Contact` (`consent_status`,
`dnc`, `language_pref`) / `ConsentRecord` (append-only audit trail) /
`Campaign` (calling window, concurrency cap) / `CallTask` (the dialer queue
itself — `state`, `next_attempt_at`, `idempotency_key`, `priority`).
`Call` / `AgentSpan` (multi-agent trace — one row per agent that
participated in a call) / `Turn` (per-utterance row, **including
`sentiment_label`/`sentiment_score`/`emotion`/`escalate` columns** — now
actually populated on the rebuild path as of the uncommitted sentiment
work, §4.14) / `ToolInvocation` (idempotency/audit log, written by
`invoke_tool`) / `CallOutcome` (one per call, `slots_json` +
`primary_outcome`) / `QAScore` (schema exists; nothing writes to it — layer
3 of the guardrail system, `qa_agent.py`, isn't built) / `WebhookDelivery`
(Odoo delivery bookkeeping, §4.16).

Table creation + config sync happens on every boot via `main.py`'s
`_init_rebuild_db` startup hook calling `db.init_db()` — previously nothing
called this at all (`models.py`'s tables were only ever created inside test
fixtures), a gap fixed as part of the Phase 2 pass.

**Startup event ordering matters** (uncommitted fix): `main.py` registers
`_init_rebuild_db` → `_reconcile_dialer_state` → `_start_dialer_loop`, in
that order, with an explicit comment explaining why: the counter
reconciliation needs the tables `_init_rebuild_db` creates, and the dialer
loop's first cycle needs the reconciled counter in place before it makes
its first admission decision. FastAPI runs `"startup"` handlers in
registration order, so this ordering is load-bearing, not cosmetic.

### 4.16 Odoo integration (`services/integrations/odoo_client.py`)

`send_event(session, *, call_id, event, payload)` — every call (regardless
of outcome) writes a `WebhookDelivery` row first (`status="pending"`),
*then* attempts delivery, so nothing is ever silently dropped from the
audit trail. Canonical (sorted-key) JSON serialization is both hashed
(`payload_hash`) and HMAC-signed (`sign_payload`, `hashlib.sha256`) with
`settings.odoo_webhook_hmac_secret`, sent as header `X-ShiftEaze-Signature`.
If `ODOO_WEBHOOK_URL` isn't configured, short-circuits to
`status="dead_letter"` immediately rather than silently no-op'ing — visible
in the audit trail that nothing was actually sent. On any HTTP exception,
`status` stays `"pending"` (eligible for a retry sweep) rather than being
marked failed. **The retry sweep itself (`workers/postcall.py`,
mentioned in comments) does not exist** — `workers/` is an empty package.

### 4.17 `routers/analytics.py` — read/reporting surface (uncommitted)

Before this file existed, the rebuild's richest data (`Call`/`Turn`/
`AgentSpan`/`CallOutcome`) had **zero read endpoints** — `agent_voice.py`
wrote it, nothing served it back.

- `GET /agent-voice/calls` — paginated, filterable by `agent_id`,
  `campaign_id`, `status`, `since`/`until`; outer-joins `CallOutcome` so a
  dropped call (no outcome yet) still appears.
- `GET /agent-voice/calls/{call_id}` — full detail: the call row, every
  `Turn` in order (with sentiment), every `AgentSpan`, the `CallOutcome`.
- `GET /analytics/summary` — aggregate stats over the same filters: total
  calls, completed calls, completion rate, average duration, average
  per-turn latency, outcome breakdown (`group by CallOutcome.primary_outcome`),
  sentiment breakdown (`group by Turn.sentiment_label`). Plain SQLAlchemy
  `func.count`/`func.avg`/`group_by` — no separate aggregation store, "not
  needed at this data volume."

Same `require_admin_key` dependency as `campaigns.py` — transcripts and
outcomes are sensitive.

### 4.18 `services/auth.py` — admin auth (uncommitted)

One dependency, `require_admin_key`, checking an `X-API-Key` header against
`settings.admin_api_key`. If that setting is unset (the default), every
request passes through — but logs one loud warning per process the first
time it's hit, rather than silently running open forever without a trace.
Deliberately **not** applied to `POST /dialer/call-status` (Twilio's own
webhook, which can't send a custom header).

---

## 5. Provider abstraction layer — built, mostly unwired

`services/telephony/base.py`'s `TelephonyProvider` ABC (`originate_call`,
`build_connect_response`, `terminate_call`, `transfer_call`) states its own
scope explicitly: *"Everything about the audio itself is Pipecat's job."*
Four implementations: `mock.py` (default, no network calls, generates
`mock-call-<uuid>` sids — lets the dialer be built/tested before any vendor
contract exists), `twilio_provider.py` (a lift of the logic that already
lived in `routers/calls.py` — "battle-tested... has actually placed and
answered real calls," though `transfer_call`'s warm-transfer is an
unfinished stub: *"requires a small TwiML endpoint... Phase 5/7"*),
`exotel.py` (*"NOT verified against a live Exotel account"*, `transfer_call`
is a hard `NotImplementedError`), `plivo.py` (*"confirm the exact `<Stream>`
attribute set... before a live pilot"*). `registry.py::get_telephony_provider()`
is the only place that branches on `settings.telephony_provider`.

**Wiring reality**: `get_telephony_provider()` has exactly one production
call site — `services/dispatch/dialer.py`, inside `run_dispatch_cycle()`.
**`routers/calls.py`'s `POST /make-call`** (Track A's actual, proven call
path) does **not** use this ABC at all — it imports `twilio.rest.Client`
directly. So the ABC layer is real and tested, but only reachable through
the newer campaign/dialer subsystem, not the original call-placement path.

`services/stt/` and `services/tts/` mirror the same pattern
(`base.py`/`registry.py`/`mock.py` + real providers), but **`get_stt_provider`
and `get_tts_provider` have zero callers anywhere outside their own files**
— confirmed by grep. None of the three ConversationRelay-based routers
(`voice.py`, `voice_scenario.py`, `agent_voice.py`) import from either
package; all three rely entirely on Twilio ConversationRelay doing STT/TTS
itself. `services/stt/deepgram.py` and `services/stt/sarvam.py` both leave
`stream_transcribe` as a hard `NotImplementedError` (Deepgram: needs a
queue bridge around its callback-based SDK; Sarvam: needs a live key to
verify the streaming websocket frame envelope) — `deepgram-sdk` is
deliberately not even in `requirements.txt`. `pipecat-ai` **is** in
`requirements.txt` (labeled "Phase 2 real-time media framework") but is
imported by **zero** `.py` files in the repo — a placeholder dependency for
work that was explicitly decided against adopting in this pass (see
`PHASE2_ARCHITECTURE_PROPOSAL.md` §2: keep ConversationRelay).

---

## 6. Local AI — fully offline pipeline (`local_ai/`)

A third, independent backend: `faster-whisper` (STT) → Ollama (LLM) →
Piper (TTS), file-in/file-out only — explicitly not a streaming/mic mode.
Wired into Track B in exactly one place: `SCENARIO_LLM_BACKEND=local` swaps
the branch classifier (§3.9). `local_ai/config.py` calls `load_dotenv()`
itself, independently of `config.py`'s own call, with a comment explaining
the duplication is intentional (git history: removing an "incidental"
import once silently broke this).

**Measured on the actual dev machine (Ryzen 7 5800HS, 8c/16t, CPU-only, no
GPU)**, classifying the same 3 Hindi test utterances:

| Model | Latency | Accuracy |
|---|---|---|
| phi4-mini (2.5GB) | 23–33s/turn | 1/3 — missed the negation trap and an escalation intent |
| qwen2.5:7b-instruct (4.7GB, default) | 33–60s/turn | 3/3 |
| Claude (cloud, comparison) | ~1–3s | 3/3, proven on a live call |

`local_ai/README`-equivalent is the root `LOCAL_AI_README.md`, which states
bluntly: *"20–60 seconds of dead air after every sentence is not a usable
product regardless of classification accuracy"* — recommendation is Claude
for live calls, local strictly for offline/batch/dev/QA. No Piper Gujarati
voice exists in the actual 171-voice catalog (verified, not assumed) — use
cloud TTS for Gujarati regardless of backend setting.

`local_ai/stt_engine.py` loads faster-whisper with `local_files_only=True`
first, only falling back to `False` if that raises — without this,
`huggingface_hub` silently phones home on every load even when nothing
needs downloading, caught by a dedicated offline-network-guard smoke test
during development, "not a hypothetical concern."

`services/verbalize.py` (Hindi/Gujarati number/time-to-speech — `verbalize_number`,
`verbalize_time`, `verbalize_date`, `verbalize_shift_code`) is real, unit
-tested, and **imported by nothing in production** — only its own test
file references it. Its own docstring flags the Gujarati tables
specifically as a "best-effort scaffold" needing native-speaker
verification before a real pilot.

---

## 7. Cross-cutting infrastructure

**`config.py`** — one `pydantic_settings.BaseSettings` object (`settings`),
read from `.env`. Calls `load_dotenv()` itself so it's the one place
guaranteed to run before anything else does (a fragile prior version
relied on an incidental import, and removing that import once silently
broke `SCENARIO_LLM_BACKEND=local` — noted in-code as a git-history lesson).
Retains a block of backwards-compatible module-level aliases
(`ANTHROPIC_API_KEY`, `TWILIO_ACCOUNT_SID`, etc.) purely so not-yet-migrated
modules (`claude_client.py`, `routers/calls.py`) can keep doing plain-name
imports instead of `from config import settings`.

**`db.py`** — async SQLAlchemy engine/session (`get_session_factory()`,
`session_scope()` for non-request code, `get_session()` FastAPI dependency)
+ Redis client (`get_redis()`), both lazily constructed singletons.
`init_db()`'s own docstring calls it a dev/test convenience
(`CREATE TABLE IF NOT EXISTS` semantics via SQLAlchemy metadata) and says
real deployments should run `alembic upgrade head` instead.

**`services/claude_client.py`** — every Anthropic API call in the repo goes
through this one module. `_get_client()` is a lazy singleton that raises a
friendly `RuntimeError` (not at import time) if `ANTHROPIC_API_KEY` is
unset, so the server can still boot and serve health checks / CRUD without
a key configured. Key functions: `generate_reply_stream`/`generate_reply`
(Track A, `thinking={"type":"disabled"}` explicitly — *"waiting on extended
thinking tokens before the callee hears anything would be a bad user
experience"*), `classify_sentiment` (shared, Haiku, structured JSON-schema
output), `classify_structured` (Track B's branch classifier, generic schema
param), `generate_conversation_turn` (the rebuild, non-streaming, tool-use
enabled — §4.9), `generate_supervisor_guidance` (built for
`services/agents/supervisor_agent.py`, which **does not exist** — a stub
for unbuilt work). Every system prompt across every track gets
`NO_TAG_LEAK_INSTRUCTION` appended: *"Do not include internal or system XML
tags in your response... no labels, no markdown, no stage directions."*

### Environment variables (full reference)

| Variable | Default | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | — (required) | all tracks |
| `CONVERSATION_MODEL` | `claude-haiku-4-5` | Track B classifier, rebuild conversation model default |
| `SUPERVISOR_MODEL` | `claude-sonnet-5` | not-yet-wired supervisor agent |
| `SENTIMENT_MODEL` | `claude-haiku-4-5` | shared sentiment classifier |
| `DEFAULT_CLAUDE_MODEL` | `claude-opus-5` | Track A default agent model |
| `DATABASE_URL` | empty → sqlite `./data/app.db` | rebuild (`db.py`) |
| `REDIS_URL` | `redis://localhost:6379/0` | dispatch/admission, roster cache |
| `TELEPHONY_PROVIDER` | `mock` | dialer only (§5) |
| `TWILIO_*` | — | Track A/B direct calls, `twilio_provider.py` |
| `EXOTEL_*`, `PLIVO_*` | — | unverified alt telephony providers |
| `STT_PROVIDER` / `TTS_PROVIDER` | `mock` | unwired (§5) |
| `SARVAM_API_KEY` / `DEEPGRAM_API_KEY` / `ELEVENLABS_API_KEY` / `AZURE_SPEECH_*` | — | unwired provider modules |
| `ODOO_WEBHOOK_URL` / `ODOO_WEBHOOK_HMAC_SECRET` | — / `change_me_in_production` | `services/integrations/odoo_client.py` |
| `SMS_PROVIDER` | `mock` | `comms_tools.send_sms_fallback` |
| `NGROK_URL` | — | Track B (fresh per-call) + rebuild dialer; Track A reads a frozen alias instead |
| `GLOBAL_KILL_SWITCH_REDIS_KEY` | `ai_calling:kill_switch` | admission gate |
| `MAX_CONCURRENT_CALLS` | `50` | dialer capacity |
| `DIALER_ENABLED` | `false` | background dispatch loop opt-in |
| `DIALER_POLL_INTERVAL_SECONDS` | `15` | background dispatch loop cadence |
| `ADMIN_API_KEY` | unset (no auth) | `routers/campaigns.py`, `routers/analytics.py` |
| `SCENARIO_LLM_BACKEND` | `claude` | Track B branch classifier local/cloud switch |
| `OLLAMA_*`, `LOCAL_*` | see `local_ai/config.py` | offline pipeline only |

---

## 8. Testing

`tests/unit/` — `test_admission.py`, `test_dialer.py` (fakeredis + in-memory
SQLite, the canonical fixture pattern used by every dispatch-layer test),
`test_analytics.py`, `test_chunker.py`, `test_flow_graph.py`,
`test_guardrails.py`, `test_idempotency.py`, `test_verbalize.py`.
`tests/integration/` — `test_conversation_agent.py` (a `FakeLLM` class
queues canned `ConversationTurnResult`s, run against a real in-memory-SQLite
`session` so tool code genuinely executes), `test_agent_voice_preview.py`
(uncommitted — monkeypatches `services.claude_client._get_client`, not
`run_turn`'s `llm_call` default, since that default is bound at function-
definition time and a later monkeypatch of the module attribute would not
retroactively affect it). `tests/local_ai/` — offline-only, includes a
network-blocking smoke test.

As of this document: **107 passed, 2 skipped** on this machine (the 2
skips are `piper-tts` not being installed in this Python environment, an
environment gap, not a code bug) — *except* the 3 new
`test_agent_voice_preview.py` tests, which currently fail on the confirmed
`hallucination_guard` bug (§4.5) rather than a test-authoring problem: the
fake LLM's second canned response happens to mention shift details before
`roster_available` is true, tripping the exact crash path. This is
evidence *for* the bug, not evidence the test is wrong.

---

## 9. Consolidated list of known gaps

Everything below is either explicitly flagged in code comments or confirmed
by direct reading this session — none of it is a guess:

1. **`hallucination_guard` KeyError** (§4.5) — the highest-priority
   correctness bug found. Unhandled `KeyError` crashes any turn where the
   agent mentions shift details before the roster's been looked up.
2. **`disclose_ai_and_recording` has no enforcement** anywhere (no
   `guardrails.j2` block, no `rules.py` entry) despite being listed in
   `shift_confirm.yaml`'s `guardrails:`.
3. **`FlowNode.on_max_reprompts`** and **`ToolSpec.requires_slots`** are
   both defined fields that nothing reads — likely scaffolding for
   unbuilt enforcement.
4. **No layer-3 QA audit** — `services/agents/qa_agent.py` is referenced in
   comments, doesn't exist; `models.QAScore` exists, nothing writes to it.
5. **No handoff execution** — `decision="handoff"` is recognized and
   flagged, but the actual cross-agent transfer isn't implemented; the call
   just ends.
6. **No double-finalize guard** on `agent_voice.py::_finalize_call` — could
   in principle write two `CallOutcome` rows for one call; not observed,
   not structurally prevented.
7. **STT/TTS provider abstractions are entirely unwired** — real code,
   zero callers outside their own modules; every live call path uses
   Twilio ConversationRelay's built-in STT/TTS instead.
8. **`pipecat-ai` is a dependency with zero imports** — reserved for future
   work explicitly *not* adopted in the Phase 2 pass (see
   `PHASE2_ARCHITECTURE_PROPOSAL.md` §2).
9. **`services/verbalize.py` is unwired** — tested, not called from
   production code; live number/time verbalization instead relies on
   callers pre-formatting `facts` correctly (Track B) or the LLM's own
   judgment (rebuild).
10. **`workers/postcall.py` retry sweep doesn't exist** — `WebhookDelivery`
    rows can sit `"pending"` forever after a failed Odoo POST with no
    process retrying them.
11. **No auth by default** on `routers/campaigns.py`/`routers/analytics.py`
    — `ADMIN_API_KEY` must be explicitly set; unset means fully open.
12. **Track B's stray `pending:*` scenario_calls rows** are never deleted
    after being re-keyed onto the real call SID.
13. **Twilio call-status field names/vocabulary are unverified live** — no
    Twilio credentials configured on this dev machine; written to Twilio's
    documented shape only.
14. **`services/telephony/twilio_provider.py`'s warm transfer** is an
    acknowledged stub (`transfer_call`, "Phase 5/7").
15. **Concurrency-counter leak fixed but startup-order-dependent** — the
    fix (§4.15) only works because of the explicit registration order in
    `main.py`; reordering those startup hooks would silently reintroduce
    the bug.

---

## 10. Glossary

- **ConversationRelay** — Twilio's public-beta product that does STT/TTS/
  barge-in itself and exchanges plain JSON text with our WebSocket. Used by
  all three tracks; none of them touch raw audio.
- **RGID** — the employee/operator identifier used throughout ShiftEaze
  (Contact.rgid, roster lookups); only ever spoken back as its last 4
  digits, never in full (`rgid_full_number_leak` guardrail).
- **CallTask** — the rebuild's dialer queue row; one per (contact, campaign,
  attempt). Its `state` machine: `pending → originating → in_call →
  completed|failed|abandoned` (with `next_attempt_at`-gated retries).
- **Flow graph / node / objective** — the rebuild's YAML-defined state
  machine; a node's `objective` is the only thing the LLM is told about
  where the conversation currently is.
- **Branch (Track B)** — one pre-written reply + outcome in a scenario
  JSON; the AI classifies onto a branch, never writes free text.
- **Idempotency key** — `sha256(call_id:tool_name:sorted_args_json)`,
  used both by the tool registry (skip re-invoking a completed idempotent
  tool) and `CallTask` (`f"{campaign_id}:{contact_id}:{attempt}"`, prevents
  duplicate dial rows for the same attempt).
