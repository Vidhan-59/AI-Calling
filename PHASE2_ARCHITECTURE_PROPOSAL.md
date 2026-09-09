# Phase 2 — Architecture Proposal

Status: implemented in this pass (not just proposed). This document records
the decisions made while building the compliance gate, the dialer, and the
live wiring of the rebuild's conversation loop, and the trade-offs behind
each one, per the master prompt's Phase 2 requirement to reason about
architecture before writing Phase 3+ code.

## 1. Scope recap

Before this pass, per AUDIT.md, the "rebuild" (config-driven agent registry,
flow-graph state machine, tool registry, guardrails) was fully built but
reachable only from tests — nothing routed a real phone call into it, nothing
decided which numbers to dial or when, and nothing enforced consent/DNC/
calling-window/kill-switch rules before a call was placed. This pass closes
all three gaps: `routers/agent_voice.py` (live call wiring), `services/
dispatch/dialer.py` + `services/dispatch/admission.py` (dialer + compliance
gate), and `routers/campaigns.py` (the REST surface to load contacts,
create campaigns, and administer the dialer).

## 2. Decision: reuse ConversationRelay, don't adopt Pipecat

Twilio ConversationRelay was already proven live in this codebase — `routers/
voice.py` and `routers/voice_scenario.py` both use it, both work today. It
hands STT, TTS, and barge-in entirely to Twilio and gives the app a plain
text-in/text-out WebSocket contract (`setup`/`prompt`/`interrupt`/`dtmf`/
`error` in; `{"type":"text",...}` / `{"type":"end",...}` out).

Pipecat (or a raw Twilio Media Streams pipeline) would mean owning STT/TTS/
VAD/barge-in ourselves — real engineering surface, and a new framework
dependency, which the master prompt's guardrails require asking about before
adopting on the core pipeline. Nothing about hooking up the rebuild's
conversation loop requires that: the flow-graph agent just needs a place to
receive an utterance and return a line of text, which is exactly what
ConversationRelay already gives Track B. So `routers/agent_voice.py` reuses
the identical contract, swapping only what sits behind it (`conversation_agent
.run_turn` instead of the scenario engine).

Trade-off accepted: this inherits ConversationRelay's constraints (Twilio-
hosted STT/TTS choice, Twilio's set of supported voices, no access to raw
audio for e.g. custom prosody or local Whisper). If a future requirement
needs audio-level control this doesn't cover, that's the point where Pipecat
becomes worth the new dependency — revisit then, not preemptively.

## 3. Decision: `models.CallTask` is the dialer queue, not Redis Streams

`models.CallTask` (state / next_attempt_at / idempotency_key / priority)
already existed in `models.py`, unused. Two options were on the table for
the actual dispatch queue: (a) use this table directly as the queue, with the
dialer polling it, or (b) introduce Redis Streams as the queue and treat
`CallTask` as a denormalized read-model kept in sync with it.

Chose (a). Reasoning: a `CallTask` row and a Streams entry are two sources of
truth for the same fact ("is this call pending/in-flight/done"), and every
extra source of truth is a place they can drift — a crash between "ack the
stream entry" and "write the row" is exactly the kind of bug that's easy to
introduce and hard to catch in a demo, and only shows up under production
failure conditions. A single source of truth removes that failure mode
entirely at the cost of polling latency (`dialer_poll_interval_seconds`,
default 15s) instead of push-based delivery — an acceptable trade for an
outbound dialer, where the )dispatch loop already isn't the caller-facing
latency path.

Redis is still used, but only for the two things that are genuinely
cross-process, ephemeral, and need to be fast: the live concurrency counter
(`ai_calling:active_call_count`, incremented on originate, decremented on
call end) and the kill switch (`GLOBAL_KILL_SWITCH_REDIS_KEY`). Neither of
those has a natural home in a durable row — a counter that's fast to
increment/decrement across processes, and a switch that must be checked on
every admission decision with minimal latency, are exactly Redis's job.

## 4. Decision: one `can_dial()` gate, checked once, in one place

`services/dispatch/admission.py::can_dial()` is the single pre-dial
compliance check: kill switch → DNC → consent (`consent_status == "granted"`)
→ calling window (timezone-aware, handles overnight/wraparound windows via
`zoneinfo`). `run_dispatch_cycle` calls it once per task, before origination,
and nowhere else in the codebase decides whether a number is allowed to be
called. That's deliberate — a compliance rule duplicated in two call sites is
a compliance rule that's one edit away from being wrong in one of them.

Reason codes distinguish *why* a call was refused because the right response
differs: `kill_switch` is transient and operational (leave the task
`pending`, retry as soon as the switch lifts — no human needed); `dnc` and
`consent` are not something retrying will fix (mark the task `abandoned`,
a human decision like re-consent is required); `calling_window` similarly
does not consume a retry attempt, since it isn't the contact's fault the
agent tried outside the allowed hours.

## 5. Decision: bootstrap-greeting via a sentinel turn, not a TwiML parameter

Track B's ShiftEaze scenarios have a scripted opening line rendered by
`scenario_engine.opening_line()`. The rebuild's flow-graph agent has no such
fixed string — the opening line is generated by the LLM from the entry
node's objective, same as every other turn.

TwiML needs a `welcomeGreeting` string before the call ever connects to the
WebSocket, so `agent_voice_twiml` runs one `run_turn()` call against a
synthetic message (`CALL_CONNECTED_SENTINEL = "[CALL_CONNECTED]"`) to
produce that line. The WebSocket handler then re-runs the *identical* call
on `"setup"`, rather than threading the already-generated string through a
TwiML `<Parameter>`, for two reasons: ConversationRelay's `<Parameter>`
values appear in Twilio's own debug console (an odd place for a multi-
sentence AI-generated greeting to show up in an operator's call log), and a
long string is an awkward fit for an XML attribute in the first place. The
cost is one extra LLM call per answered call — a fixed, small cost, paid
once at answer time, not per turn.

This is new — no other part of the codebase does bootstrap greetings this
way, since Track B never needed to. It has not been checked against a real
Twilio call in this sandbox (no Twilio credentials configured — see
AUDIT.md); it has been verified structurally end-to-end with the LLM call
stubbed out (see "What was verified" below).

## 6. What was verified, and how

- **Compliance gate and dialer**: `tests/unit/test_admission.py` (10 tests)
  and `tests/unit/test_dialer.py` (10 tests), using `fakeredis` and an
  in-memory SQLite DB — cover kill switch priority, DNC/consent refusal,
  wraparound calling windows, concurrency-ceiling admission, retry
  scheduling and exhaustion, and the call-status callback's terminal-state
  handling. All pass; a real bug (`session.flush()` missing after ORM
  mutations, causing them to be invisible/reverted on the next query in the
  same transaction) was caught by these tests and fixed in `dialer.py` — see
  the inline comment above `await session.flush()` in `run_dispatch_cycle`
  and `handle_call_status_callback`.
- **The REST surface, end to end, through real HTTP + real Redis**: a
  one-off script (`verify_e2e.py`, not part of the pytest suite, run against
  a local `redis-server` and `TestClient(main.app)` used as a context
  manager so startup events actually fire) created contacts, hit the
  duplicate-rgid 409, created a campaign, listed tasks, read/toggled the
  kill switch against real Redis, and called `POST /dialer/run-once` twice —
  once correctly failing (no `NGROK_URL` configured) and once, with
  `routers.campaigns.get_public_url` patched to a fake tunnel URL and the
  mock telephony provider, succeeding: the granted-consent contact was
  originated (task → `in_call`, a `Call` row created) and the
  unknown-consent contact was correctly skipped (task → `abandoned`,
  `skip_reasons: {"consent": 1}`).
- **The live-call wiring (`routers/agent_voice.py`)**: a second one-off
  script (`verify_agent_voice.py`) exercised the TwiML endpoint's error paths
  (missing/unknown `task_id`), the happy path (TwiML shape, greeting
  embedded, `taskId` `<Parameter>` present), and the full WebSocket
  `setup` → `prompt` → `end` flow against a real `TestClient` WebSocket
  connection, with `routers.agent_voice.run_turn` monkeypatched to a canned
  `TurnOutcome` (greeting on the bootstrap call, `end_call` decision on the
  first prompt) since no Anthropic API key is configured in this sandbox.
  Confirmed: 3 `Turn` rows persisted (greeting, caller utterance, reply), 1
  `CallOutcome` row, the `Call` row marked `completed`, and the `CallTask`
  marked `completed`. Also confirmed the two `setup`-with-no/unknown-`taskId`
  error paths correctly send `{"type": "end", "handoffData": ...
  "config_error"}` and return.

**What was not verified, and why**: the LLM call itself
(`services.claude_client.generate_conversation_turn`) was not exercised
against a real Claude call — no `ANTHROPIC_API_KEY` is configured on this
machine (see AUDIT.md; `.env` is byte-identical to `.env.example`). Nothing
about a real Twilio call was exercised — no Twilio credentials are
configured, so the exact TwiML attribute names and the exact status strings
Twilio's call-status webhook will send are written to Twilio's documented
shape but not smoke-tested against a live call, the same caveat this repo
already applies to `services/stt/deepgram.py` and the other cloud provider
integrations audited in AUDIT.md.

## 7. Known limitations (carried forward, not fixed in this pass)

- **No authentication on the new endpoints.** `routers/campaigns.py`'s
  contact/campaign/kill-switch/dialer-admin endpoints have no auth check —
  flagged in that file's module docstring. Before this is exposed anywhere
  other than a trusted internal network, it needs one; kill-switch and
  dialer-admin endpoints in particular are the wrong thing to leave open.
- **Concurrency-counter leak on crash.** `ai_calling:active_call_count` is
  incremented on originate and decremented in `mark_call_ended()`, called
  from `agent_voice.py`'s `finally` block. If the process is killed hard
  enough to skip that `finally` (not a graceful shutdown, an actual crash),
  the counter leaks a slot permanently until the process restarts — there is
  no TTL/lease self-healing on this counter. A reconciliation job (recompute
  the counter from `CallTask` rows in `in_call` state on startup, or a
  periodic sweep) would close this; not built in this pass.
  - **Startup reconciliation is now real work, not speculative.** With the
    dialer able to run for real, an ungraceful restart (OOM kill, `kill -9`,
    a bad deploy) leaves the counter wrong on every subsequent boot until
    someone notices dial volume is stuck low. Next concrete step: on
    `_init_rebuild_db()`, set `ai_calling:active_call_count` to
    `count(CallTask.state == "in_call")` rather than leaving it at whatever
    Redis last held.
- **No multi-agent handoff.** The flow graph can produce a `decision ==
  "handoff"` outcome; `agent_voice.py` recognizes it and ends the call
  gracefully with `outcome=handoff_not_implemented` rather than silently
  mishandling it, but no actual handoff (transferring context to a second
  agent) is implemented.
- **No duplicate-insert guard on double-finalize.** `_finalize_call` can in
  principle be called twice for the same call (e.g. an `end_call` decision
  followed immediately by the WebSocket disconnecting before the loop
  exits cleanly) — this would insert two `CallOutcome` rows for one call.
  Not observed in the verification run above (the `ended` flag prevents it
  on the one path tested), but not defended against structurally (no unique
  constraint, no idempotency check in `_finalize_call` itself).
- **Twilio call-status field names/vocabulary unverified live.**
  `handle_call_status_callback` maps Twilio's documented status strings
  (`queued`/`ringing`/`in-progress`/`completed`/`busy`/`failed`/`no-answer`/
  `canceled`) — written to the documented shape, not confirmed against an
  actual webhook payload, since no Twilio credentials are configured here.
- **`dialer_enabled` defaults to `False`.** Deliberate, not a limitation to
  fix — see the comment in `config.py`. An operator has to opt in per
  environment before the background loop places any real calls.

## 8. What Phase 3+ inherits from this

The dialer, admission gate, and live-call wiring are now real, working code
paths — not scaffolding. The remaining gap between this and a production
deploy is the limitations list above (auth, counter reconciliation, handoff,
double-finalize guard) plus the two categories of "written but not live-
tested" work already called out in AUDIT.md (STT/TTS provider credentials,
Twilio credentials) that this pass didn't and couldn't close without those
credentials being supplied.
