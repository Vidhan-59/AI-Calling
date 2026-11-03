# Low-latency voice architecture

How a live call gets from "the caller stopped talking" to "the caller hears
the bot" in under a second, and how the caller can cut the bot off like on
a real call. Two layers, both shared by phone calls (Twilio) and browser
calls (WebRTC).

```
caller audio -> NOISE SUPP. -> SILERO VAD -> BARGE-IN -> latency -> STT -> call logic -> TTS -> AUDIO BUFFER -> caller
                RNNoise        150ms of      controller   monitor    Sarvam   agent: Qwen,   OmniVoice   output transport
                (neural)       real speech                           or local streamed       (cached     (flushed on
                                                                     Whisper  (speculative)  pieces)     barge-in)
```

## Layer 1: full duplex + token streaming

*"Listen, think and speak at the same time."*

| Piece | Where | What it does |
|---|---|---|
| Full duplex transport | Twilio Media Streams (phone), SmallWebRTC (browser) | Audio flows both ways at once; the caller is heard while the bot speaks. A phone call can't use WebRTC (PSTN reaches us only through Twilio), so WebRTC is the browser entry point: `GET /webrtc`. |
| Speculative STT | `services/voice_pipeline/local_stt.py` | Whisper starts decoding 0.12s into the caller's pause instead of after VAD's 0.5s end-of-turn window. If the caller resumes, the guess is discarded. Takes STT from ~550ms after end-of-speech to ~0-250ms. |
| Token streaming | `services/agents/conversation_agent.py` (`on_speech`), `local_ai/llm_engine.py` | The LLM reply is streamed; the first piece goes to TTS after 3 words (`services/pipeline/chunker.py` `SpeechChunker`), later pieces at clause boundaries. Each piece passes the output guardrails before it is spoken. |
| Piecewise TTS | `services/tts/pipecat_service.py` → `tts_service/` (OmniVoice) | Each reply piece is one OmniVoice request; the short first piece uses fewer diffusion steps (`TTS_SHORT_PIECE_NUM_STEP`). Pieces go through the audio cache, so repeated phrases play instantly. OmniVoice doesn't stream audio, so ~1-1.5s per piece on the 8 GB laptop GPU. |
| Late-reply acknowledgement | `conversation_processor._late_filler` (`LATE_FILLER_MS`) | If the real reply hasn't started 0.7s after the caller finished, a cached "जी।" plays at once - never silence. The real words follow as soon as OmniVoice renders them. |
| TTS warm-up | `POST /v1/warm`, `tts_client.warm` | The TTS service parks the model in CPU RAM when idle (small GPUs). The call start and the caller's first syllable wake it (~0.6s), hidden behind STT + LLM time. |
| Think ahead | `conversation_agent.prewarm_next_turn` | While the bot speaks, Qwen pre-processes the next turn's prompt, so only the caller's new words are left when they answer. |
| Tool filler | `_SpeechGate.say_filler` | A turn that needs a tool (roster lookup) says "जी, एक सेकंड।" right away instead of silence through two LLM rounds. |

Scripted scenario calls don't need LLM streaming: every line is pre-synthesized with OmniVoice (`tts_cache.py`), so their TTS cost is already 0ms.

**Where the 8 GB laptop stands (measured, local Whisper, agent call):** transcript ready 37-550ms after end-of-speech, first LLM words 200-770ms later, OmniVoice render of the first piece ~1-1.5s. So the real reply starts ~1.5-2.5s after the caller stops; the cached acknowledgement covers the gap and puts first audio at ~0.71s. On a bigger GPU (or with Sarvam STT, which frees ~2 GB of VRAM) the real reply gets close to the acknowledgement.

## Layer 2: barge-in + Silero VAD + noise suppression

*"Let the user cut in, like a real call. Background noise never triggers a stop."*

| Piece | Where | What it does |
|---|---|---|
| Noise suppression | `audio_frontend.build_noise_filter` | RNNoise on the input transport: VAD, STT and recordings all get the cleaned signal. |
| Silero VAD | `audio_frontend.SpeculativeVADProcessor` | Speech must hold confidence ≥0.7 and volume ≥0.6 for **150ms** before it counts as the caller talking. Also emits the pause/resume signals speculative STT uses. |
| Barge-in controller | `barge_in.BargeInController` | Caller speech confirmed while the bot talks → interruption: TTS drops pending audio, the output buffer is flushed, Twilio gets `clear`. Protected lines (the scripted opening, the goodbye) play out. |
| Instant flush | (nothing to do) | A WebSocket TTS (Sarvam) had to reconnect before passing the interruption on (~600ms of the bot talking over the caller). OmniVoice is plain HTTP per piece: there is no connection to reset, so the flush goes out at once (measured 2-21ms). |
| Turn merging | `conversation_processor` | If the caller resumes mid-sentence after VAD already ended their turn, the half-generated reply is abandoned (LLM stream closed) and both fragments are answered as one turn. |

## Continuous speech with OmniVoice (why replies used to break up)

OmniVoice synthesizes a whole utterance per request, on one GPU worker. Cost
is `fixed + slope x audio`: measured warm on the RTX 5060 laptop, about
0.75s + 0.12 x audio at 10 diffusion steps and 0.7s + 0.7 x audio at 16, and
2-4x slower again when the GPU is power-capped (laptop on battery: clocks 180-870
of 3090 MHz). A streamed reply is several pieces synthesized one after another,
so piece N+1 arrives that long after piece N was ready. When that is longer
than the audio still queued from piece N, the caller hears silence in the
middle of the answer. The old pipeline released fixed-size pieces (3 words,
then every clause) at 16 steps with no notion of the playback buffer, so a short
first piece was routinely followed by a gap of 0.5-3s.

| Piece | Where | What it does |
|---|---|---|
| Playout planner | `services/tts/playout.py` | Models the buffered audio and fits `fixed + slope` per quality level from every real synthesis, so contention or a throttled GPU shows up within a few pieces. |
| Budgeted piece sizing | `SpeechChunker(budget_words=...)`, `PlayoutPlanner.next_budget_words` | Each next piece is as long as its synthesis still lands before the queued audio runs out: short first piece for latency, then it grows. If a gap is unavoidable it is made ONE gap with a long piece, not a string of short ones. |
| Adaptive quality | `OmniVoiceTTSService.run_tts` | Picks the highest step count (16/12/10) whose predicted synthesis fits the buffer. Continuity over the last notch of quality. |
| Consistent loudness | `tts_service/audio.py` (`normalize_loudness`, `fade_edges`) | Pieces used to be peak-normalized separately, so loudness jumped at every piece boundary. Now matched by RMS, with short fades. Needs the TTS service restarted to take effect. |
| Diagnostics | `GET /studio/diagnostics/audio`, `/studio#/diagnostics` | Underruns, gap lengths, synthesis p50/p95, time to first audio, GPU clocks and power-cap state, service queue. Every piece also logs `TTS piece: ... UNDERRUN gap=...ms` when a gap is audible. |

Measured against the real service with an LLM-paced token stream (4 replies x 2 rounds, tools/ probe, same text before and after):
total silence inside a reply fell from about 5.2s to 2.2s per reply on the power-capped GPU, and from about 3.0s to 0.4s once the GPU was back at full clocks.
On battery the GPU cannot keep up whatever the software does: plug the laptop in.

## Measuring it

```
.\venv\Scripts\python.exe tools\latency_smoke_test.py            # agent call
.\venv\Scripts\python.exe tools\latency_smoke_test.py --scenario # + scripted calls
```

Runs the real pipeline in-process behind a fake Twilio socket, with real
synthesized caller speech (cached under `data/smoke_audio`; new phrases need Sarvam TTS credits), and reports: time to first audio per turn (from
VAD end-of-speech, the <1s target, and from the caller's last word), whether
background noise caused a barge-in, and how fast a real barge-in flushes the
bot's audio. Every live call also logs `time to first audio: ...ms` per turn
(`services/voice_pipeline/latency.py`).

Every feature has a switch in `.env` (see `.env.example`) for A/B runs.
