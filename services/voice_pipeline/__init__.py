"""Self-hosted live-call audio pipeline (Twilio Media Streams -> Silero VAD ->
STT (Sarvam or faster-whisper) -> Qwen/Ollama -> OmniVoice TTS), built on Pipecat. Replaces
Twilio ConversationRelay on the rebuild's live-call path (routers/agent_voice.py)
- see PHASE2_ARCHITECTURE_PROPOSAL.md's "revisit when audio-level control is
needed" note. Track A (routers/voice.py) and Track B (routers/voice_scenario.py)
are untouched and still use ConversationRelay.
"""
