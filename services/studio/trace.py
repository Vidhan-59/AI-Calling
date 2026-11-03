"""Live call trace: every observable step of a call (session, audio in, STT,
LLM, retrieval, TTS, audio out, network, errors) is persisted AND fanned out
to any browser watching the call over its /studio/calls/{id}/events socket.

Only things that actually happened are recorded: provider timings, the text
that was transcribed / generated / spoken, config metadata. No reasoning is
invented - an operation the provider doesn't expose is simply not traced.
"""

import asyncio
import logging
import time
from collections import defaultdict

from services.studio import store

logger = logging.getLogger("studio.trace")

CATEGORIES = ("session", "audio_input", "speech_recognition", "llm", "prompt", "knowledge", "response",
              "tts", "audio_output", "network", "error")

_subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)


def subscribe(call_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _subscribers[call_id].add(q)
    return q


def unsubscribe(call_id: str, q: asyncio.Queue) -> None:
    _subscribers[call_id].discard(q)
    if not _subscribers[call_id]:
        _subscribers.pop(call_id, None)


def publish(call_id: str, kind: str, payload: dict) -> None:
    for q in list(_subscribers.get(call_id, ())):
        try:
            q.put_nowait({"type": kind, **payload})
        except asyncio.QueueFull:  # a stuck viewer must never slow the call
            logger.warning("dropping %s event for slow viewer of %s", kind, call_id)


class CallTrace:
    """One call's trace/transcript writer. Cheap enough to call from the
    audio path: DB writes are tiny and SQLite is in-process."""

    def __init__(self, call_id: str):
        self.call_id = call_id
        self._t0 = time.monotonic()

    def event(self, category: str, name: str, detail: str = "", *, status: str = "ok",
              duration_ms: float | None = None, data: dict | None = None) -> None:
        try:
            row = store.add_trace(self.call_id, category, name, status, detail, duration_ms, data)
        except Exception:  # noqa: BLE001 - tracing must never break a call
            logger.exception("trace write failed")
            return
        publish(self.call_id, "trace", {"event": row})

    def message(self, speaker: str, text: str, metadata: dict | None = None) -> None:
        try:
            row = store.add_message(self.call_id, speaker, text, metadata)
        except Exception:  # noqa: BLE001
            logger.exception("message write failed")
            return
        publish(self.call_id, "message", {"message": row})

    def state(self, state: str) -> None:
        publish(self.call_id, "state", {"state": state, "t": round(time.monotonic() - self._t0, 2)})
