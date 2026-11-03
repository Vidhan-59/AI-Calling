"""Runs local Whisper in a dedicated process, shared by every live call.

Why a process and not a thread: faster-whisper does real Python/numpy work
around the GPU kernels (mel features over the 30s-padded window, the token
loop), and in the server process that work fights the event loop - Pipecat
frames, Silero VAD and Twilio's 50 frames/s per call - for the GIL. Measured
on the RTX 5060 laptop: one busy Python thread took feature extraction from
2ms to ~150ms and a decode from ~540ms to ~740ms; in a simulated call decodes
ran 1.0-1.5s. A separate process has its own GIL, so decode time stays flat
no matter how many calls the server is handling.

The worker owns the CUDA models (the server process loads none, so there's
no duplicate VRAM), serves requests one at a time - one GPU gains nothing
from overlapping decodes - and is respawned automatically if it dies.
"""

import itertools
import logging
import multiprocessing as mp
import threading
from concurrent.futures import Future

logger = logging.getLogger("voice_pipeline.stt_worker")


def _boost_this_process() -> None:
    """Windows parks a process that has sat idle waiting on a pipe on an
    efficiency core with power throttling (EcoQoS); the first decode after
    each pause in a call then ran ~400ms slower than the next one. Opt this
    process (only) out of throttling and raise its priority - the same thing
    run-all.ps1 does for the server and Ollama. No-op elsewhere."""
    import os
    import sys

    if sys.platform != "win32" or os.getenv("STT_WORKER_BOOST", "1") == "0":
        return
    import ctypes
    from ctypes import wintypes

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(handle, 0x00000080)  # HIGH_PRIORITY_CLASS
    # ProcessPowerThrottling (4): control EXECUTION_SPEED (1), state off (0).
    state = _PowerThrottlingState(1, 0x1, 0x0)
    kernel32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))


def _worker_main(conn) -> None:
    """Child process entry point. Messages: ("load", id, specs) and
    ("transcribe", id, source, compute_type, pcm16_bytes, options)."""
    import numpy as np

    try:
        _boost_this_process()
    except Exception:  # noqa: BLE001 - a missing boost is slower, not broken
        pass

    from services.voice_pipeline import local_stt

    while True:
        try:
            if not conn.poll(1.0):
                local_stt.offload_if_idle()  # between calls: hand the VRAM back
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            return  # server went away
        op, rid = msg[0], msg[1]
        try:
            if op == "load":
                local_stt.preload_models(msg[2])
                result = None
            elif op == "warm":
                # Skip if a real request is already queued - the warm-up
                # exists to make that request fast, never to delay it.
                if not conn.poll():
                    local_stt.warm_gpu(msg[2], msg[3])
                result = None
            elif op == "transcribe":
                _, _, source, compute_type, pcm, options = msg
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                result = local_stt.transcribe_with(local_stt.get_model(source, compute_type), audio, **options)
            else:
                raise ValueError(f"unknown op {op!r}")
            conn.send((rid, True, result))
        except Exception as e:  # noqa: BLE001 - report it, keep serving
            conn.send((rid, False, f"{type(e).__name__}: {e}"))


class STTWorker:
    def __init__(self) -> None:
        self._lock = threading.Lock()  # guards start/stop and sends
        self._proc = None
        self._conn = None
        self._pending: dict[int, Future] = {}
        self._ids = itertools.count(1)
        self._specs: list[tuple[str, str]] = []  # reloaded into a respawned worker

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        if self._proc is not None:
            logger.warning("STT worker exited (code %s) - respawning", self._proc.exitcode)
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker_main, args=(child,), name="stt-worker", daemon=True)
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent
        threading.Thread(target=self._read_results, args=(parent,), name="stt-worker-reader", daemon=True).start()
        if self._specs:
            self._send_locked("load", list(self._specs))

    def _send_locked(self, op: str, *args) -> Future:
        rid = next(self._ids)
        fut: Future = Future()
        self._pending[rid] = fut
        try:
            self._conn.send((op, rid, *args))
        except (OSError, ValueError) as e:
            self._pending.pop(rid, None)
            fut.set_exception(RuntimeError(f"STT worker unreachable: {e}"))
        return fut

    def _submit(self, op: str, *args) -> Future:
        with self._lock:
            self._ensure_started()
            return self._send_locked(op, *args)

    def _read_results(self, conn) -> None:
        while True:
            try:
                rid, ok, payload = conn.recv()
            except (EOFError, OSError):
                break
            fut = self._pending.pop(rid, None)
            if fut is None or fut.done():
                continue
            if ok:
                fut.set_result(payload)
            else:
                fut.set_exception(RuntimeError(f"STT worker: {payload}"))
        with self._lock:
            if self._conn is not conn:
                return  # an older worker; the current one has its own reader
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("STT worker exited"))

    def load(self, specs: list[tuple[str, str]], timeout: float = 600.0) -> None:
        """Blocking: load + warm the models in the worker (call off the event loop)."""
        with self._lock:
            for spec in specs:
                if spec not in self._specs:
                    self._specs.append(spec)
        self._submit("load", list(specs)).result(timeout=timeout)

    def transcribe(self, source: str, compute_type: str, pcm16: bytes, **options) -> Future:
        """-> Future[(text, decode_ms)]. Await with asyncio.wrap_future."""
        return self._submit("transcribe", source, compute_type, pcm16, options)

    def _detach(self):
        with self._lock:
            proc, self._proc = self._proc, None
            conn, self._conn = self._conn, None
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("STT worker restarted"))
        return proc, conn

    def warm(self, source: str, compute_type: str) -> None:
        """Fire-and-forget: wake the GPU from idle clocks (see local_stt.warm_gpu)."""
        self._submit("warm", source, compute_type)

    def restart(self) -> None:
        """Kill a stuck worker; the next request starts a fresh one."""
        proc, conn = self._detach()
        if proc is not None:
            proc.kill()
            proc.join(5)
        if conn is not None:
            conn.close()

    def close(self) -> None:
        proc, conn = self._detach()
        if conn is not None:
            conn.close()  # the worker sees EOF and exits
        if proc is not None:
            proc.join(5)
            if proc.is_alive():
                proc.kill()


worker = STTWorker()
