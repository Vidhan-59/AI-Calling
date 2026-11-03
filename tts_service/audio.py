"""Audio helpers: decode uploads, resample 24 kHz model output to the 8 kHz
telephony rate, trim silence, and encode PCM16/WAV."""

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from tts_service import settings


class AudioError(ValueError):
    """Bad user-supplied audio (client error, not a server fault)."""


def decode_upload(data: bytes) -> tuple[np.ndarray, int]:
    """Decode wav/flac/ogg (anything libsndfile reads) to mono float32."""
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise AudioError("audio file too large")
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise AudioError(
            "could not decode audio - upload WAV/FLAC/OGG (the Voice Studio recorder converts to WAV for you)"
        ) from e
    return audio.mean(axis=1), rate


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    g = gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def trim_silence(audio: np.ndarray, rate: int, threshold_db: float = -45.0, pad_secs: float = 0.08) -> np.ndarray:
    """Cut leading/trailing near-silence: less dead air on a phone call, and a
    clean reference clip for cloning."""
    if audio.size == 0:
        return audio
    idx = np.flatnonzero(np.abs(audio) > 10 ** (threshold_db / 20))
    if idx.size == 0:
        return audio
    pad = int(pad_secs * rate)
    return audio[max(0, idx[0] - pad): idx[-1] + pad + 1]


def normalize_peak(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    return (audio * (10 ** (target_db / 20) / peak)).astype(np.float32)


def normalize_loudness(audio: np.ndarray, target_rms_db: float = -20.0, peak_ceiling_db: float = -1.0,
                       max_gain_db: float = 12.0) -> np.ndarray:
    """Level a synthesized piece to a fixed speech loudness (RMS of the active
    samples), capped so the peak stays under the ceiling.

    A streamed reply is several separately synthesized pieces played back to
    back. Peak-normalizing each one (as this service used to) gives every
    piece the same PEAK but a different loudness - a piece with one plosive
    came out quieter than the next - which sounds like the voice jumping
    around at every piece boundary. Matching RMS keeps them consistent."""
    if audio.size == 0:
        return audio
    active = audio[np.abs(audio) > 10 ** (-45.0 / 20)]
    if active.size < 64:
        return audio
    rms = float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-6 or peak < 1e-6:
        return audio
    gain = 10 ** (target_rms_db / 20) / rms
    gain = min(gain, 10 ** (peak_ceiling_db / 20) / peak, 10 ** (max_gain_db / 20))
    return (audio * gain).astype(np.float32)


def fade_edges(audio: np.ndarray, rate: int, ms: float = 12.0) -> np.ndarray:
    """Short fade in/out so a piece that starts or ends mid-waveform (silence
    was trimmed to the signal) can't click where it joins its neighbour."""
    n = min(int(rate * ms / 1000), audio.size // 2)
    if n < 2:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def to_wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def duration_secs(audio: np.ndarray, rate: int) -> float:
    return len(audio) / float(rate)
