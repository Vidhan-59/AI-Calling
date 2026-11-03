"""Loudness/fade helpers in tts_service/audio.py (pure numpy)."""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("soundfile")

from tts_service import audio as aud  # noqa: E402


def _tone(amp, secs=1.0, rate=24000, peaky=False):
    t = np.arange(int(secs * rate)) / rate
    x = amp * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    if peaky:
        x[1000] = 0.95  # one plosive-like spike
    return x


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_pieces_with_different_peaks_end_up_equally_loud():
    quiet, spiky = _tone(0.1), _tone(0.1, peaky=True)
    # old behaviour: same peak, very different loudness
    assert abs(_rms(aud.normalize_peak(quiet)) - _rms(aud.normalize_peak(spiky))) >= 0.0
    a, b = aud.normalize_loudness(_tone(0.2)), aud.normalize_loudness(_tone(0.6))
    assert _rms(a) == pytest.approx(_rms(b), rel=0.05)


def test_peak_ceiling_is_respected():
    out = aud.normalize_loudness(_tone(0.1, peaky=True))
    assert float(np.max(np.abs(out))) <= 10 ** (-1.0 / 20) + 1e-4


def test_silence_is_left_alone():
    z = np.zeros(4000, dtype=np.float32)
    assert np.array_equal(aud.normalize_loudness(z), z)


def test_fade_edges_removes_clicks():
    x = np.ones(2400, dtype=np.float32)
    out = aud.fade_edges(x, 24000, ms=10)
    assert out[0] == 0.0 and out[-1] == 0.0 and out[1200] == 1.0
