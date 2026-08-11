"""Test signal generation.

The calibration signal is an exponential sine sweep ("chirp"). After matched
filtering it compresses to a pulse whose width is set by the bandwidth, which
is what lets us resolve arrival times far more precisely than the sweep's own
duration.

The played test track is a *single long WAV* containing many chirps spaced at
an exact period. We deliberately do not rely on the player looping a short
file: loop boundaries are where re-buffering jitter shows up, and the whole
measurement rests on the spacing between chirps being exact.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

DEFAULT_F_START = 200.0
DEFAULT_F_END = 8000.0
DEFAULT_CHIRP_SECONDS = 0.3
DEFAULT_PERIOD_SECONDS = 1.3
DEFAULT_SAMPLE_RATE = 48000


def exponential_sweep(
    *,
    f_start: float = DEFAULT_F_START,
    f_end: float = DEFAULT_F_END,
    duration: float = DEFAULT_CHIRP_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fade_ratio: float = 0.05,
) -> np.ndarray:
    """Generate a unit-amplitude exponential sine sweep.

    Frequency rises geometrically from ``f_start`` to ``f_end``, which spends
    equal time per octave and so puts useful energy into the low end where
    cheap speakers actually reproduce something.

    ``fade_ratio`` is the fraction of the sweep covered by the raised-cosine
    fade at each end. Without it the discontinuity at the edges rings and
    smears the correlation peak.
    """
    if f_start <= 0 or f_end <= f_start:
        raise ValueError(f"need 0 < f_start < f_end, got {f_start} and {f_end}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate

    ratio = np.log(f_end / f_start)
    # Instantaneous phase of a sweep whose frequency is f_start * (f_end/f_start)^(t/T).
    phase = 2.0 * np.pi * f_start * duration / ratio * (np.exp(t / duration * ratio) - 1.0)
    sweep = np.sin(phase)

    return sweep * _fade_window(n, fade_ratio)


def _fade_window(n: int, fade_ratio: float) -> np.ndarray:
    """Tukey-style window: flat in the middle, raised-cosine at both ends."""
    window = np.ones(n, dtype=np.float64)
    fade = int(round(n * min(max(fade_ratio, 0.0), 0.5)))
    if fade < 1:
        return window
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade, dtype=np.float64) / fade))
    window[:fade] = ramp
    window[n - fade :] = ramp[::-1]
    return window


@dataclass(frozen=True)
class TestSignal:
    """A generated test track plus everything needed to interpret a recording.

    ``period_samples`` and ``chirp_starts`` are in the *track's* sample rate.
    A recording made at a different rate is handled by regenerating the
    reference chirp at the recording's rate — see :mod:`spinalign.dsp.detect`.
    """

    samples: np.ndarray
    sample_rate: int
    chirp_count: int
    period_seconds: float
    chirp_seconds: float
    f_start: float
    f_end: float

    @property
    def period_samples(self) -> float:
        return self.period_seconds * self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def chirp_starts(self) -> np.ndarray:
        """Start offset of each chirp, in samples, as exact floats."""
        return np.arange(self.chirp_count, dtype=np.float64) * self.period_samples

    def reference(self, sample_rate: int | None = None) -> np.ndarray:
        """The bare chirp, regenerated at ``sample_rate`` for matched filtering."""
        return exponential_sweep(
            f_start=self.f_start,
            f_end=self.f_end,
            duration=self.chirp_seconds,
            sample_rate=sample_rate or self.sample_rate,
        )


def build_test_signal(
    *,
    chirp_count: int = 32,
    period_seconds: float = DEFAULT_PERIOD_SECONDS,
    chirp_seconds: float = DEFAULT_CHIRP_SECONDS,
    f_start: float = DEFAULT_F_START,
    f_end: float = DEFAULT_F_END,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> TestSignal:
    """Build the full test track: ``chirp_count`` chirps at an exact period.

    The period must exceed the largest latency difference we expect to measure,
    otherwise an arrival could be attributed to the wrong chirp. The default
    1.3 s comfortably covers the ±500 ms that Music Assistant's ``sync_adjust``
    can express.
    """
    if chirp_count < 1:
        raise ValueError(f"chirp_count must be >= 1, got {chirp_count}")
    if period_seconds <= chirp_seconds:
        raise ValueError(
            f"period ({period_seconds}s) must exceed chirp duration ({chirp_seconds}s)"
        )

    chirp = exponential_sweep(
        f_start=f_start, f_end=f_end, duration=chirp_seconds, sample_rate=sample_rate
    )
    period_samples = period_seconds * sample_rate
    total = int(round(chirp_count * period_samples))
    track = np.zeros(total, dtype=np.float64)

    for index in range(chirp_count):
        # Round each start independently off the exact period so rounding error
        # never accumulates across the track.
        start = int(round(index * period_samples))
        track[start : start + len(chirp)] += chirp * amplitude

    return TestSignal(
        samples=track,
        sample_rate=sample_rate,
        chirp_count=chirp_count,
        period_seconds=period_seconds,
        chirp_seconds=chirp_seconds,
        f_start=f_start,
        f_end=f_end,
    )


def to_wav_bytes(samples: np.ndarray, sample_rate: int, *, channels: int = 2) -> bytes:
    """Encode float samples in [-1, 1] as a 16-bit PCM WAV.

    Defaults to stereo because some players quietly refuse or downmix mono
    sources, and an identical signal in both channels costs nothing here.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1).reshape(-1)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
