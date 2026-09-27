"""A synthetic room that renders what the microphone would have heard.

This is the ground truth the whole test suite is built on. Speakers are given
*known* electronic latencies, distances and reflections, the room renders a
recording, and the detector has to recover those numbers. Because the truth is
exact, accuracy claims become assertions rather than opinions.

Chirps are placed at fractional sample positions, so sub-sample interpolation
in the detector is genuinely exercised instead of being handed integers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from speaker_sync.dsp.signals import (
    DEFAULT_CHIRP_SECONDS,
    DEFAULT_F_END,
    DEFAULT_F_START,
    DEFAULT_PERIOD_SECONDS,
    exponential_sweep,
)

SPEED_OF_SOUND_M_S = 343.0
"""Dry air at roughly 20 °C — about 2.915 ms per metre."""


def distance_to_ms(metres: float) -> float:
    return metres / SPEED_OF_SOUND_M_S * 1000.0


@dataclass(frozen=True)
class VirtualSpeaker:
    """A speaker with a known, deliberately chosen latency budget."""

    player_id: str
    name: str
    hardware_latency_ms: float
    """Electronic path: DAC, DSP, Bluetooth buffering."""

    distance_m: float = 2.0
    gain: float = 1.0
    sync_adjust_ms: int = 0
    """Correction currently in force, as a real server would already have."""

    reflections: tuple[tuple[float, float], ...] = ()
    """``(delay_ms, gain)`` pairs relative to the direct sound."""

    sign: int = 1
    """How this speaker interprets ``sync_adjust``; ``-1`` models an inverted
    server convention, which the validation step must detect."""

    glitches: tuple[tuple[int, float], ...] = ()
    """``(chirp_index, extra_ms)`` — a one-off timing hiccup on that chirp.

    Models a player resynchronising its clock mid-session, or a buffer
    stumbling. The median across a round should absorb it, and the analysis
    should say it happened."""

    def glitch_at(self, chirp_index: int) -> float:
        return sum(extra for index, extra in self.glitches if index == chirp_index)

    @property
    def acoustic_ms(self) -> float:
        return distance_to_ms(self.distance_m)

    @property
    def total_latency_ms(self) -> float:
        """Ground truth: when this speaker's sound reaches the microphone."""
        return self.hardware_latency_ms + self.acoustic_ms + self.sign * self.sync_adjust_ms

    def with_adjust(self, sync_adjust_ms: int) -> "VirtualSpeaker":
        return VirtualSpeaker(
            player_id=self.player_id,
            name=self.name,
            hardware_latency_ms=self.hardware_latency_ms,
            distance_m=self.distance_m,
            gain=self.gain,
            sync_adjust_ms=sync_adjust_ms,
            reflections=self.reflections,
            sign=self.sign,
            glitches=self.glitches,
        )


@dataclass(frozen=True)
class Round:
    """A stretch of the session during which only some speakers are audible."""

    start_seconds: float
    end_seconds: float
    audible: frozenset[str]

    def covers(self, moment: float) -> bool:
        return self.start_seconds <= moment < self.end_seconds


@dataclass
class RoomConfig:
    mic_sample_rate: int = 48000
    period_seconds: float = DEFAULT_PERIOD_SECONDS
    chirp_seconds: float = DEFAULT_CHIRP_SECONDS
    f_start: float = DEFAULT_F_START
    f_end: float = DEFAULT_F_END
    chirp_count: int = 32
    snr_db: float = 30.0
    lead_in_seconds: float = 0.4
    tail_seconds: float = 0.6
    clock_ppm: float = 0.0
    """Microphone clock error. Stretches the chirp grid in the recording."""

    clip: bool = False
    seed: int = 0
    amplitude: float = 0.4
    reference_gain: float = field(default=1.0)


def _fractional_shift(chirp: np.ndarray, fraction: float) -> np.ndarray:
    """Delay a short buffer by a sub-sample amount via a spectral phase ramp.

    Exact for a band-limited signal, unlike linear interpolation which would
    quietly low-pass the chirp and blunt the very peak we are measuring.
    """
    if fraction == 0.0:
        return chirp
    pad = 64
    padded = np.concatenate([np.zeros(pad), chirp, np.zeros(pad)])
    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(len(padded), d=1.0)
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * fraction), len(padded))[
        pad : pad + len(chirp)
    ]


def render_recording(
    speakers: Sequence[VirtualSpeaker],
    rounds: Sequence[Round],
    config: RoomConfig | None = None,
) -> np.ndarray:
    """Render the microphone signal for a whole calibration session.

    Each chirp of the track is emitted by every speaker that is audible at that
    moment, delayed by that speaker's total latency, attenuated, and joined by
    its reflections. Noise is added last, at the requested SNR relative to the
    direct sound.
    """
    cfg = config or RoomConfig()
    rate = cfg.mic_sample_rate
    rng = np.random.default_rng(cfg.seed)

    chirp = exponential_sweep(
        f_start=cfg.f_start,
        f_end=cfg.f_end,
        duration=cfg.chirp_seconds,
        sample_rate=rate,
    )

    # A microphone clock that is off by some ppm stretches the observed grid;
    # the detector is told the nominal period and must cope.
    grid_step = cfg.period_seconds * rate * (1.0 + cfg.clock_ppm * 1e-6)
    lead_in = cfg.lead_in_seconds * rate

    max_latency_ms = max(
        (
            s.total_latency_ms
            + max((r[0] for r in s.reflections), default=0.0)
            + max((g[1] for g in s.glitches), default=0.0)
            for s in speakers
        ),
        default=0.0,
    )
    total = int(
        round(
            lead_in
            + cfg.chirp_count * grid_step
            + max_latency_ms * rate / 1000.0
            + cfg.tail_seconds * rate
        )
    )
    recording = np.zeros(total + len(chirp) + 128, dtype=np.float64)

    for index in range(cfg.chirp_count):
        # Playback time of this chirp in the stream's own timeline, which is
        # what the mute schedule is expressed against.
        moment = index * cfg.period_seconds
        for speaker in speakers:
            if not any(r.covers(moment) and speaker.player_id in r.audible for r in rounds):
                continue

            glitch_ms = speaker.glitch_at(index)
            arrivals = [(0.0, 1.0), *speaker.reflections]
            for extra_ms, extra_gain in arrivals:
                position = (
                    lead_in
                    + index * grid_step
                    + (speaker.total_latency_ms + glitch_ms + extra_ms) * rate / 1000.0
                )
                whole = int(np.floor(position))
                shifted = _fractional_shift(chirp, position - whole)
                gain = cfg.amplitude * speaker.gain * extra_gain
                recording[whole : whole + len(shifted)] += shifted * gain

    signal_rms = float(np.sqrt(np.mean(recording**2)))
    if signal_rms > 0.0 and np.isfinite(cfg.snr_db):
        noise_rms = signal_rms / (10.0 ** (cfg.snr_db / 20.0))
        recording += rng.normal(0.0, noise_rms, size=len(recording))

    if cfg.clip:
        recording = np.clip(recording, -1.0, 1.0)

    return recording


def solo_rounds(
    reference_id: str,
    player_ids: Sequence[str],
    *,
    seconds_per_round: float = 6.5,
    start_seconds: float = 0.0,
) -> list[Round]:
    """Build the standard schedule: every speaker alone, in turn.

    Exactly one speaker is audible at a time. The common time base comes from
    the chirp grid in the continuous recording, not from a simultaneously
    playing reference, so each round is a clean single-arrival measurement and
    one speaker's reflections can never be mistaken for another's direct sound.

    The reference is measured first and again last. Those two readings bracket
    the session: their difference is the microphone's clock drift, which is
    then divided out of everything measured in between.
    """
    rounds: list[Round] = []
    cursor = start_seconds

    def add(player_id: str) -> None:
        nonlocal cursor
        rounds.append(Round(cursor, cursor + seconds_per_round, frozenset({player_id})))
        cursor += seconds_per_round

    add(reference_id)
    for player_id in player_ids:
        if player_id != reference_id:
            add(player_id)
    add(reference_id)

    return rounds
