"""Shared harness: render a session in the virtual room and measure it back.

Tests state the truth (each speaker's latency), let the room render it, and
assert the pipeline recovers it. Keeping that round trip in one place means
every test reads as "these conditions, this accuracy".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sim.virtual_room import Round, RoomConfig, VirtualSpeaker, render_recording
from spinalign.calibration.measure import (
    MeasurementAnalysis,
    RoundPlan,
    analyze,
    plan_rounds,
    total_chirps,
)
from spinalign.dsp.detect import Arrival, detect_arrivals
from spinalign.dsp.signals import exponential_sweep


@dataclass(frozen=True)
class SessionResult:
    analysis: MeasurementAnalysis
    arrivals: list[Arrival]
    rounds: list[RoundPlan]
    config: RoomConfig
    speakers: tuple[VirtualSpeaker, ...]

    def relative_error_ms(self, player_id: str, reference_id: str) -> float | None:
        """Measured minus true latency difference against the reference.

        ``None`` when either speaker produced no reading, which the caller is
        expected to treat as a distinct outcome rather than as zero error.
        """
        measured = self.analysis.readings.get(player_id)
        base = self.analysis.readings.get(reference_id)
        if measured is None or base is None:
            return None

        truth = {s.player_id: s.total_latency_ms for s in self.speakers}
        return (measured.latency_ms - base.latency_ms) - (truth[player_id] - truth[reference_id])

    def worst_error_ms(self, reference_id: str) -> float:
        errors = [
            self.relative_error_ms(s.player_id, reference_id)
            for s in self.speakers
            if s.player_id != reference_id
        ]
        present = [abs(e) for e in errors if e is not None]
        return max(present) if present else float("nan")


def run_session(
    speakers: Sequence[VirtualSpeaker],
    *,
    reference_id: str | None = None,
    chirps_per_round: int = 5,
    **room_kwargs,
) -> SessionResult:
    """Render a full calibration session and analyse it end to end."""
    reference = reference_id or speakers[0].player_id
    rounds = plan_rounds(reference, [s.player_id for s in speakers], chirps_per_round=chirps_per_round)
    config = RoomConfig(chirp_count=total_chirps(rounds), **room_kwargs)
    rate = config.mic_sample_rate

    recording = render_recording(speakers, _room_schedule(rounds, config), config)
    reference_chirp = exponential_sweep(
        f_start=config.f_start,
        f_end=config.f_end,
        duration=config.chirp_seconds,
        sample_rate=rate,
    )
    arrivals = detect_arrivals(
        recording,
        reference_chirp,
        period_samples=config.period_seconds * rate,
        sample_rate=rate,
        f_start=config.f_start,
        f_end=config.f_end,
    )

    return SessionResult(
        analysis=analyze(arrivals, rounds, sample_rate=rate),
        arrivals=arrivals,
        rounds=rounds,
        config=config,
        speakers=tuple(speakers),
    )


def _room_schedule(rounds: Sequence[RoundPlan], config: RoomConfig) -> list[Round]:
    """Translate the chirp-indexed plan into the room's seconds-based schedule."""
    return [
        Round(
            plan.first_chirp * config.period_seconds,
            (plan.last_chirp + 1) * config.period_seconds,
            frozenset({plan.player_id}),
        )
        for plan in rounds
    ]


def three_speakers(**overrides) -> list[VirtualSpeaker]:
    """The canonical mixed setup: an ESP32, an AVR and a Bluetooth speaker.

    Latencies and distances are the ones the whole project exists to fix.
    """
    defaults = {
        "a": dict(name="ESP32", hardware_latency_ms=20.0, distance_m=2.0),
        "b": dict(name="AVR", hardware_latency_ms=80.0, distance_m=4.0),
        "c": dict(name="Bluetooth", hardware_latency_ms=220.0, distance_m=3.0),
    }
    return [
        VirtualSpeaker(player_id=key, **{**values, **overrides.get(key, {})})
        for key, values in defaults.items()
    ]
