"""Driving a calibration session end to end.

One acoustic pass looks like this: put every speaker in one group, start the
microphone, start the track, and walk the mute state forward one round at a
time so exactly one speaker is audible during each stretch of chirps. Then
detect, analyse, solve, write the corrections back, and measure again to prove
the result rather than assert it.

The verification pass is not decoration. It is what catches a wrong sign
convention, a player that ignored the write, and a room that moved between
passes — all failures that would otherwise leave the user with a confident
report and worse sound than before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from spinalign.calibration.measure import (
    MeasurementAnalysis,
    RoundPlan,
    analyze,
    plan_rounds,
    total_chirps,
)
from spinalign.calibration.solver import (
    CalibrationSolution,
    PlayerMeasurement,
    solve,
)
from spinalign.dsp.detect import detect_arrivals
from spinalign.dsp.signals import (
    DEFAULT_CHIRP_SECONDS,
    DEFAULT_F_END,
    DEFAULT_F_START,
    DEFAULT_PERIOD_SECONDS,
    DEFAULT_SAMPLE_RATE,
    TestSignal,
    build_test_signal,
)
from spinalign.ma.backend import PlayerInfo, SpeakerBackend

Sleeper = Callable[[float], Awaitable[None]]

ProgressCallback = Callable[[dict], None]
"""Notified as the session advances, so the UI can show which speaker is
playing rather than a spinner for half a minute."""

SignalUrl = "str | Callable[[TestSignal], str]"
"""Where the media server should fetch the test track.

Accepts a callable because the track's length depends on how many speakers are
being calibrated: the URL can then carry the parameters and stay stateless,
rather than the web layer having to stash the current signal somewhere the
audio route can find it."""


def _resolve_url(signal_url, signal: TestSignal) -> str:
    return signal_url(signal) if callable(signal_url) else signal_url


def _report(progress: ProgressCallback | None, **event) -> None:
    if progress is not None:
        progress(event)


class Recorder(Protocol):
    """Whatever is holding the microphone — a browser, or the simulator."""

    async def start(self, signal: TestSignal) -> None: ...

    async def stop(self) -> tuple[np.ndarray, int]:
        """Return the recording and its sample rate."""


@dataclass(frozen=True)
class SessionConfig:
    chirps_per_round: int = 5
    guard_chirps: int = 2
    period_seconds: float = DEFAULT_PERIOD_SECONDS
    chirp_seconds: float = DEFAULT_CHIRP_SECONDS
    f_start: float = DEFAULT_F_START
    f_end: float = DEFAULT_F_END
    track_sample_rate: int = DEFAULT_SAMPLE_RATE
    settle_seconds: float = 0.35
    """Pause after a mute change before the round's chirps start counting."""

    def build_signal(self, chirp_count: int) -> TestSignal:
        return build_test_signal(
            chirp_count=chirp_count,
            period_seconds=self.period_seconds,
            chirp_seconds=self.chirp_seconds,
            f_start=self.f_start,
            f_end=self.f_end,
            sample_rate=self.track_sample_rate,
        )


@dataclass(frozen=True)
class MeasurementPass:
    analysis: MeasurementAnalysis
    rounds: tuple[RoundPlan, ...]
    signal: TestSignal
    sample_rate: int

    @property
    def latencies_ms(self) -> dict[str, float]:
        return {pid: r.latency_ms for pid, r in self.analysis.readings.items()}

    def relative_spread_ms(self) -> float:
        values = list(self.latencies_ms.values())
        return max(values) - min(values) if len(values) >= 2 else 0.0


@dataclass(frozen=True)
class CalibrationReport:
    before: MeasurementPass
    solution: CalibrationSolution
    after: MeasurementPass | None = None
    applied: tuple[tuple[str, int], ...] = ()
    sign: int = 1
    problems: tuple[str, ...] = field(default=())

    @property
    def improved(self) -> bool:
        if self.after is None:
            return False
        return self.after.relative_spread_ms() < self.before.relative_spread_ms()

    @property
    def spread_before_ms(self) -> float:
        return self.before.relative_spread_ms()

    @property
    def spread_after_ms(self) -> float | None:
        return self.after.relative_spread_ms() if self.after else None


async def measure_once(
    backend: SpeakerBackend,
    recorder: Recorder,
    players: Sequence[PlayerInfo],
    *,
    reference_id: str | None = None,
    config: SessionConfig | None = None,
    signal_url="",
    sleep: Sleeper = asyncio.sleep,
    progress: ProgressCallback | None = None,
) -> MeasurementPass:
    """Run one acoustic pass and return a latency per speaker."""
    cfg = config or SessionConfig()
    if len(players) < 2:
        raise ValueError("need at least two players to calibrate")

    player_ids = [p.player_id for p in players]
    reference = reference_id or player_ids[0]
    rounds = plan_rounds(reference, player_ids, chirps_per_round=cfg.chirps_per_round)
    signal = cfg.build_signal(total_chirps(rounds))

    leader = player_ids[0]
    await backend.set_group(leader, player_ids)

    # Silence everyone before the track starts so the opening round is clean.
    for player_id in player_ids:
        await backend.set_muted(player_id, True)

    await recorder.start(signal)
    audible: str | None = None

    try:
        await backend.play_url(leader, _resolve_url(signal_url, signal))
        for index, plan in enumerate(rounds):
            if plan.player_id != audible:
                if audible is not None:
                    await backend.set_muted(audible, True)
                await backend.set_muted(plan.player_id, False)
                audible = plan.player_id

            _report(
                progress,
                stage="round",
                round=index + 1,
                rounds=len(rounds),
                player_id=plan.player_id,
            )

            # The first round also absorbs the stream's own start-up delay.
            round_seconds = cfg.chirps_per_round * cfg.period_seconds
            if index == 0:
                round_seconds += cfg.settle_seconds
            await sleep(round_seconds)
    finally:
        await backend.stop(leader)
        recording, sample_rate = await recorder.stop()
        for player_id in player_ids:
            await backend.set_muted(player_id, False)

    _report(progress, stage="analyzing")

    arrivals = detect_arrivals(
        recording,
        signal.reference(sample_rate),
        period_samples=cfg.period_seconds * sample_rate,
        sample_rate=sample_rate,
        f_start=cfg.f_start,
        f_end=cfg.f_end,
    )
    analysis = analyze(arrivals, rounds, sample_rate=sample_rate, guard_chirps=cfg.guard_chirps)

    return MeasurementPass(
        analysis=analysis,
        rounds=tuple(rounds),
        signal=signal,
        sample_rate=sample_rate,
    )


def build_measurements(
    pass_: MeasurementPass, players: Sequence[PlayerInfo]
) -> list[PlayerMeasurement]:
    """Join acoustic readings with the corrections that were in force."""
    by_id = {p.player_id: p for p in players}
    return [
        PlayerMeasurement(
            player_id=player_id,
            name=by_id[player_id].name if player_id in by_id else player_id,
            measured_ms=reading.latency_ms,
            current_adjust_ms=by_id[player_id].sync_adjust_ms if player_id in by_id else 0,
            spread_ms=reading.spread_ms,
        )
        for player_id, reading in pass_.analysis.readings.items()
    ]


async def apply_solution(
    backend: SpeakerBackend,
    solution: CalibrationSolution,
) -> tuple[tuple[str, int], ...]:
    """Write every changed correction in one batch.

    Music Assistant marks ``sync_adjust`` as requiring a reload, so each write
    interrupts the player. Batching them at the end of the session keeps that
    to a single disruption instead of one per measurement.
    """
    applied: list[tuple[str, int]] = []
    for correction in solution.changed():
        await backend.set_sync_adjust(correction.player_id, correction.target_adjust_ms)
        applied.append((correction.player_id, correction.target_adjust_ms))
    return tuple(applied)


async def calibrate(
    backend: SpeakerBackend,
    recorder: Recorder,
    *,
    reference_id: str | None = None,
    config: SessionConfig | None = None,
    signal_url="",
    sign: int = 1,
    verify: bool = True,
    sleep: Sleeper = asyncio.sleep,
    progress: ProgressCallback | None = None,
) -> CalibrationReport:
    """Measure, correct, and then prove the correction worked."""
    cfg = config or SessionConfig()
    players = [p for p in await backend.list_players() if p.is_calibratable]
    if len(players) < 2:
        raise ValueError("need at least two available Sendspin players to calibrate")

    _report(progress, stage="pass", which="before", players=len(players))
    before = await measure_once(
        backend,
        recorder,
        players,
        reference_id=reference_id,
        config=cfg,
        signal_url=signal_url,
        sleep=sleep,
        progress=progress,
    )
    problems = list(before.analysis.problems)
    solution = solve(build_measurements(before, players), sign=sign)

    _report(progress, stage="applying", writes=len(solution.changed()))
    applied = await apply_solution(backend, solution)

    after: MeasurementPass | None = None
    if verify and applied:
        _report(progress, stage="pass", which="after", players=len(players))
        refreshed = [p for p in await backend.list_players() if p.is_calibratable]
        after = await measure_once(
            backend,
            recorder,
            refreshed,
            reference_id=reference_id,
            config=cfg,
            signal_url=signal_url,
            sleep=sleep,
            progress=progress,
        )
        problems.extend(after.analysis.problems)
        if after.relative_spread_ms() > before.relative_spread_ms():
            problems.append(
                "alignment got worse after applying — the sync_adjust sign may be "
                "inverted on this server, or a player ignored the write"
            )

    return CalibrationReport(
        before=before,
        solution=solution,
        after=after,
        applied=applied,
        sign=sign,
        problems=tuple(problems),
    )
