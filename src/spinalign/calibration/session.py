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
    """Chirps each speaker gets, before the guard is deducted.

    This is the session's main trade-off. The default leaves three readings
    per speaker: enough for the median to shrug off a single bad one, and a
    ~26 s pass for three speakers. Raising it buys margin in two places at
    once — the median gets harder to move, and the outlier check in
    :mod:`spinalign.calibration.measure` gets more evidence to notice a glitch
    with — at a directly proportional cost in session length, doubled because
    the run is measured and then verified.
    """

    guard_chirps: int = 2
    """Chirps discarded at the start of each round.

    They absorb the delay between issuing a mute and it taking effect, which
    is the only place network latency touches the measurement. At the default
    settings this leaves about 2.25 seconds of slack.
    """
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
    await confirm_group(backend, leader, player_ids, sleep=sleep)

    # Open the first round's speaker *before* the track starts, not after.
    #
    # Unmuting it afterwards races the stream, and losing that race is not
    # worth a reading — it is worth the whole measurement. The analysis has no
    # other way to find the track's origin than to treat the first chirp it
    # hears as chirp zero, so opening chirps lost to a late unmute slide every
    # round onto the wrong speaker. The result still looks like a result:
    # speakers "measure" each other's arrival times and come out suspiciously
    # aligned.
    # Both directions are checked, and both matter: a speaker that cannot be
    # muted plays through every other speaker's round, and one that cannot be
    # unmuted never sounds at all.
    for player_id in player_ids:
        await backend.set_muted(player_id, True)
    await confirm_mute(backend, dict.fromkeys(player_ids, True), sleep=sleep)

    opening = rounds[0].player_id
    await backend.set_muted(opening, False)
    await confirm_mute(backend, {opening: False}, sleep=sleep)

    await recorder.start(signal)
    audible: str | None = opening

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


GROUP_CONFIRM_TIMEOUT_SECONDS = 4.0
GROUP_POLL_SECONDS = 0.5
MUTE_CONFIRM_TIMEOUT_SECONDS = 4.0


async def confirm_mute(
    backend: SpeakerBackend,
    wanted: dict[str, bool],
    *,
    sleep: Sleeper = asyncio.sleep,
    timeout: float = MUTE_CONFIRM_TIMEOUT_SECONDS,
) -> None:
    """Check the speakers really went quiet before playing anything.

    Muting is what makes a round a solo, and Music Assistant has a per-player
    setting for how — or whether — it can mute at all. A speaker that ignores
    the command plays straight through everyone else's rounds, and the damage
    is not a missing reading: every round then measures the speaker that would
    not shut up, so the speakers come out looking perfectly aligned.

    As with grouping, silence about the state is not a failure. A provider that
    never reports its mute state cannot be checked, and refusing to measure on
    that basis would ground a working setup.
    """
    deadline = timeout
    while True:
        players = {p.player_id: p for p in await backend.list_players()}
        stuck = [
            player_id
            for player_id, muted in wanted.items()
            if player_id in players
            and players[player_id].muted is not None
            and players[player_id].muted != muted
        ]
        if not stuck:
            return
        if deadline <= 0:
            names = ", ".join(f"«{players[pid].name}»" for pid in stuck)
            raise RuntimeError(
                f"эти колонки не отреагировали на команду заглушить: {names}. "
                "Замер требует, чтобы в каждом круге звучала ровно одна колонка, "
                "иначе все круги измерят одну и ту же. Проверьте в Music Assistant "
                "настройку mute_control у этих колонок."
            )

        await sleep(GROUP_POLL_SECONDS)
        deadline -= GROUP_POLL_SECONDS


async def confirm_group(
    backend: SpeakerBackend,
    leader: str,
    player_ids: Sequence[str],
    *,
    sleep: Sleeper = asyncio.sleep,
    timeout: float = GROUP_CONFIRM_TIMEOUT_SECONDS,
) -> None:
    """Check the speakers really joined the group before playing anything.

    The whole method rests on one stream reaching every speaker at once: that
    common stream is what gives the rounds a shared time base. A speaker that
    silently failed to join is not a degraded measurement, it is no measurement
    — it renders as silence, and the only symptom is "no usable chirps", which
    reads like a microphone problem and sends the user hunting in the room.

    Missing information is not treated as failure. A provider that never
    populates ``group_members`` would otherwise ground a setup that works, so
    the check needs positive evidence of exclusion before it refuses.
    """
    deadline = timeout
    while True:
        players = {p.player_id: p for p in await backend.list_players()}
        grouped = set(players[leader].group_members) if leader in players else set()
        if not grouped:
            return  # nothing reported: no evidence either way

        missing = [pid for pid in player_ids if pid != leader and pid not in grouped]
        if not missing:
            return
        if deadline <= 0:
            raise RuntimeError(_group_failure_message(players, leader, missing))

        # Grouping is a command, not a transaction; the state follows it.
        await sleep(GROUP_POLL_SECONDS)
        deadline -= GROUP_POLL_SECONDS


def _group_failure_message(players: dict, leader: str, missing: Sequence[str]) -> str:
    def describe(player_id: str) -> str:
        player = players.get(player_id)
        return player.name if player else player_id

    names = ", ".join(describe(pid) for pid in missing)
    lines = [
        f"эти колонки не встали в одну группу с «{describe(leader)}», "
        f"поэтому они не услышат тестовый трек: {names}.",
        "Замер сравнивает время прихода одного и того же потока, так что "
        "колонка вне группы измерена быть не может.",
    ]

    leader_player = players.get(leader)
    for player_id in missing:
        player = players.get(player_id)
        if player is None or leader_player is None:
            continue
        if not player.can_group_with_player(leader_player):
            lines.append(
                f"Music Assistant не объединяет «{player.name}» с «{leader_player.name}» — "
                "их провайдеры несовместимы; выключите одну из них."
            )
    return " ".join(lines)


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
        raise ValueError(
            "need at least two players that make a sound and carry a sync_adjust "
            "setting"
        )

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
        # A fault present in both passes reports itself twice; the reader
        # learns nothing from the repetition. dict preserves first-seen order.
        problems=tuple(dict.fromkeys(problems)),
    )
