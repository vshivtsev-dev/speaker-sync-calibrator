"""From detected arrivals to a latency per speaker.

The session plays one long track of evenly spaced chirps while muting all but
one speaker at a time. Because the recording runs unbroken from start to
finish and the chirp grid is exact, every round is measured against the same
time base — the unknown offset between the microphone's clock and the stream's
clock is common to all of them and cancels when speakers are compared.

Rounds are expressed in chirp indices rather than seconds. A mute command
lands somewhere inside a chirp period, so counting chirps sidesteps having to
know exactly when it took effect; the first couple of chirps in each round are
discarded as a guard instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from spinalign.dsp.analysis import RobustEstimate, robust_estimate, samples_to_ms
from spinalign.dsp.detect import Arrival

DEFAULT_CHIRPS_PER_ROUND = 5
DEFAULT_GUARD_CHIRPS = 2

# Readings inside one round come from the same speaker in the same position,
# so they should agree closely. Anything looser means the recording or the
# stream hiccuped during that round.
#
# The check is on the worst *individual* deviation from the round's median,
# not on the scatter of the group. With only a handful of readings per speaker
# a single bad one leaves the scatter untouched — for [X, X, X+5] the median
# absolute deviation is exactly zero — so a spread-based check would stay
# silent about precisely the glitch worth reporting. The median still gives
# the right answer in that case; what would be lost is telling the user that
# something moved.
OUTLIER_TOLERANCE_MS = 2.0


@dataclass(frozen=True)
class RoundPlan:
    """One stretch of the track during which a single speaker is audible."""

    player_id: str
    first_chirp: int
    chirp_count: int
    is_drift_bracket: bool = False

    @property
    def last_chirp(self) -> int:
        return self.first_chirp + self.chirp_count - 1

    def measured_chirps(self, guard: int) -> range:
        """Chirp indices actually used, after dropping the settling guard."""
        return range(self.first_chirp + guard, self.first_chirp + self.chirp_count)


@dataclass(frozen=True)
class SpeakerReading:
    player_id: str
    latency_ms: float
    """Arrival time relative to the session's common origin."""

    spread_ms: float
    arrivals_used: int
    rejected: int


@dataclass(frozen=True)
class MeasurementAnalysis:
    readings: dict[str, SpeakerReading]
    drift_ms: float
    """Microphone clock drift across the whole session, already divided out."""

    problems: tuple[str, ...] = field(default=())

    @property
    def is_usable(self) -> bool:
        return len(self.readings) >= 2 and not self.problems


def plan_rounds(
    reference_id: str,
    player_ids: Sequence[str],
    *,
    chirps_per_round: int = DEFAULT_CHIRPS_PER_ROUND,
) -> list[RoundPlan]:
    """Lay out the session: reference, every other speaker, reference again.

    The repeated reference at the end is what makes drift measurable — the two
    readings of the same speaker differ only by how far the microphone's clock
    wandered while the session ran.
    """
    if chirps_per_round < 1:
        raise ValueError(f"chirps_per_round must be >= 1, got {chirps_per_round}")

    order = [reference_id, *[p for p in player_ids if p != reference_id], reference_id]
    return [
        RoundPlan(
            player_id=player_id,
            first_chirp=position * chirps_per_round,
            chirp_count=chirps_per_round,
            is_drift_bracket=position == len(order) - 1,
        )
        for position, player_id in enumerate(order)
    ]


def total_chirps(rounds: Sequence[RoundPlan]) -> int:
    return max((r.last_chirp for r in rounds), default=-1) + 1


def analyze(
    arrivals: Sequence[Arrival],
    rounds: Sequence[RoundPlan],
    *,
    sample_rate: int,
    guard_chirps: int = DEFAULT_GUARD_CHIRPS,
) -> MeasurementAnalysis:
    """Reduce detected arrivals to one latency per speaker.

    Arrival indices are relative to whichever chirp the detector anchored on,
    so they are first rebased onto the track's own numbering by treating the
    earliest detection as chirp zero. The session always starts with the
    reference speaker audible, so chirp zero is reliably present.
    """
    if not arrivals:
        return MeasurementAnalysis({}, 0.0, ("в записи не найдено ни одного свиста",))

    offset = min(a.chirp_index for a in arrivals)
    by_chirp = {a.chirp_index - offset: a for a in arrivals}

    problems: list[str] = []
    # Index and residual are kept paired: chirps can be missing (a muted
    # transition, a dropout), and pairing keeps the drift correction attached
    # to the right point in the session.
    per_round: list[tuple[RoundPlan, list[tuple[int, float]]]] = []
    for plan in rounds:
        readings_in_round = [
            (index, by_chirp[index].residual)
            for index in plan.measured_chirps(guard_chirps)
            if index in by_chirp
        ]
        per_round.append((plan, readings_in_round))
        if not readings_in_round:
            problems.append(f"{plan.player_id}: не слышно ни одного свиста в своём круге")

    drift_rate, drift_origin, drift_total = _estimate_drift(per_round)

    readings: dict[str, SpeakerReading] = {}
    for plan, readings_in_round in per_round:
        if not readings_in_round or plan.is_drift_bracket:
            continue

        corrected = [
            residual - drift_rate * (index - drift_origin) for index, residual in readings_in_round
        ]
        estimate = robust_estimate(corrected)
        if estimate is None:
            continue

        spread_ms = samples_to_ms(estimate.spread, sample_rate)
        worst_ms = samples_to_ms(
            max((abs(sample - estimate.value) for sample in estimate.samples), default=0.0),
            sample_rate,
        )
        if worst_ms > OUTLIER_TOLERANCE_MS:
            problems.append(
                f"{plan.player_id}: один отсчёт отличается от остальных на {worst_ms:.1f} мс — "
                "в этом круге сбоила запись или поток"
            )

        readings[plan.player_id] = SpeakerReading(
            player_id=plan.player_id,
            latency_ms=samples_to_ms(estimate.value, sample_rate),
            spread_ms=spread_ms,
            arrivals_used=estimate.kept,
            rejected=estimate.rejected,
        )

    if len(readings) < 2:
        problems.append("годный отсчёт дала меньше чем одна пара колонок")

    return MeasurementAnalysis(
        readings=readings,
        drift_ms=samples_to_ms(drift_total, sample_rate),
        problems=tuple(problems),
    )


def _estimate_drift(
    per_round: Sequence[tuple[RoundPlan, list[tuple[int, float]]]],
) -> tuple[float, float, float]:
    """Drift rate per chirp, the chirp it is measured from, and the total.

    Found by comparing the opening and closing readings of the *same* speaker.
    Returns zeros when the bracket is missing or malformed: leaving a
    sub-millisecond drift uncorrected is better than inventing a correction.
    """
    opening = next(((p, r) for p, r in per_round if not p.is_drift_bracket and r), None)
    closing = next(((p, r) for p, r in reversed(per_round) if p.is_drift_bracket and r), None)
    if opening is None or closing is None:
        return 0.0, 0.0, 0.0

    open_plan, open_readings = opening
    close_plan, close_readings = closing
    if open_plan.player_id != close_plan.player_id:
        return 0.0, 0.0, 0.0

    open_estimate = robust_estimate([residual for _, residual in open_readings])
    close_estimate = robust_estimate([residual for _, residual in close_readings])
    if open_estimate is None or close_estimate is None:
        return 0.0, 0.0, 0.0

    open_centre = _centre(open_readings)
    span = _centre(close_readings) - open_centre
    if span <= 0:
        return 0.0, 0.0, 0.0

    total = close_estimate.value - open_estimate.value
    return total / span, open_centre, total


def _centre(readings: Sequence[tuple[int, float]]) -> float:
    return sum(index for index, _ in readings) / len(readings)
