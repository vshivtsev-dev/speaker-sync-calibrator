"""Turning measured latencies into Music Assistant ``sync_adjust`` values.

The measurement gives each speaker's arrival time relative to an arbitrary
common origin. Only differences carry meaning, so the solver's job is to pick
a target arrival time every speaker can reach and express the move as an
integer millisecond correction inside Music Assistant's ``sync_adjust`` range.
"""

from __future__ import annotations

from dataclasses import dataclass

# Music Assistant's CONF_SYNC_ADJUST is an integer config entry clamped to
# this range (music_assistant/constants.py).
SYNC_ADJUST_LIMIT_MS = 500


@dataclass(frozen=True)
class PlayerMeasurement:
    """What the acoustic pass learned about one speaker."""

    player_id: str
    name: str
    measured_ms: float
    """Arrival time relative to the session's common origin."""

    current_adjust_ms: int
    """The ``sync_adjust`` that was in force *during* the measurement."""

    spread_ms: float = 0.0
    """Stability of the reading across repeated chirps."""


@dataclass(frozen=True)
class PlayerCorrection:
    player_id: str
    name: str
    measured_ms: float
    intrinsic_ms: float
    """Arrival time with the currently applied correction backed out."""

    current_adjust_ms: int
    target_adjust_ms: int
    residual_error_ms: float
    """How far from the target this speaker still lands after applying."""

    clamped: bool
    spread_ms: float = 0.0

    @property
    def delta_ms(self) -> int:
        return self.target_adjust_ms - self.current_adjust_ms


@dataclass(frozen=True)
class CalibrationSolution:
    corrections: tuple[PlayerCorrection, ...]
    strategy: str
    spread_before_ms: float
    spread_after_ms: float

    @property
    def fits(self) -> bool:
        """True when every speaker can reach the target within the ± limit."""
        return not any(correction.clamped for correction in self.corrections)

    @property
    def clamped_players(self) -> tuple[PlayerCorrection, ...]:
        return tuple(c for c in self.corrections if c.clamped)

    def changed(self) -> tuple[PlayerCorrection, ...]:
        """Corrections that actually need writing back to the server."""
        return tuple(c for c in self.corrections if c.delta_ms != 0)


def solve(
    measurements: "list[PlayerMeasurement] | tuple[PlayerMeasurement, ...]",
    *,
    sign: int = 1,
    limit_ms: int = SYNC_ADJUST_LIMIT_MS,
) -> CalibrationSolution:
    """Compute the ``sync_adjust`` each speaker should be given.

    ``sign`` encodes what a positive ``sync_adjust`` does to arrival time: ``1``
    when it delays the player, ``-1`` when the server's convention is the
    opposite. It is established empirically by
    :mod:`spinalign.calibration.validate` rather than assumed, so a convention
    change upstream turns into a flipped flag instead of a silent regression.

    Two targets are tried, in order of preference:

    * **align to the slowest speaker** — every correction is a delay, which is
      always physically achievable, at the cost of needing the full spread of
      headroom;
    * **centre on the midpoint** — halves the headroom each speaker needs and
      so covers twice the spread, at the cost of asking some players to run
      early.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be 1 or -1, got {sign}")
    if not measurements:
        return CalibrationSolution((), "empty", 0.0, 0.0)

    # Back out the correction that was already in effect while measuring.
    intrinsic = {m.player_id: m.measured_ms - sign * m.current_adjust_ms for m in measurements}
    values = list(intrinsic.values())
    slowest, fastest = max(values), min(values)
    spread_before = slowest - fastest

    if spread_before <= limit_ms:
        target, strategy = slowest, "align_to_slowest"
    else:
        target, strategy = (slowest + fastest) / 2.0, "centered"

    corrections = []
    for measurement in measurements:
        own = intrinsic[measurement.player_id]
        wanted = sign * (target - own)
        applied = max(-limit_ms, min(limit_ms, int(round(wanted))))
        corrections.append(
            PlayerCorrection(
                player_id=measurement.player_id,
                name=measurement.name,
                measured_ms=measurement.measured_ms,
                intrinsic_ms=own,
                current_adjust_ms=measurement.current_adjust_ms,
                target_adjust_ms=applied,
                # What the speaker's arrival becomes, versus where we aimed.
                residual_error_ms=(own + sign * applied) - target,
                clamped=abs(wanted) > limit_ms,
                spread_ms=measurement.spread_ms,
            )
        )

    landings = [c.intrinsic_ms + sign * c.target_adjust_ms for c in corrections]
    return CalibrationSolution(
        corrections=tuple(corrections),
        strategy=strategy,
        spread_before_ms=spread_before,
        spread_after_ms=max(landings) - min(landings),
    )
