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


def _bounds(measurement: "PlayerMeasurement", limit_ms: int) -> tuple[float, float]:
    """The values this player's setting accepts."""
    if measurement.delay_range_ms is None:
        return float(-limit_ms), float(limit_ms)
    low, high = measurement.delay_range_ms
    return float(low), float(high)


def _reachable(own: float, bounds: tuple[float, float], sign: int) -> tuple[float, float]:
    """The arrival times this speaker can be moved to."""
    low, high = bounds
    ends = (own + sign * low, own + sign * high)
    return min(ends), max(ends)


def _pick_target(
    slowest: float, fastest: float, reach_low: float, reach_high: float
) -> tuple[float, str]:
    """Choose an arrival time to aim every speaker at, and name the choice."""
    if reach_low > reach_high:
        # No single arrival time is within everyone's reach. Aim at the middle
        # and let the per-speaker clamp report who could not make it.
        return (slowest + fastest) / 2.0, "centered"

    for candidate, strategy in (
        (slowest, "align_to_slowest"),
        ((slowest + fastest) / 2.0, "centered"),
    ):
        if reach_low <= candidate <= reach_high:
            return candidate, strategy

    # Neither preference is reachable, which is what an advance-only setting
    # looks like: nothing can be delayed, so everyone comes forward instead.
    target = min(max(slowest, reach_low), reach_high)
    return target, "align_to_fastest" if target <= fastest else "limited"


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

    delay_range_ms: tuple[int, int] | None = None
    """What this player's delay setting actually accepts, as the server says.

    Not every server offers the symmetric ±500 ms that Music Assistant's own
    ``sync_adjust`` has. A Sendspin player carries ``static_delay_ms``, which
    runs 0–5000: it can only ever advance a player, never delay one, so the
    alignment has to be aimed somewhere every speaker can actually reach.
    ``None`` falls back to the symmetric default.
    """

    sign: int | None = None
    """This player's own direction, when its setting has a specified one.

    Overrides the session-wide ``sign``: a Sendspin player advances with a
    larger value whatever the probe found for some other kind of setting.
    """


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
    sign: int | None = None
    """The player's specified direction, or ``None`` when the session's
    probed one was used for it."""

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
    :mod:`speaker_sync.calibration.validate` rather than assumed, so a convention
    change upstream turns into a flipped flag instead of a silent regression.

    The target arrival time is whatever every speaker can actually reach. Each
    speaker's setting has its own accepted range, which the server reports, and
    that range plus ``sign`` is what says where that speaker can land. The
    intersection of those windows is the set of workable targets, and the
    preference inside it is:

    * **align to the slowest speaker** — every correction is then a delay,
      which asks nothing of the setting beyond headroom;
    * **centre on the midpoint** — halves the headroom each speaker needs and
      so covers twice the spread, at the cost of asking some players to run
      early;
    * otherwise the nearest reachable point, which on an advance-only setting
      like Sendspin's ``static_delay_ms`` means aligning to the *fastest*
      speaker and pulling the rest forward to meet it.

    When the windows do not overlap at all, no target aligns everyone; the
    midpoint is used and the speakers that cannot reach it are marked clamped.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be 1 or -1, got {sign}")
    if not measurements:
        return CalibrationSolution((), "empty", 0.0, 0.0)

    signs = {m.player_id: m.sign or sign for m in measurements}

    # Back out the correction that was already in effect while measuring.
    intrinsic = {
        m.player_id: m.measured_ms - signs[m.player_id] * m.current_adjust_ms
        for m in measurements
    }
    values = list(intrinsic.values())
    slowest, fastest = max(values), min(values)
    spread_before = slowest - fastest

    windows = {
        m.player_id: _reachable(intrinsic[m.player_id], _bounds(m, limit_ms), signs[m.player_id])
        for m in measurements
    }
    reach_low = max(low for low, _ in windows.values())
    reach_high = min(high for _, high in windows.values())

    target, strategy = _pick_target(slowest, fastest, reach_low, reach_high)

    corrections = []
    for measurement in measurements:
        own = intrinsic[measurement.player_id]
        own_sign = signs[measurement.player_id]
        low, high = _bounds(measurement, limit_ms)
        wanted = own_sign * (target - own)
        applied = int(round(max(low, min(high, wanted))))
        corrections.append(
            PlayerCorrection(
                player_id=measurement.player_id,
                name=measurement.name,
                measured_ms=measurement.measured_ms,
                intrinsic_ms=own,
                current_adjust_ms=measurement.current_adjust_ms,
                target_adjust_ms=applied,
                # What the speaker's arrival becomes, versus where we aimed.
                residual_error_ms=(own + own_sign * applied) - target,
                clamped=not low - 0.5 <= wanted <= high + 0.5,
                spread_ms=measurement.spread_ms,
                sign=measurement.sign,
            )
        )

    landings = [c.intrinsic_ms + signs[c.player_id] * c.target_adjust_ms for c in corrections]
    return CalibrationSolution(
        corrections=tuple(corrections),
        strategy=strategy,
        spread_before_ms=spread_before,
        spread_after_ms=max(landings) - min(landings),
    )
