"""Turning measured latencies into ``sync_adjust`` values.

The arithmetic is small but every term earns its place: an already-applied
correction has to be backed out or the solver chases its own tail, the ±500 ms
limit has to be respected, and when the spread exceeds what one-sided delays
can cover the solver has to switch strategy rather than silently clamp.
"""

from __future__ import annotations

import pytest

from spinalign.calibration.solver import (
    SYNC_ADJUST_LIMIT_MS,
    PlayerMeasurement,
    solve,
)


def measurement(player_id, measured_ms, current=0):
    return PlayerMeasurement(
        player_id=player_id, name=player_id, measured_ms=measured_ms, current_adjust_ms=current
    )


def landings(solution, sign=1):
    return [c.intrinsic_ms + sign * c.target_adjust_ms for c in solution.corrections]


def test_aligns_everyone_to_the_slowest_speaker():
    solution = solve(
        [
            measurement("esp32", 25.8),
            measurement("avr", 91.7),
            measurement("bluetooth", 228.7),
        ]
    )

    assert solution.strategy == "align_to_slowest"
    assert solution.fits
    # Nothing can be made faster, so every correction is a delay.
    assert all(c.target_adjust_ms >= 0 for c in solution.corrections)
    # The slowest speaker sets the target and needs no correction of its own.
    assert solution.corrections[2].target_adjust_ms == 0
    assert solution.spread_after_ms < 1.0


def test_existing_correction_is_backed_out():
    """A speaker measured *with* a correction already applied must not have it
    counted twice, or a second calibration pass would double the delay."""
    fresh = solve([measurement("a", 20.0), measurement("b", 120.0)])
    # Same physical speakers, but 'a' already carries the correction fresh
    # would have produced, so it now measures 100 ms later.
    already = solve([measurement("a", 120.0, current=100), measurement("b", 120.0)])

    assert fresh.corrections[0].target_adjust_ms == 100
    assert already.corrections[0].target_adjust_ms == 100
    assert already.corrections[0].delta_ms == 0
    assert already.changed() == ()


def test_spread_within_limit_prefers_one_sided_delays():
    solution = solve([measurement("a", 0.0), measurement("b", 480.0)])

    assert solution.strategy == "align_to_slowest"
    assert solution.fits
    assert solution.corrections[0].target_adjust_ms == 480


def test_spread_beyond_the_limit_centres_instead_of_clamping():
    """Aligning to the slowest needs the full spread of headroom; centring
    needs only half, which is what doubles the usable range to 1000 ms."""
    solution = solve([measurement("a", 0.0), measurement("b", 900.0)])

    assert solution.strategy == "centered"
    assert solution.fits
    assert solution.corrections[0].target_adjust_ms == 450
    assert solution.corrections[1].target_adjust_ms == -450
    assert solution.spread_after_ms < 1.0


def test_spread_beyond_even_centring_is_reported_not_hidden():
    solution = solve([measurement("a", 0.0), measurement("b", 1400.0)])

    assert not solution.fits
    assert len(solution.clamped_players) == 2
    assert all(abs(c.target_adjust_ms) <= SYNC_ADJUST_LIMIT_MS for c in solution.corrections)
    # The residual is honest about how far off the pair still is.
    assert abs(solution.spread_after_ms) > 300.0


def test_inverted_sign_convention_is_handled():
    """If a positive ``sync_adjust`` made a player run *early*, the same
    measurements must produce mirrored corrections."""
    normal = solve([measurement("a", 0.0), measurement("b", 200.0)], sign=1)
    inverted = solve([measurement("a", 0.0), measurement("b", 200.0)], sign=-1)

    assert normal.corrections[0].target_adjust_ms == 200
    assert inverted.corrections[0].target_adjust_ms == -200
    # Either way the speakers end up in the same place.
    assert max(landings(inverted, sign=-1)) - min(landings(inverted, sign=-1)) < 1.0


def test_integer_rounding_stays_within_half_a_millisecond():
    solution = solve([measurement("a", 0.0), measurement("b", 100.4), measurement("c", 200.6)])

    assert all(isinstance(c.target_adjust_ms, int) for c in solution.corrections)
    assert all(abs(c.residual_error_ms) <= 0.5 for c in solution.corrections)


def test_changed_lists_only_what_needs_writing():
    solution = solve(
        [measurement("a", 0.0, current=200), measurement("b", 200.0, current=0)]
    )

    changed = {c.player_id for c in solution.changed()}
    assert "b" not in changed


def test_already_aligned_speakers_need_no_writes():
    solution = solve([measurement("a", 50.0), measurement("b", 50.0), measurement("c", 50.0)])

    assert solution.changed() == ()
    assert solution.spread_before_ms < 1.0


def test_empty_input_is_not_an_error():
    solution = solve([])

    assert solution.corrections == ()
    assert solution.strategy == "empty"


def test_rejects_a_nonsense_sign():
    with pytest.raises(ValueError):
        solve([measurement("a", 0.0)], sign=0)
