"""The closing round with every speaker playing at once.

The solo rounds measure each speaker alone; the together round checks that
those readings still hold when the whole group plays, which is how anyone
actually listens. These tests give the room a fault that only shows up with
everyone audible and expect it to be caught, named and sized — and give it
reflections and noise that must not be mistaken for one.
"""

from __future__ import annotations

import pytest

from sim.fake_ma import mixed_speakers
from speaker_sync.calibration.measure import TOGETHER, analyze, plan_rounds, total_chirps
from speaker_sync.calibration.session import calibrate
from tests.test_end_to_end import make_server


def speakers_with(**overrides):
    return [
        s.__class__(**{**s.__dict__, **overrides.get(s.player_id, {})}) for s in mixed_speakers()
    ]


def test_the_together_round_comes_last_and_is_nobodys_reading():
    rounds = plan_rounds("a", ["a", "b", "c"], chirps_per_round=5, together=True)

    assert rounds[-1].is_together
    assert rounds[-1].player_id == TOGETHER
    assert rounds[-1].first_chirp == rounds[-2].first_chirp + 5
    assert total_chirps(rounds) == 5 * 5
    assert not any(r.is_together for r in plan_rounds("a", ["a", "b"], chirps_per_round=5))


def test_a_together_round_without_any_chirps_is_not_blamed_on_a_speaker():
    rounds = plan_rounds("a", ["a", "b"], chirps_per_round=3, together=True)
    analysis = analyze([], rounds, sample_rate=48000)

    assert not any(TOGETHER in problem for problem in analysis.problems)


async def test_a_calibrated_room_is_confirmed_with_everyone_playing():
    server, recorder, clock = make_server(snr_db=30.0)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    check = report.together
    assert check is not None
    assert check.confirmed
    assert len(check.heard_ms) == len(server.speakers)
    assert check.solo_spread_ms == pytest.approx(report.spread_after_ms)
    assert check.spread_ms == pytest.approx(report.spread_after_ms, abs=0.05)
    assert check.residual_db < -20.0
    assert not report.problems


@pytest.mark.parametrize("snr_db", [30.0, 0.0])
async def test_reflections_are_part_of_the_prediction_not_a_second_speaker(snr_db):
    """Bounces 0.5–11 ms behind each direct sound, some louder than it. Judged
    by width, the together round would look smeared; against the sum of the
    solo rounds it matches."""
    speakers = speakers_with(
        esp32={"reflections": ((0.6, 0.9), (8.0, 0.8))},
        avr={"reflections": ((1.1, 1.2), (11.0, 1.3))},
        bt={"reflections": ((0.5, 1.0), (6.0, 0.7))},
    )
    server, recorder, clock = make_server(speakers, snr_db=snr_db)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.together is not None
    assert report.together.confirmed
    assert report.together.spread_ms == pytest.approx(report.spread_after_ms, abs=0.1)


@pytest.mark.parametrize("shift_ms", [1.5, -0.8, 12.0])
async def test_a_speaker_that_moves_when_the_others_play_is_caught_and_sized(shift_ms):
    """Invisible to every solo round, including the verification pass: its
    spread reads as fine. Only the together round can tell."""
    speakers = speakers_with(avr={"together_shift_ms": shift_ms})
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.spread_after_ms < 1.0  # the solo rounds see nothing wrong
    check = report.together
    assert check is not None
    assert not check.confirmed
    # The arrivals of the others stay put, so the spread grows by the shift,
    # give or take where the moved speaker sat within the solo spread.
    assert check.spread_ms == pytest.approx(abs(shift_ms), abs=check.solo_spread_ms + 0.05)
    assert any(f"{check.spread_ms:.1f}" in problem for problem in report.problems)


async def test_the_together_round_confirms_a_run_that_had_to_fix_its_own_sign():
    """Written the wrong way first, then rewritten: the together round is the
    independent evidence that the final direction is right."""
    speakers = speakers_with(**{s.player_id: {"sign": -1} for s in mixed_speakers()})
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    report = await calibrate(server, recorder, sign=1, sleep=clock.sleep)

    assert report.sign == -1
    assert report.together is not None
    assert report.together.confirmed
    assert report.together.spread_ms < 1.0


async def test_without_verification_there_is_no_together_round():
    server, recorder, clock = make_server(snr_db=30.0)

    report = await calibrate(server, recorder, verify=False, sleep=clock.sleep)

    assert report.together is None
