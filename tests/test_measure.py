"""Turning arrivals into per-speaker latencies, and noticing when not to trust them.

These build arrivals directly rather than going through the room, because what
is under test is the reduction step: which readings belong to which speaker,
how drift is divided out, and — the part that matters most here — whether a
round that misbehaved is reported as such instead of quietly averaging out.

A calibration that is silently wrong is far worse than one that says it
failed, so "the value survived" and "the user was told" are asserted
separately.
"""

from __future__ import annotations

import pytest

from speaker_sync.calibration.measure import analyze, plan_rounds, total_chirps
from speaker_sync.dsp.analysis import ms_to_samples
from speaker_sync.dsp.detect import Arrival

RATE = 48000


def arrivals_from(residuals: dict[int, float]) -> list[Arrival]:
    """Build arrivals with chosen residuals, in samples."""
    return [
        Arrival(
            chirp_index=index,
            position=float(index),
            residual=residual,
            peak_value=100.0,
            noise_floor=1.0,
        )
        for index, residual in sorted(residuals.items())
    ]


def session(**per_chirp_ms: float) -> tuple[list[Arrival], list]:
    """A two-speaker session, with named chirps nudged by a given number of ms.

    Keys are ``c<index>`` so they can be passed as keyword arguments, e.g.
    ``session(c9=5.0)`` puts one chirp of the second speaker's round 5 ms late.
    """
    rounds = plan_rounds("left", ["left", "right"])
    residuals = {index: 0.0 for index in range(total_chirps(rounds))}
    for name, offset_ms in per_chirp_ms.items():
        residuals[int(name.removeprefix("c"))] = ms_to_samples(offset_ms, RATE)
    return arrivals_from(residuals), rounds


def problems_about(analysis, player_id: str) -> list[str]:
    return [problem for problem in analysis.problems if player_id in problem]


def test_a_clean_session_reports_nothing():
    arrivals, rounds = session()

    analysis = analyze(arrivals, rounds, sample_rate=RATE)

    assert analysis.problems == ()
    assert set(analysis.readings) == {"left", "right"}
    assert analysis.readings["right"].latency_ms == pytest.approx(0.0, abs=0.01)


def test_a_lone_outlier_is_reported_even_though_the_median_absorbs_it():
    """The case a scatter check alone cannot see.

    With three readings per speaker, ``[X, X, X+5]`` has a median of X — so the
    answer is right — but a median absolute deviation of *zero*, so a check
    built on spread stays silent. The value must survive and the user must
    still be told something moved.
    """
    arrivals, rounds = session(c9=5.0)

    analysis = analyze(arrivals, rounds, sample_rate=RATE)

    assert analysis.readings["right"].latency_ms == pytest.approx(0.0, abs=0.01)
    assert problems_about(analysis, "right")


def test_a_systematic_drift_within_a_round_is_reported():
    """Readings marching in one direction do raise the spread, so this case was
    already caught; it must not regress while fixing the one above."""
    arrivals, rounds = session(c8=5.0, c9=10.0)

    analysis = analyze(arrivals, rounds, sample_rate=RATE)

    assert problems_about(analysis, "right")


def test_small_wobble_is_not_flagged():
    """Sub-millisecond scatter is ordinary room and detector noise. Warning
    about it would train the user to ignore warnings."""
    arrivals, rounds = session(c7=0.3, c8=-0.2, c9=0.4)

    analysis = analyze(arrivals, rounds, sample_rate=RATE)

    assert analysis.problems == ()


def test_a_glitch_in_one_round_does_not_implicate_another():
    arrivals, rounds = session(c9=5.0)

    analysis = analyze(arrivals, rounds, sample_rate=RATE)

    assert problems_about(analysis, "right")
    assert not problems_about(analysis, "left")


def test_latency_differences_are_recovered():
    """The reduction itself: the right speaker arrives 40 ms after the left."""
    arrivals, rounds = session(c7=40.0, c8=40.0, c9=40.0)

    analysis = analyze(arrivals, rounds, sample_rate=RATE)

    difference = analysis.readings["right"].latency_ms - analysis.readings["left"].latency_ms
    assert difference == pytest.approx(40.0, abs=0.01)
    assert analysis.problems == ()


def test_microphone_drift_is_divided_out():
    """The reference is measured first and last; the gap between those two
    readings is the microphone's clock drift and belongs to no speaker."""
    # Reference reads 0 at the start and 6 ms at the end: 6 ms of drift across
    # the session. Speaker b sits in the middle, so it picks up about half.
    arrivals, rounds = session(c12=6.0, c13=6.0, c14=6.0, c7=3.0, c8=3.0, c9=3.0)

    analysis = analyze(arrivals, rounds, sample_rate=RATE)

    assert analysis.drift_ms == pytest.approx(6.0, abs=0.1)
    difference = analysis.readings["right"].latency_ms - analysis.readings["left"].latency_ms
    assert abs(difference) < 0.5


def test_a_missing_speaker_is_named():
    rounds = plan_rounds("left", ["left", "right"])
    residuals = {index: 0.0 for index in range(total_chirps(rounds))}
    for index in (7, 8, 9):
        del residuals[index]

    analysis = analyze(arrivals_from(residuals), rounds, sample_rate=RATE)

    assert "right" not in analysis.readings
    assert problems_about(analysis, "right")
    assert not analysis.is_usable


def test_no_arrivals_at_all_is_reported_not_crashed():
    rounds = plan_rounds("left", ["left", "right"])

    analysis = analyze([], rounds, sample_rate=RATE)

    assert analysis.readings == {}
    assert analysis.problems
