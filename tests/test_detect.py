"""Accuracy of arrival detection against known ground truth.

Every case here renders a room whose latencies we chose, then asserts the
pipeline recovers them. The two cases that matter most are not the accurate
ones — they are the reflection test, which is what a naive ``argmax`` fails,
and the silent-speaker test, which is what separates "no answer" from a
confidently wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.virtual_room import VirtualSpeaker
from speaker_sync.dsp.detect import (
    correlation_envelope,
    detect_arrivals,
    detection_threshold,
    estimate_noise_floor,
    find_first_arrival,
)
from speaker_sync.dsp.signals import build_test_signal, exponential_sweep, to_wav_bytes
from tests.support import run_session, three_speakers

REFERENCE = "a"


def test_clean_signal_is_essentially_exact():
    result = run_session(three_speakers(), snr_db=40.0)

    assert not result.analysis.problems
    assert result.worst_error_ms(REFERENCE) < 0.05


@pytest.mark.parametrize("snr_db", [40.0, 20.0, 10.0, 0.0, -20.0])
def test_accuracy_survives_noise(snr_db):
    """Matched filtering buys ~30 dB of processing gain over a 300 ms sweep,
    so even a recording where the chirp is buried below the noise resolves."""
    result = run_session(three_speakers(), snr_db=snr_db)

    assert not result.analysis.problems
    assert result.worst_error_ms(REFERENCE) < 1.0


def test_hopeless_noise_reports_nothing_rather_than_guessing():
    result = run_session(three_speakers(), snr_db=-40.0)

    assert result.analysis.problems
    assert not result.analysis.is_usable
    assert len(result.analysis.readings) < 2


def test_strong_early_reflection_does_not_win():
    """A wall bounce 8 ms late at 0.8 of the direct level.

    Picking the largest correlation peak would still be right here only by
    luck; the case below removes even that.
    """
    speakers = three_speakers(
        a={"reflections": ((8.0, 0.8), (17.0, 0.5))},
        b={"reflections": ((11.0, 0.9), (23.0, 0.6))},
        c={"reflections": ((6.0, 0.75),)},
    )
    result = run_session(speakers, snr_db=25.0)

    assert not result.analysis.problems
    assert result.worst_error_ms(REFERENCE) < 1.0


def test_reflection_louder_than_direct_sound():
    """The case that breaks argmax outright: the bounce is the biggest peak.

    Reported latency must still track the direct sound, so the error stays far
    below the reflection delays (9 and 12 ms) rather than snapping to them.
    """
    speakers = [
        VirtualSpeaker("a", "ESP32", hardware_latency_ms=20.0, distance_m=2.0,
                       reflections=((9.0, 1.4),)),
        VirtualSpeaker("b", "AVR", hardware_latency_ms=80.0, distance_m=4.0,
                       reflections=((12.0, 1.3),)),
    ]
    result = run_session(speakers, snr_db=25.0)

    assert not result.analysis.problems
    assert result.worst_error_ms(REFERENCE) < 1.0


def test_silent_speaker_is_reported_missing_not_measured():
    """The failure mode we most need to avoid: a muted or dead speaker coming
    back with a plausible-looking number invented from noise."""
    speakers = three_speakers(b={"gain": 0.0})
    result = run_session(speakers, snr_db=30.0)

    assert "b" not in result.analysis.readings
    assert any("b" in problem for problem in result.analysis.problems)
    # The speakers that were audible are still measured correctly.
    assert abs(result.relative_error_ms("c", REFERENCE)) < 1.0


@pytest.mark.parametrize("clock_ppm", [0.0, 50.0, 200.0])
def test_microphone_clock_drift_is_divided_out(clock_ppm):
    result = run_session(three_speakers(), snr_db=30.0, clock_ppm=clock_ppm)

    assert not result.analysis.problems
    assert result.worst_error_ms(REFERENCE) < 1.0


@pytest.mark.parametrize("mic_sample_rate", [44100, 48000])
def test_common_microphone_sample_rates(mic_sample_rate):
    result = run_session(three_speakers(), snr_db=30.0, mic_sample_rate=mic_sample_rate)

    assert not result.analysis.problems
    assert result.worst_error_ms(REFERENCE) < 1.0


def test_clipped_recording_still_resolves():
    """Phone microphones clip readily when a speaker is loud; the sweep's
    zero crossings survive it even when the peaks do not."""
    result = run_session(three_speakers(), snr_db=30.0, clip=True, amplitude=0.9)

    assert not result.analysis.problems
    assert result.worst_error_ms(REFERENCE) < 1.0


def test_large_spread_within_one_period():
    """Latency differences approaching half the chirp period must still be
    attributed to the right chirp."""
    speakers = [
        VirtualSpeaker("a", "fast", hardware_latency_ms=15.0, distance_m=1.0),
        VirtualSpeaker("b", "very slow", hardware_latency_ms=480.0, distance_m=5.0),
    ]
    result = run_session(speakers, snr_db=30.0)

    assert not result.analysis.problems
    assert abs(result.relative_error_ms("b", "a")) < 1.0


def test_detection_threshold_grows_with_window_size():
    """The heart of the silent-speaker fix: a bigger search window needs a
    higher bar, because the largest noise sample in it is bigger."""
    small = detection_threshold(1.0, 100, 12.0)
    large = detection_threshold(1.0, 100_000, 12.0)

    assert large > small
    assert small >= 10.0 ** (12.0 / 20.0)


def test_find_first_arrival_returns_none_on_pure_noise():
    rng = np.random.default_rng(0)
    reference = exponential_sweep(duration=0.3, sample_rate=48000)
    noise = rng.normal(0.0, 1.0, 48000 * 3)

    envelope = correlation_envelope(noise, reference)
    found = find_first_arrival(envelope, noise_floor=estimate_noise_floor(envelope))

    assert found is None


def test_sub_sample_resolution():
    """A chirp placed at a fractional offset must be located to well under one
    sample, which is what keeps the millisecond budget for the room."""
    rate = 48000
    reference = exponential_sweep(duration=0.3, sample_rate=rate)
    offset = 1000.37

    padded = np.zeros(rate * 2)
    whole = int(offset)
    padded[whole : whole + len(reference)] = reference

    envelope = correlation_envelope(padded, reference)
    found = find_first_arrival(envelope, noise_floor=estimate_noise_floor(envelope))

    assert found is not None
    assert abs(found[0] - whole) < 0.5


def test_detect_arrivals_finds_every_chirp():
    signal = build_test_signal(chirp_count=8)
    arrivals = detect_arrivals(
        signal.samples,
        signal.reference(),
        period_samples=signal.period_samples,
        sample_rate=signal.sample_rate,
        f_start=signal.f_start,
        f_end=signal.f_end,
    )

    assert len(arrivals) == 8
    # An undelayed track sits exactly on the grid, so residuals are ~zero.
    assert max(abs(a.residual) for a in arrivals) < 1.0


def test_test_signal_encodes_to_playable_wav():
    signal = build_test_signal(chirp_count=4)
    data = to_wav_bytes(signal.samples, signal.sample_rate)

    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert len(data) > len(signal.samples) * 2
