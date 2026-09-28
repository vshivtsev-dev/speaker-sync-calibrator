"""Checking that the solo readings add up to what a listener hears.

Every reading is taken with one speaker audible, but nobody listens that way.
The verification pass therefore ends with a round in which every speaker
plays at once, and this module checks that round against a prediction.

The prediction is exact in principle. Sound in a room adds linearly, and so
does the correlation, so the together round must equal the sum of the solo
rounds — each speaker's pulse *with its own reflections*, already recorded on
the same chirp grid. That is what makes the check immune to reflections: a
bounce 1 ms behind the direct sound is part of the prediction, not mistaken
for a second speaker.

Measuring how *long* the combined sound lasts does not work instead. A
500 ms sweep followed by a room's reverberant tail barely changes length when
two copies of it are a millisecond apart. After correlation each chirp is a
pulse a fraction of a millisecond wide, and there the comparison is sharp.

The prediction is not compared by a fixed threshold. It is so sensitive that
a 0.1 ms shift already fails it outright — sharper than any real room holds
still — so each speaker's pulse is allowed to slide, and where the pulses
end up is the result: the spread of arrivals as actually heard, set against
the spread the solo rounds promised. What the slid prediction still fails to
explain is a second, coarser signal: a speaker that went quiet or played
something else in the together round.

*Which* speaker moved is deliberately not reported. Speakers whose pulses
look alike — same kind of box, similar placement — can trade places in the
fit without changing the sum, so only the set of arrival times is certain,
not who owns each one. Naming a speaker would sometimes name the wrong one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import next_fast_len

from speaker_sync.calibration.measure import MeasurementAnalysis, RoundPlan
from speaker_sync.dsp.analysis import ms_to_samples, samples_to_ms
from speaker_sync.dsp.detect import Arrival
from speaker_sync.i18n import say

LEAD_MS = 5.0
"""Window start ahead of the earliest direct sound."""

TAIL_MS = 30.0
"""Window end after the latest direct sound: the early reflections."""

MAX_SHIFT_MS = 20.0
"""Largest slide searched for. Anything bigger is not a timing change within
one stream but a different fault, and shows up in the residual instead."""

MAX_SPREAD_MS = 300.0
"""Past this the solo readings are too far apart for a together round to add
anything the verification pass has not already said."""

SPREAD_TOLERANCE_MS = 0.5
"""Widening of the spread below this is within what readings wander by."""

SCATTER_TOLERANCE_FACTOR = 3.0
"""A group whose speakers wander more on their own is given this many times
the worst one's scatter before a widening counts."""

RESIDUAL_LIMIT_DB = -6.0
"""Above this, sliding the solo pulses leaves more than a quarter of the
together round unexplained."""

ITERATIONS = 8


@dataclass(frozen=True)
class TogetherCheck:
    heard_ms: tuple[float, ...]
    """Direct-sound arrivals in the together round, earliest first, on the
    pass's time base. One per speaker, but not attributed — see above."""

    solo_spread_ms: float
    """The spread the solo rounds of the same pass measured."""

    residual_db: float
    """Energy of the together round that the slid prediction leaves unexplained."""

    problems: tuple[str, ...] = field(default=())

    @property
    def spread_ms(self) -> float:
        """Spread of the arrivals with every speaker playing."""
        return self.heard_ms[-1] - self.heard_ms[0] if len(self.heard_ms) >= 2 else 0.0

    @property
    def confirmed(self) -> bool:
        return not self.problems


def check_together(
    correlation: np.ndarray,
    arrivals: Sequence[Arrival],
    rounds: Sequence[RoundPlan],
    analysis: MeasurementAnalysis,
    *,
    period_samples: float,
    sample_rate: int,
    guard_chirps: int,
) -> TogetherCheck | None:
    """Compare the together round with the sum of the solo rounds.

    ``correlation`` is the complex analytic correlation of the whole recording
    (:func:`speaker_sync.dsp.detect.analytic_correlation`). Returns ``None``
    when there is no together round, or nothing to compare it with.
    """
    together = next((r for r in rounds if r.is_together), None)
    readings = analysis.readings
    if together is None or len(readings) < 2 or not arrivals:
        return None

    latencies = {pid: ms_to_samples(r.latency_ms, sample_rate) for pid, r in readings.items()}
    if samples_to_ms(max(latencies.values()) - min(latencies.values()), sample_rate) > (
        MAX_SPREAD_MS
    ):
        return None

    # The grid exactly as the analysis numbered it: chirps counted from the
    # first one heard, drift measured from the analysis's own origin.
    offset = min(a.chirp_index for a in arrivals)
    first = arrivals[0]
    anchor = first.position - first.residual - first.chirp_index * period_samples

    max_lag = int(round(ms_to_samples(MAX_SHIFT_MS, sample_rate)))
    lead = ms_to_samples(LEAD_MS, sample_rate) + max_lag
    start = min(latencies.values()) - lead
    length = int(
        round(
            max(latencies.values())
            - min(latencies.values())
            + lead
            + ms_to_samples(TAIL_MS, sample_rate)
            + max_lag
        )
    )

    def segment_at(chirp: int) -> np.ndarray | None:
        grid = (
            anchor
            + (chirp + offset) * period_samples
            + analysis.drift_per_chirp * (chirp - analysis.drift_origin)
        )
        return _read(correlation, grid + start, length)

    def average(plans: Sequence[RoundPlan]) -> np.ndarray | None:
        segments = [
            segment
            for plan in plans
            for chirp in plan.measured_chirps(guard_chirps)
            if (segment := segment_at(chirp)) is not None
        ]
        return np.mean(segments, axis=0) if segments else None

    solo: dict[str, np.ndarray] = {}
    for player_id in readings:
        plans = [
            r for r in rounds
            if r.player_id == player_id and not r.is_drift_bracket and not r.is_together
        ]
        segment = average(plans)
        if segment is not None:
            solo[player_id] = segment

    heard = average([together])
    if heard is None or len(solo) < 2:
        return TogetherCheck(
            (), 0.0, 0.0,
            (say(
                en="nothing was heard in the round with every speaker playing at once",
                ru="в круге, где играют все колонки сразу, ничего не слышно",
            ),),
        )

    shifts, residual = _fit_shifts(heard, solo, max_lag)
    heard_ms = tuple(
        sorted(
            readings[pid].latency_ms + samples_to_ms(shift, sample_rate)
            for pid, shift in shifts.items()
        )
    )
    solo_ms = [readings[pid].latency_ms for pid in solo]
    solo_spread = max(solo_ms) - min(solo_ms)
    spread = heard_ms[-1] - heard_ms[0]
    residual_db = float(10.0 * np.log10(max(residual, 1e-12)))

    problems: list[str] = []
    scatter = max(readings[pid].spread_ms for pid in solo)
    if spread > solo_spread + max(SPREAD_TOLERANCE_MS, SCATTER_TOLERANCE_FACTOR * scatter):
        problems.append(
            say(
                en=f"with every speaker playing they are {spread:.1f} ms apart, although one "
                f"by one they measured {solo_spread:.1f} ms — something changes when the "
                "whole group plays (a speaker resyncing when the others are unmuted?)",
                ru=f"когда играют все сразу, колонки расходятся на {spread:.1f} мс, хотя по "
                f"одной намерено {solo_spread:.1f} мс — что-то меняется, когда играет вся "
                "группа (колонка пересинхронизируется, когда включаются остальные?)",
            )
        )
    if residual_db > RESIDUAL_LIMIT_DB:
        problems.append(
            say(
                en=f"with every speaker playing the room does not sound like the sum of the "
                f"speakers measured one by one ({residual_db:.0f} dB unexplained) — a "
                "speaker probably stayed silent or played something else in that round",
                ru=f"когда играют все, звук не похож на сумму колонок по одной "
                f"({residual_db:.0f} дБ не объяснено) — похоже, какая-то колонка в этом "
                "круге молчала или играла что-то другое",
            )
        )

    return TogetherCheck(heard_ms, solo_spread, residual_db, tuple(problems))


def _read(signal: np.ndarray, start: float, length: int) -> np.ndarray | None:
    """``signal[start : start + length]`` at a fractional ``start``.

    Segments from different chirps have to line up to a small fraction of a
    sample: at the sweep's centre frequency a tenth of a sample is already a
    noticeable phase error, and the complex average would cancel itself.
    """
    whole = int(np.floor(start))
    fraction = start - whole
    pad = 32
    low, high = whole - pad, whole + length + pad
    if low < 0 or high > len(signal):
        return None
    chunk = signal[low:high]
    freqs = np.fft.fftfreq(len(chunk))
    advanced = np.fft.ifft(np.fft.fft(chunk) * np.exp(2j * np.pi * freqs * fraction))
    return advanced[pad : pad + length]


def _fit_shifts(
    heard: np.ndarray, solo: dict[str, np.ndarray], max_lag: int
) -> tuple[dict[str, float], float]:
    """Slide each solo pulse to best explain ``heard``; return slides and residual.

    Coordinate descent: each speaker in turn is fitted to what the others,
    at their current slides, leave unexplained. The objective is the real
    part of the correlation, so phase counts and a slide is resolved far below
    one sample.
    """
    size = next_fast_len(2 * len(heard))
    freqs = np.fft.fftfreq(size)
    target = np.fft.fft(heard, size)
    spectra = {pid: np.fft.fft(segment, size) for pid, segment in solo.items()}
    shifts = dict.fromkeys(solo, 0.0)

    def placed(player_id: str) -> np.ndarray:
        return spectra[player_id] * np.exp(-2j * np.pi * freqs * shifts[player_id])

    lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
    indices = lags % size
    for _ in range(ITERATIONS):
        for player_id in solo:
            rest = target - sum(placed(other) for other in solo if other != player_id)
            score = np.fft.ifft(rest * np.conj(spectra[player_id])).real
            best = int(np.argmax(score[indices]))
            lag = int(lags[best])
            centre = score[lag % size]
            left, right = score[(lag - 1) % size], score[(lag + 1) % size]
            denominator = left - 2.0 * centre + right
            delta = 0.5 * (left - right) / denominator if denominator < 0 else 0.0
            shifts[player_id] = lag + float(np.clip(delta, -0.5, 0.5))

    unexplained = target - sum(placed(pid) for pid in solo)
    residual = float(np.sum(np.abs(unexplained) ** 2) / max(np.sum(np.abs(target) ** 2), 1e-300))
    return shifts, residual
