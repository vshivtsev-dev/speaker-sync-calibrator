"""Arrival-time detection by matched filtering.

The measurement rests on one quantity: *when* a chirp reached the microphone.
Two things make that harder than taking an argmax.

First, rooms reflect. A wall bounce arriving 8 ms late can easily be louder
than the direct sound, and picking the loudest peak would then report a
latency that is wrong by the reflection delay. We always take the *first*
arrival above a fraction of the local peak, never the largest one.

Second, we need far better than one-sample resolution. Matched filtering
compresses the sweep to a pulse only a few samples wide, and fitting a
parabola to the envelope around its top recovers the true maximum to a small
fraction of a sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import butter, fftconvolve, hilbert, sosfiltfilt

# Fraction of the window's peak that counts as "the signal has arrived".
# Anything lower starts triggering on noise; much higher starts skipping a
# quiet direct sound in favour of a loud reflection.
DEFAULT_ARRIVAL_RATIO = 0.5

# Below this peak-to-noise ratio we report nothing rather than a number we do
# not believe. Reporting "no signal" is always better than reporting noise.
DEFAULT_MIN_PSR_DB = 12.0

# How far above the *expected* noise maximum a peak must sit. The threshold
# below already tracks window size; this is the safety margin on top of it.
NOISE_PEAK_MARGIN = 1.6

# Median of a Rayleigh distribution in units of its scale parameter. The
# envelope of band-limited noise is Rayleigh, and we estimate its floor with a
# median, so this converts that estimate back to the scale.
RAYLEIGH_MEDIAN = 1.1774

# How far ahead of the window's strongest point the direct sound may lie.
# The loudest arrival can be a reflection, but one whose extra path is more
# than ~10 m (30 ms) is not going to out-shout the direct sound in a home. The
# bound matters because of what lies further back: with an exponential sweep a
# speaker's harmonic distortion correlates with the reference *ahead* of the
# linear response (Farina, AES 108, 2000) — see :func:`harmonic_lead_seconds`.
MAX_DIRECT_LEAD_SECONDS = 0.030

# Keep the search this far clear of the 2nd harmonic's position, so a sweep
# configured shorter or wider than the default cannot slide it back in.
HARMONIC_CLEARANCE = 0.8

# Regularisation of the whitening, relative to the reference's peak power
# density. Small enough to act across the whole sweep band, large enough that
# the band edges — where the sweep has almost no energy — are not boosted into
# ringing.
WHITENING_REGULARIZATION = 1e-3

# A candidate weaker than the recording's median arrival by more than this
# factor (~34 dB) is treated as an artefact rather than a quiet speaker.
DEFAULT_RELATIVE_FLOOR = 50.0


def detection_threshold(noise_floor: float, window_length: int, min_psr_db: float) -> float:
    """Amplitude a peak must exceed to count as a real arrival.

    A fixed peak-to-noise ratio is not enough, because the largest value in a
    window of pure noise grows with the size of the window. Searching half a
    period at 48 kHz means looking at tens of thousands of samples, where the
    noise maximum alone lands around 12 dB above the median — which is how a
    muted speaker ends up "detected" at an arbitrary position.

    So the threshold carries a term for the expected maximum of ``n`` Rayleigh
    samples, ``sqrt(2 ln n)``, and a fixed floor for the small-window case.
    """
    if noise_floor <= 0.0:
        return 0.0
    fixed = 10.0 ** (min_psr_db / 20.0)
    expected_noise_peak = np.sqrt(2.0 * np.log(max(window_length, 2))) / RAYLEIGH_MEDIAN
    return noise_floor * max(fixed, expected_noise_peak * NOISE_PEAK_MARGIN)


def harmonic_lead_seconds(chirp_seconds: float, f_start: float, f_end: float, order: int = 2) -> float:
    """How far ahead of the linear response the ``order``-th harmonic appears.

    An exponential sweep at ``order`` times its frequency is the same sweep
    advanced by ``T * ln(order) / ln(f_end / f_start)``, so distortion shows
    up in the correlation as an early copy of the arrival — 94 ms ahead for
    the 2nd harmonic of the default 0.5 s, 150–6000 Hz sweep.
    """
    return chirp_seconds * np.log(order) / np.log(f_end / f_start)


def max_direct_lead(
    reference_length: int, sample_rate: int | None, f_start: float | None, f_end: float | None
) -> int | None:
    """Search depth ahead of the peak, in samples; ``None`` when unknown."""
    if sample_rate is None or f_start is None or f_end is None:
        return None
    harmonic = harmonic_lead_seconds(reference_length / sample_rate, f_start, f_end)
    return int(min(MAX_DIRECT_LEAD_SECONDS, HARMONIC_CLEARANCE * harmonic) * sample_rate)


@dataclass(frozen=True)
class Arrival:
    """One detected chirp arrival, positioned in recording samples."""

    chirp_index: int
    """Index of the chirp within the track, relative to the anchor chirp."""

    position: float
    """Sub-sample position of the first arrival, in recording samples."""

    residual: float
    """Deviation from the ideal chirp grid, in samples.

    Constant for a given speaker, so differences between speakers are exactly
    the latency differences we are after. The unknown offset between the
    microphone's clock and the stream's clock cancels out.
    """

    peak_value: float
    noise_floor: float

    @property
    def psr(self) -> float:
        """Peak-to-noise ratio, linear."""
        if self.noise_floor <= 0.0:
            return float("inf")
        return self.peak_value / self.noise_floor

    @property
    def psr_db(self) -> float:
        psr = self.psr
        return float("inf") if np.isinf(psr) else 20.0 * np.log10(max(psr, 1e-12))


def bandpass(
    signal: np.ndarray,
    *,
    sample_rate: int,
    f_start: float,
    f_end: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass limiting the signal to the sweep's own band.

    ``sosfiltfilt`` runs the filter forwards and backwards, so it introduces no
    group delay. That matters more than the filtering itself: a filter that
    shifted the signal even slightly would bias every arrival time.
    """
    nyquist = sample_rate / 2.0
    low = max(f_start / nyquist, 1e-4)
    high = min(f_end / nyquist, 0.99)
    if low >= high:
        return np.asarray(signal, dtype=np.float64)

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    # filtfilt needs a few times the filter length to settle at the edges.
    if len(signal) <= 3 * order * 2:
        return np.asarray(signal, dtype=np.float64)
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))


def whitening_filter(
    reference: np.ndarray, *, regularization: float = WHITENING_REGULARIZATION
) -> np.ndarray:
    """The reference sweep, spectrally half-whitened, ``2 * len - 1`` taps long.

    A plain matched filter returns the sweep's *autocorrelation*. An
    exponential sweep has a pink spectrum, so that pulse is dominated by the
    low end: it is wide and rippled, and the first-arrival walk can stop on a
    ripple of its rising flank. On the simulator that is a ~0.2 ms bias,
    whether from noise or from a floor or table bounce 0.3–1 ms behind the
    direct sound.

    Dividing by the sweep's power spectrum (Farina's inverse filter) flattens
    the pulse and removes the bias, but it also lifts the noise in the band's
    weak end and costs ~5 dB of reach. Dividing by the *magnitude* — halfway,
    in the spirit of SCOT weighting — keeps the pulse clean at the same noise
    reach as the matched filter. Measured on the simulator, see
    ``docs/methodology.md``.

    The kernel is laid out so that convolving with it and dropping the first
    ``len - 1`` outputs leaves index ``i`` scoring a chirp that starts at ``i``.
    """
    ref = np.asarray(reference, dtype=np.float64)
    length = len(ref)
    size = next_fast_len(2 * length)
    spectrum = rfft(ref, size)
    power = np.abs(spectrum) ** 2
    weights = np.sqrt(power + regularization * power.max())
    kernel = irfft(np.conj(spectrum) / weights, size)
    return np.roll(kernel, length - 1)[: 2 * length - 1]


def correlation_envelope(
    recording: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate: int | None = None,
    f_start: float | None = None,
    f_end: float | None = None,
) -> np.ndarray:
    """Correlate the recording with the whitened chirp; return the envelope.

    Index ``i`` of the result scores a chirp *starting* at sample ``i`` of the
    recording, so positions map straight back to the recording's timeline with
    no offset to remember.
    """
    rec = np.asarray(recording, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if len(rec) < len(ref):
        raise ValueError(f"recording ({len(rec)}) is shorter than reference ({len(ref)})")

    if sample_rate is not None and f_start is not None and f_end is not None:
        rec = bandpass(rec, sample_rate=sample_rate, f_start=f_start, f_end=f_end)

    kernel = whitening_filter(ref)
    start = len(ref) - 1
    correlation = fftconvolve(rec, kernel, mode="full")[start : start + len(rec) - len(ref) + 1]
    return _analytic_envelope(correlation, pad=max(len(ref), 4096))


def _analytic_envelope(signal: np.ndarray, *, pad: int) -> np.ndarray:
    """Envelope via the analytic signal, with the wraparound kept out.

    ``scipy.signal.hilbert`` is an FFT method and therefore circular: energy
    from a strong peak at one end of the array reappears at the other. On a
    clean recording that manifests as a phantom arrival at the very end, which
    is a real detection bug and not just cosmetic. Zero-padding both sides
    parks the wraparound in the padding, and rounding the total up to a fast
    FFT length makes the padding pay for itself.
    """
    total = next_fast_len(len(signal) + 2 * pad)
    padded = np.zeros(total, dtype=np.float64)
    padded[pad : pad + len(signal)] = signal
    return np.abs(hilbert(padded))[pad : pad + len(signal)]


def _parabolic_peak(envelope: np.ndarray, index: int) -> float:
    """Refine an integer peak index to sub-sample precision."""
    if index <= 0 or index >= len(envelope) - 1:
        return float(index)
    left, centre, right = envelope[index - 1], envelope[index], envelope[index + 1]
    denominator = left - 2.0 * centre + right
    if denominator == 0.0:
        return float(index)
    delta = 0.5 * (left - right) / denominator
    # A well-formed peak refines by less than half a sample; anything larger
    # means the three points are not bracketing a maximum.
    if not -1.0 < delta < 1.0:
        return float(index)
    return float(index) + float(delta)


def find_first_arrival(
    envelope: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
    noise_floor: float,
    arrival_ratio: float = DEFAULT_ARRIVAL_RATIO,
    min_psr_db: float = DEFAULT_MIN_PSR_DB,
    max_lead: int | None = None,
) -> tuple[float, float] | None:
    """Locate the first arrival inside ``envelope[start:stop]``.

    Returns ``(position, peak_value)`` in whole-envelope coordinates, or
    ``None`` when nothing in the window rises convincingly above the noise.

    We find the window's strongest point only to set a threshold, then walk
    *forward from the start of the window* to the first excursion above it.
    That is what makes a loud late reflection lose to a quieter direct sound.

    ``max_lead`` bounds how far ahead of the peak that walk may start. Without
    it, anything above the threshold anywhere earlier in the window wins —
    including a distorting speaker's harmonics, which a sweep places ahead of
    the real arrival.
    """
    stop = len(envelope) if stop is None else min(stop, len(envelope))
    start = max(start, 0)
    if stop - start < 3:
        return None

    window = envelope[start:stop]
    peak_index = int(np.argmax(window))
    peak_value = float(window[peak_index])

    if peak_value <= 0.0:
        return None
    if peak_value < detection_threshold(noise_floor, len(window), min_psr_db):
        return None

    # Never latch onto something the noise could have produced: the onset
    # threshold is a fraction of the peak, but never below what it takes to
    # call a peak real in the first place.
    threshold = max(
        peak_value * arrival_ratio,
        detection_threshold(noise_floor, len(window), min_psr_db),
    )
    first = 0 if max_lead is None else max(peak_index - max_lead, 0)
    above = np.flatnonzero(window[first : peak_index + 1] >= threshold)
    if len(above) == 0:
        return None

    # Climb from the first threshold crossing to the local maximum it belongs
    # to; the crossing itself sits on the rising flank, not at the top.
    cursor = first + int(above[0])
    while cursor + 1 <= peak_index and window[cursor + 1] >= window[cursor]:
        cursor += 1

    position = _parabolic_peak(window, cursor)
    return position + start, float(window[cursor])


def estimate_noise_floor(envelope: np.ndarray) -> float:
    """Robust noise floor of a correlation envelope.

    Arrivals are narrow spikes on an otherwise quiet envelope, so the median
    is dominated by the quiet part and is not dragged up by the signal.
    """
    if len(envelope) == 0:
        return 0.0
    return float(np.median(envelope))


def detect_arrivals(
    recording: np.ndarray,
    reference: np.ndarray,
    *,
    period_samples: float,
    sample_rate: int | None = None,
    f_start: float | None = None,
    f_end: float | None = None,
    arrival_ratio: float = DEFAULT_ARRIVAL_RATIO,
    min_psr_db: float = DEFAULT_MIN_PSR_DB,
    relative_floor: float = DEFAULT_RELATIVE_FLOOR,
) -> list[Arrival]:
    """Find every chirp arrival in a continuous recording.

    The track lays chirps on an exact grid, so once we have located any one of
    them the rest are a period apart. We anchor on the strongest arrival and
    step outwards in both directions, searching a window half a period wide
    around each expected position — generous, since the drift we are correcting
    for is sub-millisecond while the window is hundreds of milliseconds.

    Windows containing no credible arrival are skipped rather than guessed at,
    which is what makes muted rounds and dropouts show up as missing data
    instead of bad data.
    """
    if period_samples <= 0:
        raise ValueError(f"period_samples must be positive, got {period_samples}")

    envelope = correlation_envelope(
        recording, reference, sample_rate=sample_rate, f_start=f_start, f_end=f_end
    )
    noise_floor = estimate_noise_floor(envelope)
    lead = max_direct_lead(len(reference), sample_rate, f_start, f_end)

    anchor = find_first_arrival(
        envelope,
        noise_floor=noise_floor,
        arrival_ratio=arrival_ratio,
        min_psr_db=min_psr_db,
        max_lead=lead,
    )
    if anchor is None:
        return []

    # The global first arrival may belong to a quiet speaker; re-anchor on the
    # strongest point so the grid is pinned to the most reliable detection.
    anchor_position = _parabolic_peak(envelope, int(np.argmax(envelope)))

    half_window = period_samples / 2.0
    first_index = int(np.floor(-anchor_position / period_samples))
    last_index = int(np.ceil((len(envelope) - anchor_position) / period_samples))

    arrivals: list[Arrival] = []
    for index in range(first_index, last_index + 1):
        centre = anchor_position + index * period_samples
        start = int(round(centre - half_window))
        stop = int(round(centre + half_window))
        if stop <= 0 or start >= len(envelope):
            continue

        found = find_first_arrival(
            envelope,
            start=max(start, 0),
            stop=min(stop, len(envelope)),
            noise_floor=noise_floor,
            arrival_ratio=arrival_ratio,
            min_psr_db=min_psr_db,
            max_lead=lead,
        )
        if found is None:
            continue

        position, peak_value = found
        arrivals.append(
            Arrival(
                chirp_index=index,
                position=position,
                residual=position - anchor_position - index * period_samples,
                peak_value=peak_value,
                noise_floor=noise_floor,
            )
        )

    return _drop_disproportionate(arrivals, relative_floor)


def _drop_disproportionate(arrivals: list[Arrival], relative_floor: float) -> list[Arrival]:
    """Discard candidates far weaker than the rest of the recording.

    Chirps from real speakers land within a modest range of each other, while
    filter ringing and numerical residue sit orders of magnitude down. An
    absolute noise-floor test cannot separate them on a recording that is
    digitally silent in places, because there the floor collapses to machine
    epsilon and every artefact clears it.

    A speaker quiet enough to be cut here would not have produced a
    trustworthy reading anyway, and dropping it surfaces as "no signal" rather
    than as a wrong number.
    """
    if len(arrivals) < 2 or relative_floor <= 0:
        return arrivals

    median_peak = float(np.median([a.peak_value for a in arrivals]))
    if median_peak <= 0.0:
        return arrivals

    cutoff = median_peak / relative_floor
    return [a for a in arrivals if a.peak_value >= cutoff]
