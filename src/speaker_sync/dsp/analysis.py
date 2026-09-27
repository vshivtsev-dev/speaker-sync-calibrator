"""Robust reduction of repeated measurements.

Every speaker is measured over several chirps. A single bad reading — someone
coughs, a door closes, a packet arrives late — should not move the result, so
we reduce with a median and reject outliers by median absolute deviation
rather than averaging everything together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Scale factor making the MAD a consistent estimator of the standard deviation
# for normally distributed data.
MAD_TO_SIGMA = 1.4826

DEFAULT_MAX_DEVIATIONS = 3.0
DEFAULT_MIN_SAMPLES = 3


@dataclass(frozen=True)
class RobustEstimate:
    """Outlier-resistant summary of repeated readings of the same quantity."""

    value: float
    spread: float
    """MAD-derived sigma of the kept readings; a stability indicator."""

    kept: int
    rejected: int
    samples: tuple[float, ...]

    @property
    def total(self) -> int:
        return self.kept + self.rejected

    @property
    def is_reliable(self) -> bool:
        return self.kept >= DEFAULT_MIN_SAMPLES


def robust_estimate(
    values: "np.ndarray | list[float] | tuple[float, ...]",
    *,
    max_deviations: float = DEFAULT_MAX_DEVIATIONS,
) -> RobustEstimate | None:
    """Median of ``values`` after discarding MAD outliers.

    Returns ``None`` for an empty input — an absent measurement is reported as
    absent rather than substituted with a default.
    """
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return None

    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    sigma = mad * MAD_TO_SIGMA

    if sigma > 0.0:
        keep_mask = np.abs(data - median) <= max_deviations * sigma
        # An all-false mask cannot happen (the median itself always passes),
        # but guard anyway rather than produce an empty median.
        kept = data[keep_mask] if keep_mask.any() else data
    else:
        kept = data

    return RobustEstimate(
        value=float(np.median(kept)),
        spread=float(np.median(np.abs(kept - np.median(kept))) * MAD_TO_SIGMA),
        kept=int(len(kept)),
        rejected=int(len(data) - len(kept)),
        samples=tuple(float(v) for v in data),
    )


def samples_to_ms(samples: float, sample_rate: int) -> float:
    return samples * 1000.0 / sample_rate


def ms_to_samples(milliseconds: float, sample_rate: int) -> float:
    return milliseconds * sample_rate / 1000.0
