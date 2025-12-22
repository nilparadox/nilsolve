from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PeriodMetrics:
    n_peaks: int
    period_est: Optional[float]
    freq_est: Optional[float]


def estimate_period_from_peaks(t: np.ndarray, x: np.ndarray) -> PeriodMetrics:
    """
    Estimate oscillation period by detecting local maxima in x(t).

    Rules:
    - Peaks: x[i-1] < x[i] > x[i+1]
    - Period estimate: mean of successive peak-to-peak time differences
    - If <2 peaks -> period_est=None

    This is intended for robust, quick diagnostics in scan workflows.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("t and x must be 1D arrays")
    if t.size != x.size:
        raise ValueError("t and x must have same length")
    if t.size < 5:
        return PeriodMetrics(n_peaks=0, period_est=None, freq_est=None)

    # basic finite check
    m = np.isfinite(t) & np.isfinite(x)
    t = t[m]
    x = x[m]
    if t.size < 5:
        return PeriodMetrics(n_peaks=0, period_est=None, freq_est=None)

    # ensure time increasing (not strictly required, but expected)
    # if not increasing, sort by time
    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t = t[idx]
        x = x[idx]

    # local maxima
    xm1 = x[:-2]
    x0 = x[1:-1]
    xp1 = x[2:]
    peaks_mask = (x0 > xm1) & (x0 > xp1)
    peak_idx = np.where(peaks_mask)[0] + 1  # shift for middle indexing

    n_peaks = int(peak_idx.size)
    if n_peaks < 2:
        return PeriodMetrics(n_peaks=n_peaks, period_est=None, freq_est=None)

    peak_times = t[peak_idx]
    dts = np.diff(peak_times)
    # discard non-positive diffs (shouldn't happen, but safe)
    dts = dts[dts > 0]
    if dts.size == 0:
        return PeriodMetrics(n_peaks=n_peaks, period_est=None, freq_est=None)

    period = float(np.mean(dts))
    freq = float(1.0 / period) if period > 0 else None
    return PeriodMetrics(n_peaks=n_peaks, period_est=period, freq_est=freq)
