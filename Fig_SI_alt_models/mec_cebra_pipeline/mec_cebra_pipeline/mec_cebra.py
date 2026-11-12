"""Load the Figure 3 MEC spikes and bin them for CEBRA's high-level API."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.io import loadmat


def load_mec_spikes(
    path: str | Path, gap_seconds: float = 0.02
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Read MEC spike times and uninterrupted intervals from ``27765_2.mat``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the original MAT file used for C3PO Figure 3.
    gap_seconds : float, default=0.02
        Minimum gap between behavior timestamps that starts a new interval.

    Returns
    -------
    spike_times : list of numpy.ndarray
        Spike times in seconds, each shaped (n_spikes_for_unit,). The order
        of all 460 entries in ``Dsession.units.mec`` is preserved.
    intervals : numpy.ndarray, shape (n_intervals, 2)
        First and last behavior timestamps of each uninterrupted interval,
        in seconds. Interval ends are not extrapolated.
    """
    session = loadmat(path, simplify_cells=True)["Dsession"]
    spike_times = [
        np.asarray(unit["spikeTimes"], dtype=np.float64).ravel()
        for unit in session["units"]["mec"]
    ]
    time = np.asarray(session["t"], dtype=np.float64).ravel()
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Behavior timestamps must be finite and increasing.")
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        raise ValueError("gap_seconds must be positive and finite.")
    cuts = np.r_[0, np.flatnonzero(np.diff(time) >= gap_seconds) + 1, len(time)]
    intervals = np.column_stack((time[cuts[:-1]], time[cuts[1:] - 1]))
    return spike_times, intervals


def bin_spikes(
    spike_times: Sequence[ArrayLike],
    intervals: ArrayLike,
    bin_seconds: float = 1 / 120,
    min_bins: int = 40,
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float64]]]:
    """Bin spikes separately within each interval, without smoothing or scaling.

    Parameters
    ----------
    spike_times : sequence of array_like
        One array per unit, shaped (n_spikes_for_unit,), in seconds.
    intervals : array_like, shape (n_intervals, 2)
        Ordered, nonoverlapping start/stop times in seconds. Bins include
        their left edge and exclude their right edge.
    bin_seconds : float, default=1/120
        Bin width in seconds, matching CEBRA's 120 Hz Neuropixels input.
    min_bins : int, default=40
        Skip intervals shorter than the encoder's 40-bin receptive field.
        Incomplete final bins are also omitted.

    Returns
    -------
    count_blocks : list of numpy.ndarray
        Raw float32 spike counts, each shaped (n_bins_in_interval, n_units).
        Silent units are retained. Gaps contain no rows.
    time_blocks : list of numpy.ndarray
        Corresponding bin centers in seconds, each shaped (n_bins_in_interval,).
    """
    bounds = np.asarray(intervals, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all():
        raise ValueError("intervals must be a finite array shaped (n_intervals, 2).")
    if np.any(bounds[:, 1] < bounds[:, 0]) or np.any(bounds[1:, 0] < bounds[:-1, 1]):
        raise ValueError("Intervals must be ordered and nonoverlapping.")
    if not np.isfinite(bin_seconds) or bin_seconds <= 0 or min_bins < 1:
        raise ValueError("bin_seconds must be positive and finite; min_bins >= 1.")
    spikes = [
        np.sort(np.asarray(unit, dtype=np.float64).ravel()) for unit in spike_times
    ]
    if not spikes or any(not np.isfinite(unit).all() for unit in spikes):
        raise ValueError("Provide at least one unit, with finite spike timestamps.")

    count_blocks, time_blocks = [], []
    for start, stop in bounds:
        n_bins = int(np.floor((stop - start) / bin_seconds + 1e-8))
        if n_bins < min_bins:
            continue
        edges = start + np.arange(n_bins + 1) * bin_seconds
        counts = np.column_stack(
            [np.diff(np.searchsorted(unit, edges, side="left")) for unit in spikes]
        ).astype(np.float32)
        count_blocks.append(counts)
        time_blocks.append(edges[:-1] + bin_seconds / 2)
    if not count_blocks:
        raise ValueError("No interval contains min_bins complete bins.")
    return count_blocks, time_blocks
