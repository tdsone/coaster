"""Total-variation signatures of read-induced coverage.

A coverage profile from N reads with insert lengths drawn from a
distribution with mean E[L] satisfies the strict identity

    TV(cov)  =  sum over read events of |±1 contribution|  ≤  2 N

in count units (each read contributes a +1 at its start and a -1 at its end;
when starts and ends fall on the same position the contributions add
algebraically, not in absolute value, so realized TV is less than or equal
to 2N). Because the total mass of the profile is sum(cov) = N · E[L], the
dimensionless quantity

    tv_ratio  =  TV(cov) · E[L]  /  (2 · sum(cov))

is bounded above by 1 for any read-aggregation process. Profiles with
tv_ratio > 1 cannot have been produced by reads with the given length
distribution at all. Within the bound, smoother profiles (predicted means,
oversmoothed neural outputs) have low tv_ratio; profiles whose edges are
fully resolved (single-cell, sparse reads) have tv_ratio near 1.

For discriminating predicted vs real coverage, the most useful comparison
is tv_ratio(pred) against tv_ratio(real) for the same window — see the
driver, which logs both.
"""
from __future__ import annotations

import numpy as np


def total_variation(cov: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(np.asarray(cov, dtype=np.float64)))))


def tv_ratio(cov: np.ndarray, lengths: np.ndarray) -> float:
    """TV(cov) · E[L] / (2 · sum(cov)). Bounded above by 1 for any
    profile produced by read aggregation."""
    cov = np.asarray(cov, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.size == 0 or cov.sum() <= 0:
        return float("nan")
    mean_len = float(lengths.mean())
    return total_variation(cov) * mean_len / (2.0 * float(cov.sum()) + 1e-12)


def tv_excess(cov: np.ndarray, lengths: np.ndarray) -> float:
    """max(0, tv_ratio - 1): strictly positive only for profiles whose TV
    exceeds the read-aggregation upper bound, i.e. impossible by any read
    configuration with the given length distribution."""
    r = tv_ratio(cov, lengths)
    if np.isnan(r):
        return float("nan")
    return max(0.0, r - 1.0)


# Legacy alias kept so older scripts importing max_slope_excess still resolve.
max_slope_excess = tv_excess
