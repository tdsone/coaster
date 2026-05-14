"""Insert-length distribution → convolution kernel.

A coverage profile from N reads with start positions rho[s] and i.i.d. insert
lengths drawn from distribution pi is

    cov[i] = sum_{s} rho[s] * I(s <= i < s + L_s)
           = sum_{s} rho[s] * K[i - s]                  (in expectation)

where K[d] = P(L > d) is the survival function of pi. K is monotonically
non-increasing, K[0] = 1, and K[d] = 0 for d >= L_max. This module builds K
from an empirical sample of insert lengths.
"""
from __future__ import annotations

import numpy as np


def survival_kernel(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    """Empirical survival kernel K[d] = P(L > d) from observed insert lengths.

    Args:
        lengths: 1-D int array of observed insert lengths (>= 1).
        max_len: Truncate the kernel after this many positions. Defaults to
            max(lengths) so the kernel has support [0, max_len).

    Returns:
        float64 array of shape (max_len,) with K[0] = 1 and K[max_len-1] >= 0.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        raise ValueError("lengths is empty")
    if max_len is None:
        max_len = int(lengths.max())
    if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}")

    # CDF via cumulative histogram, then K[d] = 1 - P(L <= d) = P(L > d).
    counts = np.bincount(np.clip(lengths, 0, max_len), minlength=max_len + 1)
    cdf = np.cumsum(counts) / lengths.size
    # P(L > d) for d = 0, ..., max_len - 1
    return (1.0 - cdf[:max_len]).astype(np.float64)
