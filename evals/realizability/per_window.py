"""Top-level per-window scoring entry point.

`score_window(cov, lengths)` returns a dict of realizability scalars for a
single coverage profile against its empirical insert-length distribution.
The driver calls this twice per test window — once on the predicted
coverage and once on the matching real coverage — to subtract off the
finite-sample noise floor and isolate predictor-specific implausibility.

The default deconvolution is the fast unconstrained inverse filter
(`method="lfilter"`); pass `method="nnls"` for a true non-negative solve
(~1000x slower).
"""
from __future__ import annotations

import numpy as np

from .bounds import tv_excess, tv_ratio
from .deconvolve import nnls_deconvolve, unconstrained_deconvolve
from .kernel import survival_kernel
from .spectrum import high_freq_energy_fraction


def score_window(
    cov: np.ndarray,
    lengths: np.ndarray,
    *,
    hf_cutoff_bp: float = 50.0,
    method: str = "lfilter",
) -> dict[str, float]:
    """Score a single coverage profile for read-realizability.

    Args:
        cov: coverage profile, shape (L,) — either a predicted profile or a
            real one. The scoring is identical and treats the input as a
            candidate to be explained by reads of length distribution
            `lengths`.
        lengths: empirical insert lengths for this window, shape (n_reads,).
        hf_cutoff_bp: period (bp) below which spectral energy counts as
            high-frequency.
        method: "lfilter" for fast unconstrained inverse filter (default),
            "nnls" for slow true non-negative least squares.

    Returns:
        Dict of named scalars; all floats, NaN where undefined.
    """
    cov = np.asarray(cov, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)

    if lengths.size == 0:
        return dict(
            neg_mass_fraction=float("nan"),
            deconv_residual=float("nan"),
            tv_ratio=float("nan"),
            tv_excess=float("nan"),
            hf_energy_fraction=float("nan"),
        )

    K = survival_kernel(lengths)
    if method == "lfilter":
        dec = unconstrained_deconvolve(cov, K)
    elif method == "nnls":
        dec = nnls_deconvolve(cov, K, compute_neg_mass=True)
    else:
        raise ValueError(f"unknown method: {method!r}")

    return dict(
        neg_mass_fraction=dec.neg_mass_fraction,
        deconv_residual=dec.residual_l2,
        tv_ratio=tv_ratio(cov, lengths),
        tv_excess=tv_excess(cov, lengths),
        hf_energy_fraction=high_freq_energy_fraction(cov, hf_cutoff_bp),
    )
