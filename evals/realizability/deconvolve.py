"""Deconvolution of a coverage profile against an insert-length kernel.

Given a predicted coverage profile `cov` of length N and a survival-function
kernel `K` of length Kmax (K[0] = 1), we ask: does there exist a non-negative
read-start density `rho` such that K * rho approximates `cov`? If not, the
profile is inconsistent with reads drawn from the distribution that produced
`K`.

Because K[0] = P(L > 0) = 1, the convolution `cov = K * rho` is a banded
lower-triangular linear system that we can solve in O(N * Kmax) by inverse
filtering (`scipy.signal.lfilter([1], K, cov)`). The recovered `rho` is exact
in the limit of no boundary effects; the first Kmax positions are corrupted
because reads that started before the window aren't represented in `rho`,
so they show up as negative read-start mass at the 5' edge. The reported
`neg_mass_fraction` excludes that boundary band.

A true non-negative least-squares solve via `lsq_linear` is also available
behind `method="nnls"` for cases where the inverse filter's interior
negative mass needs to be confirmed (rather than attributed to noise). It
is ~1000x slower per window at N=3000.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.signal as sps
from scipy.optimize import lsq_linear


@dataclass
class DeconvResult:
    rho: np.ndarray            # (N,) recovered read-start density (may be negative for "unconstrained")
    reconstruction: np.ndarray  # (N,) K * rho
    residual_l2: float         # ||cov - K*rho||_2 / ||cov||_2  (≈ 0 for inverse filter)
    neg_mass_fraction: float   # ||rho^-||_1 / ||rho||_1 over the interior, excluding boundary band


def _interior_neg_fraction(rho: np.ndarray, boundary: int) -> float:
    """Negative-mass fraction over the interior, excluding the first `boundary` positions
    (corrupted by reads originating before the window). NaN if interior is empty."""
    if rho.size <= boundary:
        return float("nan")
    interior = rho[boundary:]
    total = float(np.sum(np.abs(interior))) + 1e-12
    return float(np.sum(np.clip(-interior, 0.0, None)) / total)


def unconstrained_deconvolve(coverage: np.ndarray, kernel: np.ndarray) -> DeconvResult:
    """Solve cov = K * rho exactly via inverse IIR filtering. Fast (microseconds).

    rho may go negative — that is the realizability signal. The interior
    negative-mass fraction is the implausibility score; pure read aggregation
    cannot produce it.
    """
    cov = np.asarray(coverage, dtype=np.float64)
    K = np.asarray(kernel, dtype=np.float64)
    if K.size == 0 or K[0] == 0.0:
        raise ValueError("kernel must have K[0] > 0")
    rho = sps.lfilter(np.array([1.0]), K, cov)
    # Reconstruction error should be ~ machine epsilon for the inverse filter,
    # but report it anyway as a sanity check.
    recon = np.convolve(rho, K, mode="full")[: cov.size]
    denom = np.linalg.norm(cov) + 1e-12
    residual = float(np.linalg.norm(cov - recon) / denom)
    neg_frac = _interior_neg_fraction(rho, boundary=K.size)
    return DeconvResult(
        rho=rho,
        reconstruction=recon,
        residual_l2=residual,
        neg_mass_fraction=neg_frac,
    )


def _banded_toeplitz(kernel: np.ndarray, n: int) -> sp.csr_matrix:
    kmax = kernel.size
    diagonals = [np.full(n - d, kernel[d]) for d in range(min(kmax, n)) if kernel[d] != 0.0]
    offsets = [-d for d in range(min(kmax, n)) if kernel[d] != 0.0]
    return sp.diags(diagonals, offsets=offsets, shape=(n, n), format="csr")


def nnls_deconvolve(
    coverage: np.ndarray,
    kernel: np.ndarray,
    *,
    compute_neg_mass: bool = True,
    max_iter: int | None = None,
) -> DeconvResult:
    """True non-negative least-squares deconvolution via `lsq_linear`.

    Slow (seconds per window at N=3000) and only needed when the unconstrained
    inverse filter shows ambiguous interior negative mass. Reports the L2
    residual after projecting onto rho >= 0.
    """
    cov = np.asarray(coverage, dtype=np.float64)
    K = np.asarray(kernel, dtype=np.float64)
    n = cov.size
    A = _banded_toeplitz(K, n)

    sol = lsq_linear(
        A, cov, bounds=(0.0, np.inf), method="trf",
        max_iter=max_iter, tol=1e-8,
    )
    rho = sol.x
    recon = A @ rho
    denom = np.linalg.norm(cov) + 1e-12
    residual = float(np.linalg.norm(cov - recon) / denom)

    neg_frac = float("nan")
    if compute_neg_mass:
        sol_unc = lsq_linear(A, cov, bounds=(-np.inf, np.inf), method="trf", tol=1e-8)
        x = sol_unc.x
        total = np.sum(np.abs(x)) + 1e-12
        neg_frac = float(np.sum(np.clip(-x, 0, None)) / total)

    return DeconvResult(
        rho=rho,
        reconstruction=recon,
        residual_l2=residual,
        neg_mass_fraction=neg_frac,
    )
