"""Per-window scorers for how plausibly a coverage profile could have been
produced by a finite set of reads with a given length distribution.

Used to evaluate whether a discriminative coverage predictor (e.g. Yorzoi)
emits profiles that live on the realizable manifold of read-induced
coverages, and whether departure from that manifold correlates with
prediction error.
"""
from .kernel import survival_kernel
from .deconvolve import nnls_deconvolve, unconstrained_deconvolve
from .bounds import max_slope_excess, tv_excess, tv_ratio
from .spectrum import high_freq_energy_fraction
from .per_window import score_window

__all__ = [
    "survival_kernel",
    "unconstrained_deconvolve",
    "nnls_deconvolve",
    "tv_ratio",
    "tv_excess",
    "max_slope_excess",
    "high_freq_energy_fraction",
    "score_window",
]
