"""Analytic Gaussian HJM forward-bond log-variance under the S-forward measure.

For independent Brownian factors with sigma_i(t,u)=a_i exp(-b_i(u-t)),
the forward bond F(t;S,T)=P(t,T)/P(t,S), T>S, has lognormal volatility
v_i(t)=integral_S^T sigma_i(t,u) du. Its total log-variance over [0,S]
is sum_i integral_0^S v_i(t)^2 dt.

This module provides an independent deterministic reference for simulations.
"""
from __future__ import annotations

import numpy as np


def forward_bond_logvariance(expiry, maturity, amplitudes, decays):
    """Integrated log-variance of P(S,T), divided by P(S,S)=1.

    Parameters
    ----------
    expiry : float
        Option expiry S, in years; must be non-negative.
    maturity : float
        Bond maturity T >= S, in years.
    amplitudes, decays : array_like
        Independent factor amplitude a_i and nonnegative decay b_i.
    """
    s, t = float(expiry), float(maturity)
    a = np.atleast_1d(np.asarray(amplitudes, dtype=float))
    b = np.atleast_1d(np.asarray(decays, dtype=float))
    if not np.isfinite(s) or not np.isfinite(t) or s < 0 or t < s:
        raise ValueError("Require finite 0 <= expiry <= maturity")
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError("Factors must be matching nonempty one-dimensional arrays")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(b < 0):
        raise ValueError("Amplitudes finite and decays finite/nonnegative")
    if s == 0 or t == s:
        return 0.0
    result = 0.0
    for ai, bi in zip(a, b):
        if bi == 0:
            result += (ai * (t - s)) ** 2 * s
        else:
            # Integral_S^T exp(-b(u-r))du =
            # exp(-b(S-r)) * [-expm1(-b(T-S))]/b.
            # Integrate its square over r in [0,S].
            span = -np.expm1(-bi * (t - s)) / bi
            time_integral = -np.expm1(-2 * bi * s) / (2 * bi)
            result += ai * ai * span * span * time_integral
    return float(result)


def forward_bond_mc(expiry, maturity, amplitudes, decays,
                    initial_forward_bond, n_paths=10000, seed=1729):
    """Simulate terminal forward-bond price under S-forward measure.

    Exact terminal lognormal draw, not Euler path discretisation.
    Returns paths for independent verification of martingale expectations.
    """
    f0 = float(initial_forward_bond)
    if not np.isfinite(f0) or f0 <= 0:
        raise ValueError("Initial forward bond must be positive and finite")
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    variance = forward_bond_logvariance(expiry, maturity, amplitudes, decays)
    z = np.random.default_rng(seed).standard_normal(n_paths)
    return f0 * np.exp(-0.5 * variance + np.sqrt(variance) * z)


def martingale_zscore(samples, initial_forward_bond):
    """Studentised sample-mean deviation from the forward-measure expectation."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("At least two finite observations required")
    stderr = float(np.std(x, ddof=1) / np.sqrt(x.size))
    if stderr == 0:
        return 0.0 if np.mean(x) == initial_forward_bond else float("inf")
    return float((np.mean(x) - initial_forward_bond) / stderr)
