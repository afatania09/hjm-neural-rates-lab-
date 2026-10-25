"""Exact S-forward-measure Gaussian HJM forward-bond paths.

For t <= S < T, F(t;S,T)=P(t,T)/P(t,S) is a martingale under
the S-forward measure. For sigma_i(t,u)=a_i exp(-b_i(u-t)),
dF/F = -sum_i [integral_S^T sigma_i(t,u)du] dW_i^S(t).
This module integrates the deterministic log-variance *exactly* on each
time interval and therefore avoids Euler discretisation error for F.
"""
from dataclasses import dataclass
import numpy as np
from .bond_options import forward_bond_log_variance


@dataclass(frozen=True)
class ForwardBondPaths:
    times: np.ndarray
    prices: np.ndarray
    integrated_variances: np.ndarray


def simulate_forward_bond_paths(model, curve, expiry, maturity, times,
                                n_paths=10000, seed=2026):
    """Simulate P(t,T)/P(t,S) under the S-forward measure.

    Returns prices with shape (n_paths, n_times). Independent Gaussian
    increments are scaled using exact interval integrated variance.
    """
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("times must be a finite 1D grid with >=2 entries")
    if grid[0] != 0 or np.any(np.diff(grid) <= 0) or grid[-1] > expiry:
        raise ValueError("times must start at zero, increase, and end by expiry")
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer >=2")
    full_var = forward_bond_log_variance(model, expiry, maturity)
    # V(t) = sum_i A_i^2 int_0^t exp(-2 b_i (S-u))du,
    # with A_i = a_i (1-exp(-b_i(T-S)))/b_i.
    delta = maturity - expiry
    cumulative = np.zeros(len(grid), dtype=float)
    for a, b in zip(model.amplitudes, model.decays):
        if b == 0:
            cumulative += (a * delta)**2 * grid
        else:
            exposure_at_expiry = a * (-np.expm1(-b * delta)) / b
            cumulative += (exposure_at_expiry**2
                           * np.exp(-2 * b * expiry)
                           * np.expm1(2 * b * grid) / (2 * b))
    increments = np.diff(cumulative)
    if np.any(increments < -1e-12):
        raise ArithmeticError("Integrated variance is not monotonic")
    increments = np.maximum(increments, 0)
    if grid[-1] == expiry and not np.isclose(cumulative[-1], full_var, rtol=1e-10, atol=1e-13):
        raise ArithmeticError("Cumulative variance inconsistent with analytical formula")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, len(increments)))
    log_relative = np.cumsum(-0.5 * increments[None, :] +
                             z * np.sqrt(increments)[None, :], axis=1)
    f0 = float(curve.discount(maturity)) / float(curve.discount(expiry))
    if not np.isfinite(f0) or f0 <= 0:
        raise ValueError("Initial forward bond price must be positive and finite")
    paths = f0 * np.exp(np.column_stack((np.zeros(n_paths), log_relative)))
    return ForwardBondPaths(grid, paths, cumulative)
