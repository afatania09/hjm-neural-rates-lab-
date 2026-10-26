"""Exact finite-grid joint Ho-Lee short-rate, discount, and bond paths.

Constant HJM volatility a, initially flat instantaneous forwards r0.
The Brownian time integral on each interval is sampled exactly using a
Brownian bridge, avoiding time-step bias in the money-market numeraire.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HoLeePaths:
    times: np.ndarray
    short_rates: np.ndarray
    discount_factors: np.ndarray
    bond_prices: np.ndarray


def simulate_ho_lee_paths(rate, volatility, maturity, times, n_paths=10000, seed=2026):
    """Return risk-neutral Q paths with shapes (n_paths, n_times).

    Bond prices P(t,T) are obtained by analytically integrating the
    exact HJM forward curve. Maturity must exceed final observation time.
    """
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("times must be finite one-dimensional array")
    if grid[0] != 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("times must start at zero and increase strictly")
    if not np.isfinite(rate) or not np.isfinite(volatility) or volatility < 0:
        raise ValueError("rate and nonnegative volatility must be finite")
    if not np.isfinite(maturity) or maturity <= grid[-1]:
        raise ValueError("maturity must exceed last observation time")
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    rng = np.random.default_rng(seed)
    dt = np.diff(grid)
    z1 = rng.standard_normal((n_paths, len(dt)))
    z2 = rng.standard_normal((n_paths, len(dt)))
    d_w = z1 * np.sqrt(dt)[None, :]
    w = np.column_stack((np.zeros(n_paths), np.cumsum(d_w, axis=1)))
    # Integral_{t_j}^{t_{j+1}} W_s ds:
    # dt*(W_j+W_{j+1})/2 + sqrt(dt^3/12)*Z_independent.
    areas = dt[None, :] * (w[:, :-1] + w[:, 1:]) / 2 + (
        dt**1.5 / np.sqrt(12))[None, :] * z2
    integrated_w = np.column_stack((np.zeros(n_paths), np.cumsum(areas, axis=1)))
    t = grid[None, :]
    short_rates = rate + 0.5 * volatility**2 * t**2 + volatility * w
    log_discount = -(rate * t + volatility**2 * t**3 / 6 + volatility * integrated_w)
    discount = np.exp(log_discount)
    length = maturity - t
    # int_t^T f(t,u)du = r0 L + a^2 [t(T^2-t^2)/2 - t^2 L/2] + a W_t L
    integrated_forward = (rate * length + volatility**2 * (
        t * (maturity**2 - t**2) / 2 - t**2 * length / 2)
        + volatility * w * length)
    bonds = np.exp(-integrated_forward)
    return HoLeePaths(grid, short_rates, discount, bonds)
