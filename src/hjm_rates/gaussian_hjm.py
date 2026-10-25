"""Gaussian HJM with deterministic, exponentially decaying maturity volatility.

Under the risk-neutral money-market measure:
df(t,T) = alpha(t,T) dt + sum_i sigma_i(t,T) dW_i(t),
alpha(t,T) = sum_i sigma_i(t,T) integral_t^T sigma_i(t,u) du.

This is an educational fixed-maturity-grid Euler simulator, not a
production rates pricing or hedging system.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GaussianHJM:
    amplitudes: tuple[float, ...] = (0.01,)
    decays: tuple[float, ...] = (0.15,)

    def __post_init__(self):
        if len(self.amplitudes) == 0 or len(self.amplitudes) != len(self.decays):
            raise ValueError("Need matching nonempty factor parameters")
        if not np.all(np.isfinite(self.amplitudes)) or not np.all(np.isfinite(self.decays)):
            raise ValueError("Factor parameters must be finite")
        if any(x < 0 for x in self.decays):
            raise ValueError("Decays must be nonnegative")

    @property
    def n_factors(self):
        return len(self.amplitudes)

    def volatility(self, time, maturities):
        T = np.asarray(maturities, dtype=float)
        if np.any(T < time):
            raise ValueError("Maturities cannot precede observation time")
        tau = T - time
        return np.stack([a * np.exp(-b * tau) for a, b in zip(self.amplitudes, self.decays)], axis=-1)

    def integrated_volatility(self, time, maturities):
        T = np.asarray(maturities, dtype=float)
        if np.any(T < time):
            raise ValueError("Maturities cannot precede observation time")
        tau = T - time
        vals = [a * (-np.expm1(-b * tau)) / b if b > 0 else a * tau
                for a, b in zip(self.amplitudes, self.decays)]
        return np.stack(vals, axis=-1)

    def drift(self, time, maturities):
        return np.sum(self.volatility(time, maturities) *
                      self.integrated_volatility(time, maturities), axis=-1)


def simulate_forward_paths(model, initial_curve, maturities, times, n_paths=1000, seed=7):
    """Euler simulation of f(t,T) on fixed absolute maturity nodes.

    Returns paths with shape (n_paths, n_times, n_maturities).
    Nodes with T<t are NaN, because their forwards are no longer defined.
    """
    T = np.asarray(maturities, dtype=float)
    ts = np.asarray(times, dtype=float)
    if T.ndim != 1 or ts.ndim != 1 or len(T) < 2 or len(ts) < 2:
        raise ValueError("Need one-dimensional maturity and time grids")
    if not np.all(np.isfinite(T)) or not np.all(np.isfinite(ts)):
        raise ValueError("Non-finite grids")
    if np.any(np.diff(T) <= 0) or np.any(np.diff(ts) <= 0):
        raise ValueError("Grids must be strictly increasing")
    if ts[0] != 0 or T[0] < 0 or ts[-1] > T[-1]:
        raise ValueError("Time grid must start at zero and end by last maturity")
    if not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be positive integer")
    rng = np.random.default_rng(seed)
    paths = np.full((n_paths, len(ts), len(T)), np.nan)
    paths[:, 0, :] = initial_curve.forward(T)
    for j, (t, nxt) in enumerate(zip(ts[:-1], ts[1:])):
        dt = nxt - t
        live = T >= nxt
        if not np.any(live):
            continue
        z = rng.normal(size=(n_paths, model.n_factors))
        sigma = model.volatility(t, T[live])
        drift = model.drift(t, T[live])
        paths[:, j + 1, live] = (paths[:, j, live] + drift[None, :] * dt
                                  + np.sqrt(dt) * (z @ sigma.T))
    return paths
