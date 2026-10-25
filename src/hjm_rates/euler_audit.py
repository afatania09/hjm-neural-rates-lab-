"""Independent Ho-Lee (constant-volatility) HJM Euler convergence benchmark.

For sigma(t,T)=a, f(t,T)=f(0,T)+a^2(T*t-t^2/2)+a*W_t.
The existing Euler engine uses left-endpoint drift, which produces a
deterministic positive bias of a^2*t*dt/2 for uniform steps.
"""
from dataclasses import dataclass
import numpy as np
from .gaussian_hjm import GaussianHJM, simulate_forward_paths
from .curves import FlatForwardCurve


@dataclass(frozen=True)
class EulerAudit:
    dt: float
    sample_mean: float
    theoretical_mean: float
    predicted_euler_mean: float
    sampling_se: float
    predicted_drift_bias: float
    observed_minus_euler_mean: float
    n_paths: int


def ho_lee_forward_audit(volatility=0.025, rate=0.03, maturity=5.0,
                         observation_time=2.0, n_steps=40, n_paths=20000,
                         seed=2026):
    """Audit a fixed absolute maturity T under a single constant-vol HJM factor.

    Uses the actual simulate_forward_paths implementation. Because its grid
    must contain at least two maturity nodes, [observation_time, maturity]
    is supplied, and the last node is inspected.
    """
    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be nonnegative and finite")
    if not np.isfinite(rate) or not np.isfinite(maturity) or not np.isfinite(observation_time):
        raise ValueError("non-finite input")
    if observation_time <= 0 or maturity <= observation_time:
        raise ValueError("Require 0 < observation_time < maturity")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be >= 2")
    dt = observation_time / n_steps
    times = np.linspace(0, observation_time, n_steps + 1)
    model = GaussianHJM((volatility,), (0.0,))
    paths = simulate_forward_paths(model, FlatForwardCurve(rate),
                                   np.array([observation_time, maturity]),
                                   times, n_paths=n_paths, seed=seed)
    sample = paths[:, -1, -1]
    exact_mean = rate + volatility**2 * (
        maturity * observation_time - 0.5 * observation_time**2)
    # Sum_j a^2 (T-t_j) dt = exact integral + a^2*t*dt/2.
    bias = 0.5 * volatility**2 * observation_time * dt
    euler_mean = exact_mean + bias
    se = float(np.std(sample, ddof=1) / np.sqrt(n_paths))
    return EulerAudit(dt, float(np.mean(sample)), exact_mean, euler_mean,
                      se, bias, float(np.mean(sample) - euler_mean), n_paths)
