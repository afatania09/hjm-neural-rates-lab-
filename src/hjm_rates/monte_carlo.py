"""Independent S-forward-measure Monte Carlo benchmark for Gaussian HJM bond options.

This simulates the *terminal forward bond price* exactly in distribution,
not the full HJM forward curve and not a time-discretised Euler path.
It is intended to test statistical estimators and compare against the
analytical Black-style bond-option benchmark.
"""
from dataclasses import dataclass
from statistics import NormalDist
import numpy as np
from .bond_options import forward_bond_log_variance


@dataclass(frozen=True)
class MonteCarloEstimate:
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    confidence: float


def bond_option_forward_measure_mc(
    model, curve, expiry, maturity, strike, option_type="call",
    n_paths=100_000, seed=2026, confidence=0.95, antithetic=True,
):
    """Risk-neutral time-zero bond option price via S-forward measure.

    F_S = F_0 exp(-V/2 + sqrt(V) Z), Z standard normal,
    price = P(0,S) E^S[max(+/-(F_S-K),0)].

    Confidence interval uses the ordinary sample-variance normal approximation;
    it is not a guaranteed finite-sample coverage statement.
    """
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between zero and one")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be call or put")
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be positive and finite")
    variance = forward_bond_log_variance(model, expiry, maturity)
    p_s = float(curve.discount(expiry))
    p_t = float(curve.discount(maturity))
    if not np.isfinite(p_s) or not np.isfinite(p_t) or min(p_s, p_t) <= 0:
        raise ValueError("discount factors must be positive and finite")
    rng = np.random.default_rng(seed)
    if antithetic:
        # Each pair is correlated. Compute standard errors on independent
        # *pair averages*, not on individual correlated payoffs.
        if n_paths % 2:
            raise ValueError("antithetic n_paths must be even")
        z = rng.standard_normal(n_paths // 2)
        v = np.sqrt(variance)
        f_plus = (p_t / p_s) * np.exp(-0.5 * variance + v * z)
        f_minus = (p_t / p_s) * np.exp(-0.5 * variance - v * z)
        if option_type == "call":
            samples = 0.5 * (np.maximum(f_plus - strike, 0) + np.maximum(f_minus - strike, 0))
        else:
            samples = 0.5 * (np.maximum(strike - f_plus, 0) + np.maximum(strike - f_minus, 0))
    else:
        z = rng.standard_normal(n_paths)
        terminal = (p_t / p_s) * np.exp(-0.5 * variance + np.sqrt(variance) * z)
        samples = np.maximum(terminal - strike, 0) if option_type == "call" else np.maximum(strike - terminal, 0)
    price = float(p_s * np.mean(samples))
    std_error = float(p_s * np.std(samples, ddof=1) / np.sqrt(len(samples)))
    critical = NormalDist().inv_cdf(0.5 + confidence / 2)
    return MonteCarloEstimate(price, std_error, price - critical * std_error,
                              price + critical * std_error, n_paths, confidence)
