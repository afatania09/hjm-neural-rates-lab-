"""Exact joint Gaussian terminal law for discounting and bond prices in HJM.

For deterministic factor volatilities, both log D(S) and log P(S,T)
have Gaussian random components. Their covariance is determined by the
HJM volatility kernels. We use a joint Gaussian construction with
means fixed by the risk-neutral bond identities.

This is an exact *terminal joint-law sampler*, NOT an Euler path
simulation or an independently implemented money-market account.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class TerminalBondSample:
    discount_factors: np.ndarray
    bond_prices: np.ndarray
    covariance: np.ndarray


def _integrated_sigma(a, b, start, end, u):
    """Integral_start^end a exp(-b(v-u)) dv, with u<=start."""
    length = end - start
    return a * np.exp(-b * (start - u)) * (
        -np.expm1(-b * length) / b if b > 0 else length
    )


def terminal_log_covariance(model, expiry, maturity):
    """Covariance of (log D(S), log P(S,T)) random components.

    The Brownian kernels are -int_u^S sigma(u,v)dv for log D(S)
    and -int_S^T sigma(u,v)dv for log P(S,T).
    """
    if not np.isfinite(expiry) or not np.isfinite(maturity) or expiry <= 0 or maturity <= expiry:
        raise ValueError("Require 0 < expiry < maturity, finite")
    v_d = v_p = cov = 0.0
    for a, b in zip(model.amplitudes, model.decays):
        def kd(u):
            return _integrated_sigma(a, b, u, expiry, u)
        def kp(u):
            return _integrated_sigma(a, b, expiry, maturity, u)
        v_d += quad(lambda u: kd(u)**2, 0, expiry, epsabs=1e-12)[0]
        v_p += quad(lambda u: kp(u)**2, 0, expiry, epsabs=1e-12)[0]
        cov += quad(lambda u: kd(u)*kp(u), 0, expiry, epsabs=1e-12)[0]
    return np.array([[v_d, cov], [cov, v_p]], dtype=float)


def simulate_terminal_bond_risk_neutral(model, curve, expiry, maturity,
                                        n_paths=100_000, seed=2026):
    """Jointly simulate D(S)=exp(-int_0^S r du) and P(S,T) under Q.

    Means enforce E_Q[D(S)]=P(0,S) and
    E_Q[D(S)P(S,T)]=P(0,T).
    """
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer >=2")
    covariance = terminal_log_covariance(model, expiry, maturity)
    p_s = float(curve.discount(expiry))
    p_t = float(curve.discount(maturity))
    if not np.isfinite(p_s) or not np.isfinite(p_t) or min(p_s, p_t) <= 0:
        raise ValueError("Initial discounts must be finite and positive")
    var_d, cov_dp, var_p = covariance[0, 0], covariance[0, 1], covariance[1, 1]
    mu_d = np.log(p_s) - 0.5 * var_d
    mu_p = np.log(p_t / p_s) - 0.5 * var_p - cov_dp
    rng = np.random.default_rng(seed)
    # Eigen decomposition accommodates rank-deficient covariance (e.g. zero vol).
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) < -1e-10:
        raise ArithmeticError("Non-PSD covariance")
    transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    shocks = rng.standard_normal((n_paths, 2)) @ transform.T
    discounts = np.exp(mu_d + shocks[:, 0])
    bonds = np.exp(mu_p + shocks[:, 1])
    return TerminalBondSample(discounts, bonds, covariance)
