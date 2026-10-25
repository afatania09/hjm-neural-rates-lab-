"""European options on zero-coupon bonds in deterministic-volatility Gaussian HJM.

The expiry-S forward bond price P(S,T) / P(S,S) is lognormal under the
S-forward measure for deterministic HJM volatility. Here P(S,S)=1.
All rates/volatilities use continuously compounded conventions.
"""
import numpy as np
from scipy.stats import norm
from .gaussian_hjm import GaussianHJM


def forward_bond_log_variance(model: GaussianHJM, expiry: float, maturity: float) -> float:
    """Integral_0^S ||integral_S^T sigma(t,u) du||^2 dt.

    Valid for sigma_i(t,u)=a_i exp(-b_i(u-t)), independent Brownian factors.
    """
    if not np.isfinite(expiry) or not np.isfinite(maturity) or expiry < 0 or maturity <= expiry:
        raise ValueError("Require finite 0 <= expiry < maturity")
    delta = maturity - expiry
    total = 0.0
    for a, b in zip(model.amplitudes, model.decays):
        if b == 0:
            total += (a * delta) ** 2 * expiry
        else:
            c = a * (-np.expm1(-b * delta)) / b
            total += c * c * (-np.expm1(-2 * b * expiry)) / (2 * b)
    return float(total)


def zero_coupon_bond_option(
    model: GaussianHJM,
    curve,
    expiry: float,
    maturity: float,
    strike: float,
    option_type: str = "call",
) -> float:
    """Time-zero undiscounted-notional option price, currency units per 1 face.

    Payoff at expiry S: max(P(S,T)-K,0) for call, or reverse for put.
    P(0,S), P(0,T) from supplied initial discount curve.
    """
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("Strike must be positive and finite")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    variance = forward_bond_log_variance(model, expiry, maturity)
    p_s, p_t = float(curve.discount(expiry)), float(curve.discount(maturity))
    if not np.isfinite(p_s) or not np.isfinite(p_t) or p_s <= 0 or p_t <= 0:
        raise ValueError("Initial discount factors must be finite and positive")
    intrinsic = p_t - strike * p_s
    if variance <= 1e-30:
        return float(max(intrinsic, 0) if option_type == "call" else max(-intrinsic, 0))
    vol = np.sqrt(variance)
    d1 = np.log(p_t / (strike * p_s)) / vol + 0.5 * vol
    d2 = d1 - vol
    call = p_t * norm.cdf(d1) - strike * p_s * norm.cdf(d2)
    return float(call if option_type == "call" else call - intrinsic)
