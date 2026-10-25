"""Analytical and finite-difference parallel-rate sensitivities for Gaussian HJM.

A parallel *instantaneous-forward* curve shift by h changes
P(0,u) to P(0,u) exp(-h*u). Volatility is held fixed.
This is NOT a market-standard bootstrapped/key-rate DV01.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from .bond_options import forward_bond_log_variance, zero_coupon_bond_option


@dataclass(frozen=True)
class ShiftedDiscountCurve:
    base_curve: object
    shift: float

    def discount(self, maturity):
        return np.asarray(self.base_curve.discount(maturity)) * np.exp(-self.shift * np.asarray(maturity))


def parallel_rate_derivative(model, curve, expiry, maturity, strike, option_type="call"):
    """Analytical d(option price)/d(parallel continuous forward shift), per unit rate."""
    variance = forward_bond_log_variance(model, expiry, maturity)
    ps = float(curve.discount(expiry))
    pt = float(curve.discount(maturity))
    if not np.isfinite(strike) or strike <= 0 or option_type not in ("call", "put"):
        raise ValueError("Invalid option contract")
    if variance <= 1e-30:
        intrinsic = pt - strike * ps
        if abs(intrinsic) < 1e-12:
            raise ValueError("Sensitivity undefined at zero-volatility exercise boundary")
        active = intrinsic > 0 if option_type == "call" else intrinsic < 0
        return float((-maturity * pt + strike * expiry * ps) * (1 if option_type == "call" else -1)) if active else 0.0
    v = np.sqrt(variance)
    d1 = np.log(pt / (strike * ps)) / v + v / 2
    d2 = d1 - v
    derivative_call = -maturity * pt * norm.cdf(d1) + strike * expiry * ps * norm.cdf(d2)
    if option_type == "call":
        return float(derivative_call)
    return float(derivative_call + maturity * pt - strike * expiry * ps)


def parallel_rate_derivative_fd(model, curve, expiry, maturity, strike, option_type="call", bump=1e-5):
    """Central finite difference; bump is absolute decimal rate (1e-4 = 1 bp)."""
    if not np.isfinite(bump) or bump <= 0:
        raise ValueError("bump must be positive and finite")
    plus = zero_coupon_bond_option(model, ShiftedDiscountCurve(curve, bump),
                                   expiry, maturity, strike, option_type)
    minus = zero_coupon_bond_option(model, ShiftedDiscountCurve(curve, -bump),
                                    expiry, maturity, strike, option_type)
    return float((plus - minus) / (2 * bump))


def parallel_dv01(model, curve, expiry, maturity, strike, option_type="call"):
    """Signed option PV change for a +1bp parallel continuously-compounded rate move."""
    return 1e-4 * parallel_rate_derivative(model, curve, expiry, maturity, strike, option_type)
