"""European single-period interest-rate caplets/floorlets in Gaussian HJM.

Simple forward L(S,T)=(1/P(S,T)-1)/(T-S). Payment at T.
The T-settled caplet has exact time-zero value equal to a scaled
European put on a zero-coupon bond expiring at S.
"""
from dataclasses import dataclass
import numpy as np
from .bond_options import zero_coupon_bond_option


@dataclass(frozen=True)
class RateOptionQuote:
    caplet: float
    floorlet: float
    forward_rate: float
    accrual: float
    parity_residual: float


def caplet_price(model, curve, reset, payment, strike, notional=1.0):
    """Value of notional*(T-S)*max(L(S,T)-K,0), paid at T."""
    delta = _validate(reset, payment, strike, notional)
    bond_strike = 1.0 / (1.0 + delta * strike)
    return float(notional * (1.0 + delta * strike) *
                 zero_coupon_bond_option(model, curve, reset, payment,
                                         bond_strike, option_type="put"))


def floorlet_price(model, curve, reset, payment, strike, notional=1.0):
    """Value of notional*(T-S)*max(K-L(S,T),0), paid at T."""
    delta = _validate(reset, payment, strike, notional)
    bond_strike = 1.0 / (1.0 + delta * strike)
    return float(notional * (1.0 + delta * strike) *
                 zero_coupon_bond_option(model, curve, reset, payment,
                                         bond_strike, option_type="call"))


def caplet_floorlet_quote(model, curve, reset, payment, strike, notional=1.0):
    """Prices and a diagnostic based on caplet-floorlet parity."""
    delta = _validate(reset, payment, strike, notional)
    p_s = float(curve.discount(reset))
    p_t = float(curve.discount(payment))
    if not np.isfinite(p_s) or not np.isfinite(p_t) or min(p_s, p_t) <= 0:
        raise ValueError("Discount factors must be positive and finite")
    forward = (p_s / p_t - 1.0) / delta
    cap = caplet_price(model, curve, reset, payment, strike, notional)
    floor = floorlet_price(model, curve, reset, payment, strike, notional)
    parity = notional * (p_s - (1.0 + delta * strike) * p_t)
    return RateOptionQuote(cap, floor, float(forward), delta,
                           float(cap - floor - parity))


def _validate(reset, payment, strike, notional):
    if not np.isfinite(reset) or not np.isfinite(payment) or reset < 0 or payment <= reset:
        raise ValueError("Require finite 0 <= reset < payment")
    delta = payment - reset
    if not np.isfinite(strike) or 1.0 + delta * strike <= 0:
        raise ValueError("Require finite strike with 1+accrual*strike > 0")
    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("Notional must be finite and positive")
    return delta
