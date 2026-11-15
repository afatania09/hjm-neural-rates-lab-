"""Single-curve, deterministic discounting benchmarks for vanilla swaps.

Continuous zero-rate discounting; annualised fixed coupons. No credit,
multi-curve basis, collateral, calendars, or floating-leg spread modelling.
"""
from __future__ import annotations

import numpy as np

from hjm_neural_rates_lab.curve_risk import discount_factors


def _validate_schedule(payment_times, accrual_fractions):
    times = np.asarray(payment_times, dtype=float)
    accruals = np.asarray(accrual_fractions, dtype=float)
    if times.ndim != 1 or accruals.shape != times.shape or times.size == 0:
        raise ValueError("Matching nonempty one-dimensional schedules required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(accruals)):
        raise ValueError("Schedule must be finite")
    if np.any(times <= 0) or np.any(np.diff(times) <= 0) or np.any(accruals <= 0):
        raise ValueError("Payment times must increase and accruals be positive")
    return times, accruals


def par_swap_rate(payment_times, accrual_fractions, zero_rates, start_discount=1.0):
    """Par fixed rate for a single-curve spot/forward-starting swap.

    Formula: K=(P(0,start)-P(0,end))/sum_i alpha_i P(0,T_i).
    Caller supplies P(0,start), defaulting to spot start.
    """
    times, accruals = _validate_schedule(payment_times, accrual_fractions)
    rates = np.asarray(zero_rates, dtype=float)
    if rates.shape != times.shape or not np.all(np.isfinite(rates)):
        raise ValueError("Zero rates must match schedule and be finite")
    if not np.isfinite(start_discount) or not 0 < start_discount <= 1:
        raise ValueError("Start discount must be in (0,1]")
    discounts = discount_factors(times, rates)
    annuity = float(np.dot(accruals, discounts))
    return float((start_discount - discounts[-1]) / annuity)


def payer_swap_pv(payment_times, accrual_fractions, zero_rates, fixed_rate,
                  notional=1.0, start_discount=1.0):
    """Receive floating, pay fixed; returns present value in notional units."""
    times, accruals = _validate_schedule(payment_times, accrual_fractions)
    rates = np.asarray(zero_rates, dtype=float)
    if rates.shape != times.shape or not np.all(np.isfinite(rates)):
        raise ValueError("Zero rates must match schedule and be finite")
    if not np.isfinite(fixed_rate) or not np.isfinite(notional) or notional <= 0:
        raise ValueError("Fixed rate must be finite; notional positive")
    if not np.isfinite(start_discount) or not 0 < start_discount <= 1:
        raise ValueError("Start discount must be in (0,1]")
    discounts = discount_factors(times, rates)
    return float(notional * (start_discount - discounts[-1]
                            - fixed_rate * np.dot(accruals, discounts)))
