"""Deterministic discount-curve diagnostics and parallel-shift risk.

All maturities are in years and zero rates are continuously compounded.
These utilities are deliberately independent of stochastic model calibration.
"""
from __future__ import annotations

import numpy as np


def discount_factors(maturities, zero_rates):
    """Return P(0,T)=exp(-T*z(T)); supports scalar or broadcastable arrays."""
    t = np.asarray(maturities, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(z)):
        raise ValueError("Maturities and zero rates must be finite")
    if np.any(t < 0):
        raise ValueError("Maturities must be non-negative")
    return np.exp(-t * z)


def instantaneous_forwards(maturities, zero_rates):
    """Estimate f(0,T)=d[T*z(T)]/dT on a strictly increasing maturity grid.

    Interior derivatives use numpy.gradient's second-order nonuniform stencil.
    Endpoint estimates are numerical, not market-observed forwards.
    """
    t = np.asarray(maturities, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    if t.ndim != 1 or z.shape != t.shape or len(t) < 3:
        raise ValueError("Provide matching one-dimensional grids of >=3 nodes")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(z)):
        raise ValueError("Inputs must be finite")
    if t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Maturities must be non-negative and strictly increasing")
    return np.gradient(t * z, t, edge_order=2)


def parallel_shift_pv01(maturities, zero_rates, cashflows, bump=1e-4):
    """Central-difference PV01: PV(z+1bp)-PV(z), per +1bp shift.

    Returns a signed monetary change, not a positive duration convention.
    """
    t = np.asarray(maturities, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    c = np.asarray(cashflows, dtype=float)
    if t.ndim != 1 or t.shape != z.shape or t.shape != c.shape:
        raise ValueError("Maturities, zero rates and cashflows must match")
    if not np.all(np.isfinite(c)) or not np.isfinite(bump) or bump <= 0:
        raise ValueError("Cashflows must be finite and bump must be positive")
    base = np.sum(c * discount_factors(t, z))
    up = np.sum(c * discount_factors(t, z + bump))
    down = np.sum(c * discount_factors(t, z - bump))
    derivative = (up - down) / (2 * bump)
    return float(base), float(derivative * 1e-4)
