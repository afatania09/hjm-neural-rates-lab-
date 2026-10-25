"""Deterministic initial forward curves and bond-price identities."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlatForwardCurve:
    rate: float = 0.03

    def forward(self, maturities):
        t = np.asarray(maturities, dtype=float)
        if np.any(t < 0):
            raise ValueError("Maturities must be non-negative")
        return np.full_like(t, self.rate, dtype=float)

    def discount(self, maturities):
        t = np.asarray(maturities, dtype=float)
        if np.any(t < 0):
            raise ValueError("Maturities must be non-negative")
        return np.exp(-self.rate * t)


def bond_price_from_forwards(maturities, forwards, valuation_time=0.0):
    """P(t,T)=exp(-integral_t^T f(t,u)du), trapezoidal quadrature.

    Maturity grid must include valuation_time and target maturity.
    """
    t = np.asarray(maturities, dtype=float)
    f = np.asarray(forwards, dtype=float)
    if t.ndim != 1 or f.shape != t.shape or len(t) < 2:
        raise ValueError("Matching one-dimensional arrays of length >= 2 required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
        raise ValueError("Non-finite input")
    if not np.all(np.diff(t) > 0) or not np.isclose(t[0], valuation_time):
        raise ValueError("Grid must increase strictly and start at valuation_time")
    return float(np.exp(-np.trapezoid(f, t))) if hasattr(np, "trapezoid") else float(np.exp(-np.trapz(f, t)))
