"""Risk-neutral terminal-law audit: bond martingale and option pricing."""
import numpy as np
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.risk_neutral import simulate_terminal_bond_risk_neutral
from hjm_rates.bond_options import zero_coupon_bond_option

curve = FlatForwardCurve(0.035)
model = GaussianHJM((0.02, 0.008), (0.15, 0.7))
expiry, maturity = 2, 6
strike = curve.discount(maturity)/curve.discount(expiry)
r = simulate_terminal_bond_risk_neutral(model, curve, expiry, maturity, 100000, 71)
for name, x, ref in [
    ("discount factor", r.discount_factors, curve.discount(expiry)),
    ("discounted bond", r.discount_factors*r.bond_prices, curve.discount(maturity)),
    ("discounted call", r.discount_factors*np.maximum(r.bond_prices-strike, 0),
     zero_coupon_bond_option(model, curve, expiry, maturity, strike)),
]:
    se = np.std(x, ddof=1)/np.sqrt(len(x))
    print(f"{name:17s} estimate={np.mean(x):.8f} reference={ref:.8f} SE={se:.8f} z={(np.mean(x)-ref)/se if se else 0:.3f}")
print("Terminal joint Gaussian law only; this is not a pathwise Euler convergence test.")
