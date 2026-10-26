"""Exact Ho-Lee risk-neutral discounted-bond martingale audit."""
import numpy as np
from hjm_rates.ho_lee_paths import simulate_ho_lee_paths

r0, a, maturity = 0.03, 0.025, 5.
paths = simulate_ho_lee_paths(r0, a, maturity, np.linspace(0, 2, 9), 50000, 71)
reference = np.exp(-r0*maturity)
print("Ho-Lee risk-neutral discounted bond martingale; exact bridge integration")
print("time       mean[D(t)P(t,T)]       reference         standard error      z")
for j, t in enumerate(paths.times):
    values = paths.discount_factors[:, j] * paths.bond_prices[:, j]
    se = np.std(values, ddof=1)/np.sqrt(len(values))
    print(f"{t:4.2f}       {np.mean(values):.9f}          {reference:.9f}       {se:.9f}        {(np.mean(values)-reference)/se if se else 0:.3f}")
print("This is a single-factor Ho-Lee exact-grid benchmark, not a multi-factor Euler audit.")
