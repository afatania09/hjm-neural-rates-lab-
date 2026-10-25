"""Martingale audit for the S-forward measure; synthetic Gaussian HJM."""
import numpy as np
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.forward_bonds import simulate_forward_bond_paths

curve = FlatForwardCurve(0.035)
model = GaussianHJM((0.015, 0.008), (0.2, 0.7))
expiry, maturity = 2., 6.
paths = simulate_forward_bond_paths(model, curve, expiry, maturity,
                                    np.linspace(0, expiry, 9), 50000, seed=71)
initial = curve.discount(maturity)/curve.discount(expiry)
print("Forward bond martingale audit under S-forward measure")
print("t       sample_mean     initial_F      mean_minus_initial    standard_error")
for j, t in enumerate(paths.times):
    sample = paths.prices[:, j]
    se = np.std(sample, ddof=1) / np.sqrt(len(sample))
    print(f"{t:4.2f}    {np.mean(sample):.8f}     {initial:.8f}       {np.mean(sample)-initial:+.8f}        {se:.8f}")
print("A single finite-sample run cannot establish a martingale theorem.")
