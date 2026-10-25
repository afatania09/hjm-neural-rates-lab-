"""Reproducible minimal demonstration; run after pip install -e ."""
import numpy as np
from hjm_rates import FlatForwardCurve, GaussianHJM, simulate_forward_paths

maturities = np.linspace(0, 10, 101)
times = np.linspace(0, 1, 13)
model = GaussianHJM(amplitudes=(0.01, 0.006), decays=(0.12, 0.8))
paths = simulate_forward_paths(model, FlatForwardCurve(0.035), maturities, times, n_paths=500, seed=2026)
terminal = paths[:, -1, :]
live = maturities >= times[-1]
print("HJM research demonstration (synthetic data)")
print(f"Paths: {paths.shape[0]}, factors: {model.n_factors}")
print(f"Mean simulated f(1,5): {terminal[:, np.argmin(abs(maturities - 5))].mean():.6f}")
print(f"Terminal forward dispersion: {np.std(terminal[:, live], axis=0).mean():.6f}")
print("Not a pricing or trading recommendation.")
