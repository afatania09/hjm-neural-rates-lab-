"""Demonstrate the deterministic Ho-Lee Euler drift bias."""
from hjm_rates.euler_audit import ho_lee_forward_audit

for n in (5, 10, 20, 40, 80):
    r = ho_lee_forward_audit(n_steps=n, n_paths=20000, seed=71)
    print("steps", n, "dt", r.dt, "MC mean", r.sample_mean,
          "exact mean", r.theoretical_mean, "Euler mean", r.predicted_euler_mean,
          "predicted bias", r.predicted_drift_bias, "sampling SE", r.sampling_se)
