# Independent multi-factor Gaussian HJM reference

This addition supplies a closed-form integrated log-variance for an
S-forward-measure forward bond under **independent** deterministic
exponentially decaying Gaussian HJM factors.

For 0 <= S <= T, factor i has volatility
`sigma_i(t,u) = a_i exp(-b_i(u-t))`, b_i >= 0.
The forward-bond diffusion coefficient magnitude is
`g_i(t)=integral_S^T sigma_i(t,u)du`.
Under the S-forward measure, the forward bond is a martingale, and
its terminal log-variance is `V=sum_i integral_0^S g_i(t)^2dt`.

For b_i>0:
`V_i = a_i^2 [(1-exp(-b_i(T-S)))/b_i]^2 [(1-exp(-2b_i S))/(2b_i)]`.
For b_i=0: `V_i=a_i^2(T-S)^2 S`.

An exact terminal sampler uses
`F(S)=F(0)exp(-V/2+sqrt(V)Z)`, with Z standard normal.
Tests compare the closed form to nested numerical quadrature, test
additivity across independent factors, reproduce the Ho-Lee limit,
and check a seeded Monte Carlo sample mean using a studentised z-score.

## Reproduction

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_gaussian_factor_risk.py
```

**Research integrity:** this is an exact *terminal-law* benchmark, not a
multi-factor Euler time/maturity-grid simulation, pathwise numeraire,
calibration or neural surrogate. The statistical test has a small
nonzero false-failure probability. Passing a single seeded test is not
a comprehensive convergence study.
