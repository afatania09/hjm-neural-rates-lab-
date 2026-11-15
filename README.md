# HJM Neural Rates Lab

**An open, reproducible research platform for interest-rate modelling, numerical validation, and eventual neural pricing under the Heath–Jarrow–Morton framework.**

> **Status: Stage 1 — mathematical and simulation foundations.** This is not yet a validated derivative pricer, neural solver, or published research result.

## What exists today

- Multi-factor Gaussian HJM with exponentially decaying deterministic maturity volatility.
- Risk-neutral no-arbitrage drift calculated from the factor volatility integrals.
- Fixed-absolute-maturity forward-rate Euler simulator with seeded randomness.
- Flat initial forward curve and numerical zero-coupon bond identity.
- Unit tests for drift, integrated volatility, reproducibility and input validation.
- Analytical Gaussian HJM European zero-coupon bond call/put pricing, with independent variance-quadrature and parity tests.
- Exact terminal forward-measure Monte Carlo bond-option estimator with antithetic variates and approximate confidence intervals.
- Analytic parallel-forward curve sensitivities and signed DV01, cross-checked with central finite differences.
- Exact Gaussian HJM S-forward-measure forward-bond paths and statistical martingale audit.
- Joint risk-neutral Gaussian terminal distribution for money-market discount factors and bond prices, with discounted-bond and option consistency tests.
- Independent Ho-Lee constant-volatility benchmark of the original Euler forward-rate simulator, including an exact formula for its first-order drift bias.
- Exact Ho-Lee risk-neutral short-rate paths and Brownian-bridge-integrated money-market discount factors, with discounted bond martingale tests.
- Single-period caplet and floorlet valuation via equivalent zero-coupon bond options, including parity and volatility checks.
- Research protocol describing numerical baselines and failure modes.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/first_simulation.py
python examples/bond_option_benchmark.py
python examples/monte_carlo_audit.py
python examples/hedge_sensitivity_audit.py
python examples/forward_bond_martingale_audit.py
python examples/risk_neutral_audit.py
python examples/euler_convergence_audit.py
python examples/ho_lee_numeraire_audit.py
python examples/caplet_pricing.py
```

## Mathematical formulation

Under the risk-neutral money-market measure, the instantaneous forward rate follows

```text
df(t,T) = alpha(t,T) dt + sum_i sigma_i(t,T) dW_i(t)

alpha(t,T) = sum_i sigma_i(t,T) * integral[t,T] sigma_i(t,u) du

P(t,T) = exp(-integral[t,T] f(t,u) du)
```

For the first benchmark, factor volatility is `sigma_i(t,T) = a_i exp(-b_i(T-t))` with `b_i >= 0`. The integral is analytic, including the zero-decay limit.

## Research roadmap

| Stage | Objective | Status |
| --- | --- | --- |
| 1 | HJM drift, forward curves, simulation and tests | Initial implementation |
| 2 | Analytical Gaussian bond-option reference | Initial implementation (tests pending) |
| 2b | Forward-measure Monte Carlo statistical audit | Implemented (CI pending) |
| 2c | Analytic parallel-forward DV01 and finite-difference audit | Implemented (CI pending) |
| 2d | S-forward-measure forward-bond martingale audit | Implemented (CI pending) |
| 2e | Risk-neutral terminal discount-factor/bond consistency audit | Implemented (CI pending) |
| 2f | Ho-Lee fixed-maturity Euler drift convergence | Implemented (CI pending) |
| 2g | Exact Ho-Lee pathwise numeraire and discounted-bond martingale audit | Implemented (CI pending) |
| 2h | Multi-factor Euler-grid numeraire and joint time/maturity convergence | Planned |
| 3 | Caplets, bond options, multi-curve conventions | Planned |
| 4 | Literature review and reproducibility of published neural HJM baselines | Planned |
| 5 | Neural pricing and Greeks, out-of-distribution stress tests | Planned |
| 6 | Controlled improvements and research paper | Conditional on evidence |

See [research protocol](docs/research_protocol.md) for experiment controls.

## Limitations and integrity

The simulator evolves forward rates on **fixed absolute maturity nodes**, not a fully implemented Musiela SPDE. It does not yet implement a multi-factor Euler-grid short-rate numeraire, pathwise Euler Monte Carlo bond-option valuation, hedge P&L, a PINN, or any arbitrage verification beyond the HJM drift formula. Analytical European zero-coupon bond-option pricing is implemented but CI verification is pending. No performance or novelty claims are made. Synthetic inputs are used; this is not investment advice.

## Licence

MIT; see [LICENSE](LICENSE). Copyright (c) 2026 Ashutosh Fatania.

## Deterministic curve and swap benchmarks (November 2026)

The `curve_risk` module adds continuously compounded discount factors,
finite-grid instantaneous forwards and signed parallel-shift PV01. The
`swap_pricing` module adds single-curve par swap rates and payer swap PV,
including forward-starting contracts. Independent analytical tests are in
`tests/test_curve_risk.py` and `tests/test_swap_pricing.py`.

See [curve risk validation](docs/curve_risk_validation.md) and
[swap pricing validation](docs/swap_pricing_validation.md). These additions
are deterministic reference calculations, **not** calibrated market pricing,
HJM swaptions or neural-network results. Existing GitHub Actions is configured
to execute the test suite; successful execution has not been verified here.
