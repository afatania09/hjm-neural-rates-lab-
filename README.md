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
| 2d | Euler time-step convergence, numeraire and martingale audit | Planned |
| 3 | Caplets, bond options, multi-curve conventions | Planned |
| 4 | Literature review and reproducibility of published neural HJM baselines | Planned |
| 5 | Neural pricing and Greeks, out-of-distribution stress tests | Planned |
| 6 | Controlled improvements and research paper | Conditional on evidence |

See [research protocol](docs/research_protocol.md) for experiment controls.

## Limitations and integrity

The simulator evolves forward rates on **fixed absolute maturity nodes**, not a fully implemented Musiela SPDE. It does not yet implement an integrated short-rate numeraire, pathwise Euler Monte Carlo bond-option valuation, hedge P&L, a PINN, or any arbitrage verification beyond the HJM drift formula. Analytical European zero-coupon bond-option pricing is implemented but CI verification is pending. No performance or novelty claims are made. Synthetic inputs are used; this is not investment advice.

## Licence

MIT; see [LICENSE](LICENSE). Copyright (c) 2026 Ashutosh Fatania.
