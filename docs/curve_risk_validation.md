# Deterministic curve-risk validation (November 2026)

This is a deliberately narrow, auditable addition to the HJM research lab.
It is **not** a neural model, a stochastic calibration, or a claim of live-market accuracy.

## Scope

- Convert continuously compounded zero rates to discount factors.
- Recover approximate instantaneous forwards by differentiating `T z(T)`.
- Price deterministic cashflows and estimate signed PV01 under a parallel
  continuously compounded zero-rate bump.
- Compare numerical results with independent analytic flat-curve identities.

## Conventions

`P(0,T)=exp(-T z(T))`, `f(0,T)=d(T z(T))/dT`.
For cashflows `C_i`, `PV=sum_i C_i P(0,T_i)`.
The analytic parallel-rate derivative is
`dPV/dz=-sum_i T_i C_i exp(-T_i z_i)`.
The reported signed PV01 is this derivative times `1e-4`; a positive
fixed-cashflow bond ordinarily has negative signed PV01.

**Limitations:** these are continuously compounded *zero-rate* shifts, not
par swap-rate shocks or key-rate durations. Grid derivatives can be noisy
and endpoint estimates depend on the numerical stencil. No bootstrapping,
day-count, collateral discounting or curve interpolation is claimed.

## Reproduction

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_curve_risk.py
```

The tests encode expected results; GitHub CI or a local test run is needed
to establish whether they pass in the target environment.

## Next controlled experiment

1. Run the existing HJM test suite and capture the environment and seeds.
2. Introduce independent multi-factor discounted-bond martingale checks
   across both time and maturity grids.
3. Report sampling uncertainty separately from discretisation bias.
4. Only then generate synthetic holdouts for neural surrogate benchmarks.
