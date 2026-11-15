# Single-curve vanilla swap pricing benchmark

The module `hjm_neural_rates_lab.swap_pricing` supplies a deterministic
discount-curve benchmark for later stochastic HJM swaption pricing.

For fixed-leg payment dates `T_i`, accruals `alpha_i`, and discount factors
`P(0,T_i)`, the single-curve par rate is:

`K = [P(0,T_start)-P(0,T_end)] / sum_i alpha_i P(0,T_i)`.

The payer swap PV is notional times
`[P(0,T_start)-P(0,T_end)-K sum_i alpha_i P(0,T_i)]`.

Tests verify zero value at par, the sign of an off-market fixed rate,
forward-starting discounting, and invalid schedule rejection.

## Important limitations

The floating leg assumes single-curve replication with no spreads.
Accrual fractions and payment times are supplied, not calculated from
market calendars. The spot-start default assumes `P(0,0)=1`.
These are benchmark formulas, not a market-ready swap pricer.

## Run

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_swap_pricing.py tests/test_curve_risk.py
```

GitHub Actions is configured to run the entire test suite on pushes and
pull requests with Python 3.10 and 3.12. A passing CI run is not asserted
until its outcome has been inspected.
