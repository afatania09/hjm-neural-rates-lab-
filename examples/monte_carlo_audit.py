"""Statistical convergence demonstration for the exact S-forward terminal law."""
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.bond_options import zero_coupon_bond_option
from hjm_rates.monte_carlo import bond_option_forward_measure_mc

curve = FlatForwardCurve(0.035)
model = GaussianHJM((0.025, 0.01), (0.15, 0.7))
expiry, maturity = 2.0, 7.0
strike = curve.discount(maturity) / curve.discount(expiry)
reference = zero_coupon_bond_option(model, curve, expiry, maturity, strike)
print(f"Analytical call price: {reference:.8f}")
print("paths      estimate      standard_error     abs_error      CI_covers_reference")
for n in (2_000, 20_000, 200_000):
    result = bond_option_forward_measure_mc(model, curve, expiry, maturity, strike,
                                            n_paths=n, seed=2026)
    print(f"{n:>7}    {result.price:.8f}      {result.std_error:.8f}        {abs(result.price-reference):.8f}       {result.ci_lower <= reference <= result.ci_upper}")
print("Intervals are approximate; coverage of an individual interval is not guaranteed.")
