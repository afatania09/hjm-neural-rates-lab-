"""Analytical benchmark: bond options under Gaussian HJM."""
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.bond_options import zero_coupon_bond_option, forward_bond_log_variance

curve = FlatForwardCurve(0.035)
model = GaussianHJM((0.01, 0.006), (0.15, 0.7))
expiry, maturity = 2.0, 7.0
strike = curve.discount(maturity) / curve.discount(expiry)
call = zero_coupon_bond_option(model, curve, expiry, maturity, strike)
put = zero_coupon_bond_option(model, curve, expiry, maturity, strike, "put")
print("Gaussian HJM zero-coupon bond option benchmark (synthetic)")
print(f"Expiry={expiry}, maturity={maturity}, strike={strike:.8f}")
print(f"Forward log variance={forward_bond_log_variance(model, expiry, maturity):.8g}")
print(f"ATM-forward call={call:.8f}, put={put:.8f}")
print("Analytical benchmark, not an observed market price.")
