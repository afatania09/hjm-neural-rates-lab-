"""Compare exact and finite-difference rate sensitivities for synthetic HJM bond options."""
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.sensitivities import parallel_rate_derivative, parallel_rate_derivative_fd, parallel_dv01

model = GaussianHJM((0.015, 0.009), (0.12, 0.6))
curve = FlatForwardCurve(0.035)
print("European zero-coupon bond call: parallel forward-rate sensitivity audit")
print("strike   analytic dPV/dr    central FD dPV/dr    absolute error      signed DV01")
for strike in (0.7, 0.8, 0.9, 1.0):
    exact = parallel_rate_derivative(model, curve, 2, 6, strike)
    fd = parallel_rate_derivative_fd(model, curve, 2, 6, strike)
    print(f"{strike:5.2f}    {exact:14.8f}     {fd:14.8f}     {abs(exact-fd):14.4g}    {parallel_dv01(model, curve, 2, 6, strike):12.8f}")
print("This is parallel continuous-forward DV01, not a bootstrapped market key-rate risk measure.")
