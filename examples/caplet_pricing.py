"""Synthetic rate caplet pricing and put-call parity demonstration."""
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.caplets import caplet_floorlet_quote

curve = FlatForwardCurve(0.035)
model = GaussianHJM((0.015, 0.008), (0.2, 0.7))
print("T-settled caplet and floorlet prices per 1m notional")
print("strike      caplet PV       floorlet PV       parity residual")
for strike in (0.02, 0.03, 0.04, 0.05):
    q = caplet_floorlet_quote(model, curve, 2.0, 2.5, strike, 1_000_000)
    print(f"{strike:.2%}      {q.caplet:12.4f}    {q.floorlet:12.4f}      {q.parity_residual:+.3g}")
print("Synthetic continuous discount curve; not market calibration.")
