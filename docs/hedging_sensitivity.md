# First sensitivity benchmark: parallel-forward DV01

For a uniform shift h in the initial instantaneous forward curve, P_h(0,u)=P(0,u)exp(-hu). Holding deterministic Gaussian HJM volatility fixed, differentiating the Black-style zero-coupon bond option formula gives

    dC/dh = -T P(0,T) Phi(d1) + K S P(0,S) Phi(d2).

Put sensitivity follows by differentiating put-call parity:

    dPut/dh = dCall/dh + T P(0,T) - K S P(0,S).

The code compares this analytic derivative with a central finite-difference approximation. Signed DV01 = (dPV/dh)*1e-4; this is the local linear price change for a +1bp shift. No bump-and-reprice claim is made for non-infinitesimal shocks.

## Important limitations

- A **parallel instantaneous-forward** curve shift is not a market-standard key-rate DV01 or sensitivity to a specific traded rate instrument.
- The model's volatility parameters are held constant under curve shifts.
- This test measures the correctness of derivatives of the analytical formula, **not hedging P&L** and not neural-network Greek quality.
- At zero volatility exactly on the exercise boundary, the option price is nondifferentiable; the analytic derivative rejects that case.

## Next research gate

Implement a hedging experiment with self-financing cashflows and an independently generated stochastic path model. Track hedge error distributions across time discretisations, volatility regimes and yield-curve shocks. Only after this validation should neural Greek estimators be compared.
