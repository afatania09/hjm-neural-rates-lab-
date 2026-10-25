# Gaussian HJM forward-measure Monte Carlo audit

The analytic benchmark provides an exact integrated variance for the forward bond price F(t)=P(t,T)/P(t,S), with S<T. Under the S-forward measure, deterministic-volatility Gaussian HJM gives

    F(S) = [P(0,T)/P(0,S)] exp(-V/2 + sqrt(V) Z),  Z ~ N(0,1).

The option price is P(0,S) times the expected payoff under this measure.

## What this validates

- The analytical Black-style option formula against a *separate statistical payoff estimator*.
- Sampling uncertainty using standard errors and approximate confidence intervals.
- Antithetic pairing, with standard errors calculated from independent **pair means**.
- Reproducibility using fixed pseudorandom seeds.

## What this does NOT validate

This method samples the exact **terminal distribution**. It does not validate the fixed-maturity Euler HJM simulator, its time-step convergence, bond discounting along paths, martingale identities, or neural pricing. A good match between this Monte Carlo estimate and the analytical formula is useful but not sufficient to establish numerical correctness of a general HJM engine.

## Next steps

1. Implement integrated short rates and consistent pathwise bond prices.
2. Validate discounted-bond martingales with uncertainty bands.
3. Compare Euler time-step refinements to analytical results.
4. Investigate hedge sensitivity errors and out-of-distribution stress.

No real market data, calibrated parameters, or novel research results are claimed.
