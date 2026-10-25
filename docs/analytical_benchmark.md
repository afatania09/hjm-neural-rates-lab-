# Analytical Gaussian HJM bond-option benchmark

Let S be option expiry and T>S the underlying zero-coupon bond maturity. For deterministic volatility vectors sigma(t,u), under the S-forward measure the forward bond price F(t)=P(t,T)/P(t,S) has lognormal dynamics. The integrated log variance over [0,S] is

    V = integral_0^S ||integral_S^T sigma(t,u) du||^2 dt.

For independent factors sigma_i(t,u)=a_i exp(-b_i(u-t)), b_i>=0, put D=T-S. Each factor contributes

    V_i = [a_i (1-exp(-b_i D))/b_i]^2 * [ (1-exp(-2b_i S))/(2b_i) ].

The continuous b_i->0 limit is a_i^2 D^2 S.

Define v=sqrt(V), d1=ln(P(0,T)/(K P(0,S)))/v + v/2 and d2=d1-v. The call price per unit face value is

    C(0) = P(0,T) Phi(d1) - K P(0,S) Phi(d2).

The put follows from put-call parity. At V=0, use discounted intrinsic value.

## Scope

- Deterministic, independent-factor Gaussian HJM volatility; no stochastic volatility.
- A European option exercised at S on a zero-coupon bond maturing at T.
- This is not a swaption, a caplet, or a calibration to market data.
- The analytical result provides a separate target for future Monte Carlo validation.
- Tests check the Ho-Lee limit, quadrature consistency, put-call parity, and monotonicity with volatility.

## Next benchmark

Implement a pathwise bond-price and money-market numeraire estimator; compare discounted Monte Carlo option values against this formula over decreasing time steps, independent seeds and confidence intervals. Then audit discounted bond martingales. Do not infer no-arbitrage from matching a single option price.
