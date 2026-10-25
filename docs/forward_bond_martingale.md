# Forward-bond martingale benchmark

For S<T, the forward bond F(t;S,T)=P(t,T)/P(t,S) is a martingale under the S-forward measure for 0<=t<=S, subject to usual integrability assumptions.

In deterministic-volatility Gaussian HJM its log dynamics are Gaussian. This module uses exact **interval log-variance** for each observation interval; it is not Euler integration of the full forward curve.

## Validation checks

1. Initial forward bond ratio matches P(0,T)/P(0,S).
2. Cumulative integrated variance at S matches the independent bond-option formula.
3. Zero volatility yields a deterministic constant forward bond.
4. Sample mean at multiple dates is within a prespecified multiple of its standard error from the theoretical initial forward.
5. A simulated terminal option payoff agrees statistically with the analytical European bond-option price.

## Scope and nonclaims

This is a martingale test **under the S-forward measure**. It does not verify discounted bond prices under the risk-neutral money-market measure, simulate short rates, or test the previously implemented fixed-maturity Euler forward-rate engine. It is a meaningful intermediate benchmark, not a substitute for those audits.

The confidence-based tests are stochastic, with fixed seeds; their tolerances reflect sampling uncertainty. Future work will assess repeated-seed coverage and sensitivity to simulation size.
