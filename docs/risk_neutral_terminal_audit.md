# Risk-neutral terminal-law audit

This experiment introduces the **joint distribution** of the money-market discount factor D(S) and a terminal zero-coupon bond price P(S,T) under Q.

For deterministic Gaussian HJM volatility, their stochastic log components are Gaussian with kernels:

- log D(S): minus integral_u^S sigma(u,v) dv;
- log P(S,T): minus integral_S^T sigma(u,v) dv.

Covariance is obtained by integrating the products of these kernels from u=0 to S. Deterministic log means are fixed using

    E_Q[D(S)] = P(0,S),
    E_Q[D(S) P(S,T)] = P(0,T).

The simulation estimates both identities and discounted European option payoffs with sampling standard errors.

**Important:** this constructs an exact joint terminal law; it does not integrate a simulated short-rate path or independently verify the Euler HJM simulator. The no-arbitrage identities are enforced in the construction, so agreement is an implementation consistency test rather than an independent proof of arbitrage freedom.

Next: build a genuine short-rate/forward-curve path discretisation and investigate convergence of discounted bond estimates against these references.
