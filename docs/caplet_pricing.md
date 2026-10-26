# Gaussian HJM caplet/floorlet pricing

A simple-compounded forward rate over [S,T] is

    L(S,T) = (1/P(S,T) - 1)/(T-S).

With delta=T-S, strike K, and payment at T, a caplet pays

    N delta max(L(S,T)-K,0).

At reset S its present value is

    N (1+delta*K) max(1/(1+delta*K) - P(S,T),0).

Thus the time-zero caplet price is N(1+delta*K) times the **put** price on the T-maturity zero-coupon bond, expiring at S, with strike 1/(1+delta*K). The floorlet is the corresponding scaled bond **call**.

Put-call parity gives

    Caplet - Floorlet = N[P(0,S) - (1+delta*K)P(0,T)].

These formulas assume a single-curve framework, no credit or collateral basis, a simple-compounded rate, and payment at T. They do not model multi-curve RFR conventions, fixing lags, payment delays, smile calibration or market day-count calendars. The demonstration uses synthetic parameters and is not a tradable quote.
