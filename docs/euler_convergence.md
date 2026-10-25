# First independent Euler HJM convergence audit

For a single Ho-Lee factor with constant forward volatility a, the risk-neutral HJM drift is alpha(t,T)=a^2(T-t). At fixed absolute maturity T, the exact Gaussian forward rate is

    f(t,T)=f(0,T)+a^2(Tt-t^2/2)+a W_t.

For N uniform left-endpoint Euler steps of length dt=t/N, the expected numerical forward rate is

    E[f_N(t,T)] = E[f(t,T)] + a^2 t dt/2.

Therefore the deterministic weak bias is positive and decreases linearly with dt. Its exact coefficient is known, and does not rely on comparing two simulated results. This is a direct audit of the original fixed-maturity Euler simulator, unlike the earlier exact terminal-law benchmarks.

The test compares the sampled Euler mean against its **discrete** expectation within sampling error. The example reports both the exact and discretised theoretical means.

## Interpretation

- The expected drift bias has first-order time-step convergence.
- The stochastic increment a W_t is exact in distribution at observation time for constant a; its sampling error does not disappear merely by refining the time grid.
- This is a single-factor, flat-initial-curve Ho-Lee benchmark; it is not yet a full multi-factor risk-neutral numeraire or discounted-bond martingale audit.
- A visual reduction in raw Monte Carlo error is not guaranteed without common random numbers or sufficiently many paths.

Next: construct a genuine pathwise money-market account from short-rate paths and price bonds on a maturity grid. Quantify both time-step and maturity-grid errors.
