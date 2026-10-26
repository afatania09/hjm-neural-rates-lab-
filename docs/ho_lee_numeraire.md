# Independently integrated Ho-Lee money-market numeraire

With a flat initial instantaneous forward curve r0 and constant HJM volatility a, the exact risk-neutral forward curve is

    f(t,T)=r0 + a^2(Tt - t^2/2) + a W_t.

Thus the short rate is r(t)=f(t,t)=r0+a^2 t^2/2+a W_t and the discount factor is

    D(t)=exp[-r0*t - a^2*t^3/6 - a*integral_0^t W_s ds].

For each time step of length h, conditional on its Brownian endpoints,

    integral_{t_j}^{t_{j+1}} W_s ds
      = h*(W_j+W_{j+1})/2 + sqrt(h^3/12)*Z_j,

where Z_j is independent standard normal. This **exact Brownian bridge area** eliminates the trapezoidal short-rate integration bias.

Bond prices are computed by analytically integrating f(t,u) from t to T. The program then audits

    E_Q[D(t) P(t,T)] = P(0,T)

at several observation times with Monte Carlo standard errors.

This is the first actual *pathwise* integrated short-rate numeraire in the repository. It is an exact finite-grid Ho-Lee benchmark, **not** a numerical integration of the existing Euler simulator, and does not establish multi-factor arbitrage freedom. The next experiment must use the Euler forward-rate grid itself to build short rates and bond prices and quantify maturity/time-grid discretisation error.
