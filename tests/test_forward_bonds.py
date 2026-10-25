import numpy as np
import pytest
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.forward_bonds import simulate_forward_bond_paths
from hjm_rates.bond_options import forward_bond_log_variance, zero_coupon_bond_option


def test_forward_bond_initial_value_and_variance():
    m = GaussianHJM((0.02, 0.008), (0.1, 0.6))
    c = FlatForwardCurve(0.035)
    result = simulate_forward_bond_paths(m, c, 2, 5, np.linspace(0, 2, 9), 500, 4)
    assert result.prices.shape == (500, 9)
    assert np.allclose(result.prices[:, 0], c.discount(5)/c.discount(2))
    assert result.integrated_variances[-1] == pytest.approx(forward_bond_log_variance(m, 2, 5), rel=1e-10)
    assert np.all(np.diff(result.integrated_variances) >= 0)


def test_zero_volatility_constant_paths():
    result = simulate_forward_bond_paths(GaussianHJM((0.0,), (0.0,)),
                                         FlatForwardCurve(0.03), 1, 4,
                                         np.linspace(0, 1, 6), 50)
    assert np.allclose(result.prices, np.exp(-0.03 * 3))


def test_forward_measure_martingale_mean_with_uncertainty():
    m = GaussianHJM((0.015,), (0.2,))
    c = FlatForwardCurve()
    r = simulate_forward_bond_paths(m, c, 1, 4, np.linspace(0, 1, 5), 20000, 71)
    f0 = c.discount(4) / c.discount(1)
    for col in (1, 2, 3, 4):
        standard_error = np.std(r.prices[:, col], ddof=1) / np.sqrt(len(r.prices))
        assert abs(np.mean(r.prices[:, col]) - f0) < 5 * standard_error + 1e-10


def test_option_matches_analytic_with_sampling_error():
    m = GaussianHJM((0.02,), (0.15,))
    c = FlatForwardCurve(0.03)
    strike = c.discount(4)/c.discount(1)
    r = simulate_forward_bond_paths(m, c, 1, 4, np.array([0., 0.2, 0.6, 1.]), 40000, 4)
    payoff = c.discount(1) * np.maximum(r.prices[:, -1] - strike, 0)
    se = np.std(payoff, ddof=1) / np.sqrt(len(payoff))
    analytic = zero_coupon_bond_option(m, c, 1, 4, strike)
    assert abs(np.mean(payoff) - analytic) < 5 * se + 1e-10


def test_invalid_grid_rejected():
    with pytest.raises(ValueError):
        simulate_forward_bond_paths(GaussianHJM(), FlatForwardCurve(), 1, 4, [0, 1.1])
