import numpy as np
import pytest

from hjm_neural_rates_lab.gaussian_factor_risk import (
    forward_bond_logvariance, forward_bond_mc, martingale_zscore,
)


def test_ho_lee_constant_volatility_limit():
    a, s, t = 0.015, 2., 5.
    expected = a * a * (t - s) ** 2 * s
    assert forward_bond_logvariance(s, t, [a], [0]) == pytest.approx(expected)


def test_exponential_factor_matches_independent_quadrature():
    from scipy.integrate import quad
    s, t, a, b = 2., 7., 0.02, 0.3
    numerical = quad(
        lambda r: (quad(lambda u: a * np.exp(-b * (u - r)), s, t)[0]) ** 2,
        0, s, epsabs=1e-12
    )[0]
    assert forward_bond_logvariance(s, t, [a], [b]) == pytest.approx(numerical, rel=1e-10)


def test_independent_factor_variances_add():
    v1 = forward_bond_logvariance(2, 6, [0.01], [0.1])
    v2 = forward_bond_logvariance(2, 6, [0.02], [0.4])
    assert forward_bond_logvariance(2, 6, [0.01, 0.02], [0.1, 0.4]) == pytest.approx(v1 + v2)


def test_forward_bond_exact_terminal_martingale_statistical():
    samples = forward_bond_mc(1., 5., [0.01, 0.02], [0.0, 0.2],
                              initial_forward_bond=0.88, n_paths=50000, seed=2026)
    assert abs(martingale_zscore(samples, 0.88)) < 4.0


def test_reproducible_and_degenerate_cases():
    a = forward_bond_mc(0, 5, [0.01], [0.1], 0.9, 10, seed=5)
    np.testing.assert_array_equal(a, np.full(10, 0.9))
    np.testing.assert_array_equal(
        forward_bond_mc(1, 5, [0.01], [0.1], 0.9, 10, seed=5),
        forward_bond_mc(1, 5, [0.01], [0.1], 0.9, 10, seed=5)
    )


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        forward_bond_logvariance(3, 2, [0.01], [0.1])
    with pytest.raises(ValueError):
        forward_bond_logvariance(1, 2, [0.01], [-0.1])
    with pytest.raises(ValueError):
        forward_bond_mc(1, 2, [0.01], [0.1], -1)
