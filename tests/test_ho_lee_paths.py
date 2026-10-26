import numpy as np
import pytest
from hjm_rates.ho_lee_paths import simulate_ho_lee_paths


def test_zero_volatility_deterministic():
    grid = np.array([0., 0.25, 1., 2.])
    p = simulate_ho_lee_paths(0.03, 0, 4, grid, 100)
    assert np.allclose(p.short_rates, 0.03)
    assert np.allclose(p.discount_factors, np.exp(-0.03 * grid))
    assert np.allclose(p.bond_prices, np.exp(-0.03 * (4-grid)))


def test_initial_conditions_and_shapes():
    p = simulate_ho_lee_paths(0.04, 0.02, 5, np.linspace(0, 2, 9), 500)
    assert p.bond_prices.shape == (500, 9)
    assert np.allclose(p.discount_factors[:, 0], 1)
    assert np.allclose(p.bond_prices[:, 0], np.exp(-0.04*5))
    assert np.allclose(p.short_rates[:, 0], 0.04)


def test_discounted_bond_martingale_under_q():
    p = simulate_ho_lee_paths(0.03, 0.025, 5, np.linspace(0, 2, 9), 50000, 19)
    reference = np.exp(-0.03*5)
    for j in (2, 4, 6, 8):
        values = p.discount_factors[:, j] * p.bond_prices[:, j]
        se = np.std(values, ddof=1)/np.sqrt(len(values))
        assert abs(np.mean(values)-reference) < 5*se + 1e-10


def test_discount_factor_expectation():
    grid = np.linspace(0, 2, 9)
    p = simulate_ho_lee_paths(0.03, 0.02, 5, grid, 50000, 7)
    values = p.discount_factors[:, -1]
    se = np.std(values, ddof=1)/np.sqrt(len(values))
    assert abs(np.mean(values)-np.exp(-0.03*2)) < 5*se+1e-10


def test_reproducibility():
    args = (0.03, 0.02, 4, np.linspace(0, 1, 5), 100, 19)
    a, b = simulate_ho_lee_paths(*args), simulate_ho_lee_paths(*args)
    np.testing.assert_array_equal(a.discount_factors, b.discount_factors)
    np.testing.assert_array_equal(a.bond_prices, b.bond_prices)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_ho_lee_paths(0.03, 0.02, 1, [0, 1])
    with pytest.raises(ValueError):
        simulate_ho_lee_paths(0.03, -0.02, 4, [0, 1])
