import numpy as np
import pytest

from hjm_neural_rates_lab.curve_risk import (
    discount_factors, instantaneous_forwards, parallel_shift_pv01,
)


def test_flat_curve_discount_factors():
    t = np.array([0., 1., 5.])
    np.testing.assert_allclose(discount_factors(t, 0.03), np.exp(-0.03 * t))


def test_flat_curve_forwards_are_flat():
    t = np.array([0., 0.5, 1., 2., 5.])
    np.testing.assert_allclose(
        instantaneous_forwards(t, np.full_like(t, 0.03)), 0.03, atol=1e-14
    )


def test_quadratic_integrated_zero_curve():
    t = np.linspace(0, 5, 11)
    z = 0.02 + 0.002 * t
    np.testing.assert_allclose(
        instantaneous_forwards(t, z), 0.02 + 0.004 * t, atol=1e-12
    )


def test_pv01_matches_analytic_flat_curve_derivative():
    t = np.array([1., 2., 3.])
    z = np.full(3, 0.04)
    c = np.array([5., 5., 105.])
    pv, pv01 = parallel_shift_pv01(t, z, c)
    expected = -np.sum(t * c * np.exp(-z * t)) * 1e-4
    assert pv == pytest.approx(np.sum(c * np.exp(-z * t)))
    assert pv01 == pytest.approx(expected, rel=1e-7)
    assert pv01 < 0


def test_bad_grid_rejected():
    with pytest.raises(ValueError):
        instantaneous_forwards([0, 1, 1], [0.02, 0.03, 0.04])
    with pytest.raises(ValueError):
        discount_factors([-1], [0.02])
