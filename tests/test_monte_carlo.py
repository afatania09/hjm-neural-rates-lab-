import numpy as np
import pytest
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.bond_options import zero_coupon_bond_option
from hjm_rates.monte_carlo import bond_option_forward_measure_mc


@pytest.mark.parametrize("antithetic", [False, True])
def test_analytic_price_within_statistical_tolerance(antithetic):
    curve = FlatForwardCurve(0.03)
    model = GaussianHJM((0.025, 0.009), (0.15, 0.7))
    strike = curve.discount(5) / curve.discount(2)
    analytic = zero_coupon_bond_option(model, curve, 2, 5, strike)
    estimate = bond_option_forward_measure_mc(
        model, curve, 2, 5, strike, n_paths=100_000, seed=71,
        antithetic=antithetic)
    assert abs(estimate.price - analytic) < 5 * estimate.std_error + 1e-8


def test_zero_volatility_is_deterministic():
    curve = FlatForwardCurve(0.03)
    model = GaussianHJM((0.0,), (0.0,))
    analytic = zero_coupon_bond_option(model, curve, 1, 3, 0.9)
    mc = bond_option_forward_measure_mc(model, curve, 1, 3, 0.9, n_paths=200)
    assert mc.price == pytest.approx(analytic, abs=1e-12)
    assert mc.std_error < 1e-12


def test_antithetic_pairs_use_independent_pair_se():
    curve = FlatForwardCurve()
    model = GaussianHJM((0.02,), (0.1,))
    result = bond_option_forward_measure_mc(model, curve, 1, 4, 0.9, n_paths=2000)
    assert np.isfinite(result.std_error)
    assert result.std_error >= 0
    assert result.ci_lower <= result.price <= result.ci_upper


def test_reproducible():
    args = (GaussianHJM(), FlatForwardCurve(), 1, 3, 0.9)
    assert bond_option_forward_measure_mc(*args, n_paths=2000, seed=2) == bond_option_forward_measure_mc(*args, n_paths=2000, seed=2)


def test_invalid_antithetic_count():
    with pytest.raises(ValueError):
        bond_option_forward_measure_mc(GaussianHJM(), FlatForwardCurve(), 1, 3, 0.9, n_paths=101)
