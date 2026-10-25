import numpy as np
import pytest
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.bond_options import forward_bond_log_variance, zero_coupon_bond_option


def test_ho_lee_variance_exact():
    model = GaussianHJM((0.02,), (0.0,))
    assert forward_bond_log_variance(model, 2, 5) == pytest.approx(0.02**2 * 3**2 * 2)


def test_exponential_variance_matches_numerical_quadrature():
    model = GaussianHJM((0.012, 0.007), (0.2, 0.9))
    expiry, maturity = 1.5, 4.0
    grid = np.linspace(0, expiry, 10001)
    exposure = np.array([
        [a * np.exp(-b * (expiry - t)) * (-np.expm1(-b * (maturity - expiry))) / b
         for a, b in zip(model.amplitudes, model.decays)]
        for t in grid
    ])
    numeric = np.trapz(np.sum(exposure**2, axis=1), grid)
    assert forward_bond_log_variance(model, expiry, maturity) == pytest.approx(numeric, rel=1e-7)


def test_call_put_parity():
    model = GaussianHJM((0.015,), (0.2,))
    curve = FlatForwardCurve(0.03)
    s, t, k = 2, 5, 0.9
    call = zero_coupon_bond_option(model, curve, s, t, k)
    put = zero_coupon_bond_option(model, curve, s, t, k, "put")
    assert call - put == pytest.approx(curve.discount(t) - k * curve.discount(s), abs=1e-12)


def test_zero_volatility_intrinsic():
    model = GaussianHJM((0.0,), (0.0,))
    curve = FlatForwardCurve(0.03)
    s, t, k = 1, 2, 0.9
    intrinsic = curve.discount(t) - k * curve.discount(s)
    assert zero_coupon_bond_option(model, curve, s, t, k) == pytest.approx(max(intrinsic, 0))


def test_more_volatility_increases_atm_option_value():
    curve = FlatForwardCurve(0.04)
    s, t = 1, 4
    k = curve.discount(t) / curve.discount(s)
    low = zero_coupon_bond_option(GaussianHJM((0.005,), (0.1,)), curve, s, t, k)
    high = zero_coupon_bond_option(GaussianHJM((0.03,), (0.1,)), curve, s, t, k)
    assert high > low > 0


def test_invalid_contract_rejected():
    with pytest.raises(ValueError):
        forward_bond_log_variance(GaussianHJM(), 2, 1)
    with pytest.raises(ValueError):
        zero_coupon_bond_option(GaussianHJM(), FlatForwardCurve(), 1, 3, -1)
