import numpy as np
import pytest
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.risk_neutral import terminal_log_covariance, simulate_terminal_bond_risk_neutral
from hjm_rates.bond_options import forward_bond_log_variance, zero_coupon_bond_option


def test_terminal_covariance_positive_semidefinite():
    c = terminal_log_covariance(GaussianHJM((0.02, 0.01), (0.2, 0.7)), 2, 5)
    assert c.shape == (2, 2)
    assert np.min(np.linalg.eigvalsh(c)) >= -1e-12


def test_terminal_bond_variance_matches_forward_measure_benchmark():
    model = GaussianHJM((0.02, 0.01), (0.2, 0.7))
    c = terminal_log_covariance(model, 2, 5)
    assert c[1, 1] == pytest.approx(forward_bond_log_variance(model, 2, 5), rel=1e-10)


def test_zero_volatility_recovers_deterministic_prices():
    curve = FlatForwardCurve(0.03)
    r = simulate_terminal_bond_risk_neutral(GaussianHJM((0.,), (0.,)), curve, 1, 4, 20)
    assert np.allclose(r.discount_factors, curve.discount(1))
    assert np.allclose(r.bond_prices, curve.discount(4)/curve.discount(1))


def test_risk_neutral_discounted_bond_identities():
    curve = FlatForwardCurve(0.03)
    model = GaussianHJM((0.015, 0.008), (0.1, 0.7))
    r = simulate_terminal_bond_risk_neutral(model, curve, 2, 5, 80000, 23)
    for samples, reference in ((r.discount_factors, curve.discount(2)),
                               (r.discount_factors * r.bond_prices, curve.discount(5))):
        se = np.std(samples, ddof=1) / np.sqrt(len(samples))
        assert abs(np.mean(samples) - reference) < 5 * se + 1e-10


def test_risk_neutral_discounted_option_matches_analytic():
    curve = FlatForwardCurve(0.03)
    model = GaussianHJM((0.02,), (0.15,))
    expiry, maturity = 1, 4
    strike = curve.discount(maturity)/curve.discount(expiry)
    r = simulate_terminal_bond_risk_neutral(model, curve, expiry, maturity, 100000, 71)
    payoff = r.discount_factors * np.maximum(r.bond_prices-strike, 0)
    se = np.std(payoff, ddof=1)/np.sqrt(len(payoff))
    analytic = zero_coupon_bond_option(model, curve, expiry, maturity, strike)
    assert abs(np.mean(payoff)-analytic) < 5*se+1e-10


def test_invalid_expiry_rejected():
    with pytest.raises(ValueError):
        terminal_log_covariance(GaussianHJM(), 0, 4)
