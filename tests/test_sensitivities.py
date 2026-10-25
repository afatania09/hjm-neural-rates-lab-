import pytest
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.sensitivities import parallel_rate_derivative, parallel_rate_derivative_fd, parallel_dv01


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [0.7, 0.85, 0.95])
def test_analytic_rate_sensitivity_matches_fd(kind, strike):
    model = GaussianHJM((0.012, 0.008), (0.2, 0.7))
    curve = FlatForwardCurve(0.035)
    analytic = parallel_rate_derivative(model, curve, 2, 6, strike, kind)
    numeric = parallel_rate_derivative_fd(model, curve, 2, 6, strike, kind, bump=1e-5)
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7)


def test_zero_volatility_deep_in_money_call():
    model = GaussianHJM((0.0,), (0.0,))
    curve = FlatForwardCurve(0.03)
    analytic = parallel_rate_derivative(model, curve, 1, 3, 0.5)
    numeric = parallel_rate_derivative_fd(model, curve, 1, 3, 0.5)
    assert analytic == pytest.approx(numeric, abs=1e-7)


def test_dv01_units():
    model = GaussianHJM((0.01,), (0.1,))
    curve = FlatForwardCurve()
    args = (model, curve, 1, 5, 0.9)
    assert parallel_dv01(*args) == pytest.approx(1e-4 * parallel_rate_derivative(*args))
