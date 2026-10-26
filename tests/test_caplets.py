import pytest
from hjm_rates import FlatForwardCurve, GaussianHJM
from hjm_rates.caplets import caplet_price, floorlet_price, caplet_floorlet_quote


def test_caplet_floorlet_parity():
    m = GaussianHJM((0.015, 0.01), (0.1, 0.5))
    c = FlatForwardCurve(0.03)
    quote = caplet_floorlet_quote(m, c, 2, 2.5, 0.035, 1_000_000)
    assert quote.parity_residual == pytest.approx(0.0, abs=1e-8)


def test_zero_volatility_caplet_is_discounted_forward_intrinsic():
    c = FlatForwardCurve(0.04)
    m = GaussianHJM((0.0,), (0.0,))
    quote = caplet_floorlet_quote(m, c, 1, 1.5, 0.02)
    assert quote.caplet == pytest.approx(
        0.5 * c.discount(1.5) * max(quote.forward_rate - 0.02, 0), abs=1e-12)


def test_volatility_increases_atm_caplet():
    c = FlatForwardCurve(0.03)
    delta = 0.5
    forward = (c.discount(1)/c.discount(1.5)-1)/delta
    low = caplet_price(GaussianHJM((0.005,), (0.2,)), c, 1, 1.5, forward)
    high = caplet_price(GaussianHJM((0.04,), (0.2,)), c, 1, 1.5, forward)
    assert high > low > 0


def test_caplet_decreases_with_strike():
    m = GaussianHJM((0.02,), (0.2,))
    c = FlatForwardCurve()
    assert caplet_price(m, c, 1, 1.5, 0.02) > caplet_price(m, c, 1, 1.5, 0.05)


def test_notional_scaling():
    m, c = GaussianHJM(), FlatForwardCurve()
    assert caplet_price(m, c, 1, 1.5, 0.03, 1_000_000) == pytest.approx(
        1_000_000 * caplet_price(m, c, 1, 1.5, 0.03))


@pytest.mark.parametrize("reset,payment,strike,notional", [
    (1, 1, 0.03, 1), (2, 1, 0.03, 1),
    (1, 1.5, -3.0, 1), (1, 1.5, 0.03, 0),
])
def test_invalid_contract(reset, payment, strike, notional):
    with pytest.raises(ValueError):
        caplet_price(GaussianHJM(), FlatForwardCurve(), reset, payment, strike, notional)
