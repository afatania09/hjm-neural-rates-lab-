import numpy as np
import pytest

from hjm_neural_rates_lab.swap_pricing import par_swap_rate, payer_swap_pv


def test_par_swap_has_zero_value():
    times = np.arange(1., 6.)
    accruals = np.ones(5)
    rates = np.full(5, 0.035)
    k = par_swap_rate(times, accruals, rates)
    assert payer_swap_pv(times, accruals, rates, k, 1_000_000) == pytest.approx(0, abs=1e-9)


def test_receiver_is_negative_payer_by_definition():
    times = np.array([1., 2., 3.])
    rates = np.full(3, 0.04)
    k = par_swap_rate(times, np.ones(3), rates)
    assert payer_swap_pv(times, np.ones(3), rates, k + 0.01) < 0
    assert payer_swap_pv(times, np.ones(3), rates, k - 0.01) > 0


def test_forward_start_uses_start_discount():
    times = np.array([2., 3.])
    rates = np.array([0.03, 0.03])
    start_df = np.exp(-0.03)
    k = par_swap_rate(times, np.ones(2), rates, start_discount=start_df)
    assert payer_swap_pv(times, np.ones(2), rates, k, start_discount=start_df) == pytest.approx(0, abs=1e-14)


def test_invalid_schedule():
    with pytest.raises(ValueError):
        par_swap_rate([1, 1], [1, 1], [0.02, 0.02])
    with pytest.raises(ValueError):
        par_swap_rate([1, 2], [1, -1], [0.02, 0.02])
