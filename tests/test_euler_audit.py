import pytest
from hjm_rates.euler_audit import ho_lee_forward_audit


def test_ho_lee_predicted_bias_halves_with_step_size():
    coarse = ho_lee_forward_audit(n_steps=10, n_paths=500, seed=9)
    fine = ho_lee_forward_audit(n_steps=20, n_paths=500, seed=9)
    assert coarse.predicted_drift_bias == pytest.approx(2 * fine.predicted_drift_bias)
    assert coarse.predicted_drift_bias > 0


def test_ho_lee_simulated_mean_matches_discrete_euler_expectation():
    r = ho_lee_forward_audit(n_steps=20, n_paths=20000, seed=42)
    assert abs(r.observed_minus_euler_mean) < 5 * r.sampling_se + 1e-12


def test_zero_volatility_exact_at_any_grid_resolution():
    r = ho_lee_forward_audit(volatility=0, n_steps=3, n_paths=100)
    assert r.sample_mean == pytest.approx(r.theoretical_mean, abs=1e-12)
    assert r.predicted_drift_bias == 0


def test_invalid_audit_inputs():
    with pytest.raises(ValueError):
        ho_lee_forward_audit(n_steps=0)
    with pytest.raises(ValueError):
        ho_lee_forward_audit(maturity=1, observation_time=2)
