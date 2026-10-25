import numpy as np
import pytest
from hjm_rates import FlatForwardCurve, GaussianHJM, bond_price_from_forwards, simulate_forward_paths


def test_flat_curve_bond_identity():
    grid = np.linspace(0, 5, 101)
    curve = FlatForwardCurve(0.04)
    assert bond_price_from_forwards(grid, curve.forward(grid)) == pytest.approx(np.exp(-0.2), abs=1e-12)


def test_hjm_drift_zero_at_current_maturity():
    model = GaussianHJM((0.01, 0.02), (0.1, 0.3))
    assert model.drift(1.0, np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-14)


def test_constant_volatility_drift_formula():
    model = GaussianHJM((0.02,), (0.0,))
    maturities = np.array([1.0, 2.0, 5.0])
    assert np.allclose(model.drift(0, maturities), 0.02**2 * maturities)


def test_exponential_vol_integral():
    model = GaussianHJM((0.02,), (0.5,))
    result = model.integrated_volatility(0, np.array([2.0]))[0, 0]
    assert result == pytest.approx(0.02 * (1 - np.exp(-1)) / 0.5)


def test_reproducibility_and_shape():
    m = GaussianHJM()
    t = np.array([0., 0.25, 0.5])
    T = np.linspace(0, 2, 9)
    a = simulate_forward_paths(m, FlatForwardCurve(), T, t, n_paths=20, seed=11)
    b = simulate_forward_paths(m, FlatForwardCurve(), T, t, n_paths=20, seed=11)
    assert a.shape == (20, 3, 9)
    np.testing.assert_array_equal(a, b)
    assert np.isnan(a[:, -1, 0]).all()
    assert np.isfinite(a[:, -1, -1]).all()


def test_zero_volatility_keeps_flat_forward_curve():
    m = GaussianHJM((0.0,), (0.1,))
    T = np.linspace(0, 3, 13)
    times = np.linspace(0, 1, 5)
    paths = simulate_forward_paths(m, FlatForwardCurve(0.03), T, times, n_paths=10)
    assert np.allclose(paths[np.isfinite(paths)], 0.03)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        GaussianHJM((0.01,), (0.1, 0.2))
    with pytest.raises(ValueError):
        simulate_forward_paths(GaussianHJM(), FlatForwardCurve(), [0, 1], [0.1, 0.2])
