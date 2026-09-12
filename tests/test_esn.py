"""Correctness tests for the from-scratch ESN implementation.

Run with: uv run pytest -q
"""

import numpy as np
import pytest

from esn import fit_ridge, make_reservoir, predict
from metrics import memory_capacity
from tasks import narma10
from esn import make_input, run_esn


def test_spectral_radius_matches_target():
    for rho in (0.3, 0.9, 1.25):
        W = make_reservoir(n=200, density=0.1, rho=rho, seed=0)
        actual_radius = np.max(np.abs(np.linalg.eigvals(W)))
        assert actual_radius == pytest.approx(rho, abs=1e-8)


def test_narma10_hand_computed_first_step():
    # y_0..y_9 = 0 by construction, so the recursion's first nontrivial
    # output, y_10, collapses to:
    #   y_10 = 0.3*y_9 + 0.05*y_9*sum(y_0..y_9) + 1.5*u_0*u_9 + 0.1
    #        = 0 + 0 + 1.5*u_0*u_9 + 0.1
    T = 15
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 0.5, size=T)
    y = narma10(u)

    expected_y10 = 1.5 * u[0] * u[9] + 0.1
    assert y[10] == pytest.approx(expected_y10, abs=1e-12)
    # everything before index 10 must still be exactly zero
    assert np.all(y[:10] == 0.0)


def test_fit_ridge_recovers_exact_linear_coefficients():
    rng = np.random.default_rng(1)
    T, n = 500, 10
    X = rng.standard_normal((T, n))
    true_bias = 0.7
    true_w = rng.standard_normal(n)
    Y = true_bias + X @ true_w  # noiseless linear target

    # lam ~ 0 (tiny, for numerical conditioning) recovers exact OLS coefficients
    Wout = fit_ridge(X, Y, lam=1e-10, washout=0)
    assert Wout[0] == pytest.approx(true_bias, abs=1e-6)
    assert np.allclose(Wout[1:], true_w, atol=1e-6)

    Yhat = predict(X, Wout)
    assert np.allclose(Yhat, Y, atol=1e-5)


def test_fit_ridge_washout_is_explicit_and_discards_prefix():
    rng = np.random.default_rng(2)
    T, n = 100, 5
    X = rng.standard_normal((T, n))
    w = rng.standard_normal(n)
    Y = 0.1 + X @ w
    # corrupt the first 20 rows so that including them would break the fit
    Y_corrupted = Y.copy()
    Y_corrupted[:20] = 1000.0

    Wout = fit_ridge(X, Y_corrupted, lam=1e-10, washout=20)
    Yhat = predict(X[20:], Wout)
    assert np.allclose(Yhat, Y[20:], atol=1e-4)

    with pytest.raises(TypeError):
        # washout has no default: calling without it must fail, not
        # silently apply washout=0.
        fit_ridge(X, Y_corrupted, lam=1e-10)


def test_memory_capacity_higher_for_larger_rho():
    n = 200
    T = 4000
    washout = 200
    Win = make_input(n, scale=1.0, seed=7)

    rng = np.random.default_rng(123)
    u = rng.uniform(-1.0, 1.0, size=T)

    W_low = make_reservoir(n, density=0.1, rho=0.05, seed=7)
    W_high = make_reservoir(n, density=0.1, rho=0.9, seed=7)

    states_low = run_esn(u, W_low, Win, alpha=1.0)
    states_high = run_esn(u, W_high, Win, alpha=1.0)

    mc_low, _ = memory_capacity(states_low, u, max_delay=30, washout=washout, lam=1e-6)
    mc_high, _ = memory_capacity(states_high, u, max_delay=30, washout=washout, lam=1e-6)

    assert mc_high > mc_low
