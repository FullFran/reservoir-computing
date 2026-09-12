"""Evaluation metrics for reservoir computing experiments.

All metrics are plain NumPy; no external ML/RC library is used.
"""

from __future__ import annotations

import numpy as np

from esn import fit_ridge, predict, run_esn
from tasks import delayed_memory


def nmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Normalized mean squared error: mean((y - yhat)^2) / var(y)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean((y - yhat) ** 2) / np.var(y))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot.

    Can be negative if the predictor is worse than predicting the mean of
    y (this is intentional and NOT clipped here -- callers that need a
    non-negative summary, such as memory_capacity, clip explicitly).
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def memory_capacity(
    states: np.ndarray,
    u: np.ndarray,
    max_delay: int,
    washout: int,
    lam: float,
    test_frac: float = 0.5,
):
    """Linear memory capacity via delayed reconstruction, with a proper
    train/test split (never fit and scored on the same data).

    For each delay k = 0, ..., max_delay, a separate ridge readout is
    fit on a training slice to reconstruct u_{t-k} from the reservoir
    state x_t, and R^2 is measured out-of-sample on a held-out test
    slice.

    Parameters
    ----------
    states : (T, n) reservoir states.
    u : (T,) input sequence that drove the reservoir.
    max_delay : largest delay k to test (inclusive), so k = 0..max_delay.
    washout : number of initial transient steps discarded before doing
        anything else (train/test split happens only on the remainder).
    lam : ridge regularization strength used for every delay's readout.
    test_frac : fraction of the post-washout data held out for testing
        (default 0.5, i.e. a 50/50 train/test split).

    Returns
    -------
    mc_total : float, sum over k of max(r2_k, 0) -- the total linear
        memory capacity (always >= 0, as conventional in the RC
        literature).
    per_delay_r2 : (max_delay + 1,) ndarray of raw (unclipped) test-set
        R^2 values, one per delay -- useful for plotting the forgetting
        curve exactly as measured.
    """
    states_use = states[washout:]
    u_use = u[washout:]
    T_use = states_use.shape[0]

    per_delay_r2 = np.zeros(max_delay + 1)
    for k in range(max_delay + 1):
        X_k = states_use[k:]
        y_k = u_use[: T_use - k]
        n = X_k.shape[0]
        split = int(n * (1.0 - test_frac))
        X_train, X_test = X_k[:split], X_k[split:]
        y_train, y_test = y_k[:split], y_k[split:]

        Wout = fit_ridge(X_train, y_train, lam, washout=0)
        yhat_test = predict(X_test, Wout)
        per_delay_r2[k] = r2(y_test, yhat_test)

    mc_total = float(np.sum(np.clip(per_delay_r2, 0.0, None)))
    return mc_total, per_delay_r2


def effective_dimension(states: np.ndarray) -> float:
    """Participation ratio of the state covariance eigenvalue spectrum.

        D_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2

    where lambda_i are the eigenvalues of the (n x n) covariance matrix of
    the reservoir states. D_eff ranges from 1 (all variance in one
    direction) to n (variance spread equally across all directions).

    Parameters
    ----------
    states : (T, n) reservoir states (caller should already have removed
        any washout transient).

    Returns
    -------
    d_eff : float.
    """
    X = states - states.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / X.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    s1 = np.sum(eigvals)
    s2 = np.sum(eigvals**2)
    if s2 == 0.0:
        return 1.0
    return float((s1**2) / s2)


def perturbation_growth(
    u: np.ndarray,
    W: np.ndarray,
    Win: np.ndarray,
    alpha: float,
    eps: float,
    seed: int,
) -> np.ndarray:
    """Distance between two trajectories under identical input, one with
    an eps-perturbed initial state.

    Both trajectories see the exact same input sequence `u`; only the
    initial reservoir state differs, by eps along a random unit direction.
    This isolates the reservoir's own sensitivity to initial conditions
    (a proxy for the largest Lyapunov exponent) from any effect of input
    randomness.

    Parameters
    ----------
    u : (T,) input sequence (shared by both trajectories).
    W, Win, alpha : reservoir parameters (see esn.run_esn).
    eps : magnitude of the initial state perturbation.
    seed : seed for the random perturbation direction.

    Returns
    -------
    dist : (T,) ndarray, dist[t] = ||x_t - x'_t||_2.
    """
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(n)
    direction = direction / np.linalg.norm(direction)

    x0_a = np.zeros(n)
    x0_b = eps * direction

    states_a = run_esn(u, W, Win, alpha, x0=x0_a)
    states_b = run_esn(u, W, Win, alpha, x0=x0_b)

    dist = np.linalg.norm(states_a - states_b, axis=1)
    return dist
