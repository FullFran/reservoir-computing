"""Minimal, from-scratch Echo State Network (ESN) building blocks.

This module implements the core pieces of a leaky-integrator Echo State
Network using only NumPy:

- a sparse random reservoir matrix with a prescribed spectral radius,
- a random input weight vector,
- the leaky-integrator state update,
- ridge-regression readout fitting and prediction.

No reservoir-computing library is used anywhere in this project; every
equation below is written out explicitly so the numbers can be checked by
hand.
"""

from __future__ import annotations

import numpy as np


def make_reservoir(n: int, density: float, rho: float, seed: int) -> np.ndarray:
    """Build an n x n sparse random reservoir matrix with spectral radius rho.

    Construction:
    1. Draw a dense n x n matrix of i.i.d. standard Gaussian entries.
    2. Draw an independent Bernoulli(density) mask (entry kept with
       probability `density`) and multiply element-wise -> a sparse random
       matrix with i.i.d. Gaussian nonzero entries.
    3. Rescale the whole matrix by a single scalar so that its spectral
       radius (the largest magnitude eigenvalue) equals `rho` exactly.

    Parameters
    ----------
    n : size of the square reservoir matrix.
    density : probability that a given entry is nonzero, in (0, 1].
    rho : target spectral radius (rho = 0 is degenerate and not supported
        since an all-zero matrix cannot be rescaled).
    seed : seed for the local random generator (reproducible).

    Returns
    -------
    W : (n, n) ndarray with spectral_radius(W) == rho (up to floating point
        precision).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not (0.0 < density <= 1.0):
        raise ValueError("density must be in (0, 1]")

    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((n, n))
    mask = rng.random((n, n)) < density
    W = gaussian * mask

    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius == 0.0:
        raise ValueError(
            "sampled reservoir has spectral radius 0 (all entries zero); "
            "increase density or re-seed"
        )
    W = W * (rho / current_radius)
    return W


def make_input(n: int, scale: float, seed: int) -> np.ndarray:
    """Build a random input weight vector of length n.

    Entries are i.i.d. Uniform[-scale, scale].

    Parameters
    ----------
    n : reservoir size (length of the returned vector).
    scale : half-width of the uniform distribution.
    seed : seed for the local random generator (reproducible).

    Returns
    -------
    Win : (n,) ndarray.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=n)


def run_esn(
    u: np.ndarray,
    W: np.ndarray,
    Win: np.ndarray,
    alpha: float,
    bias: float = 0.0,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Run the leaky-integrator ESN forward and return the state trajectory.

    Update equation, applied for t = 0, ..., T-1, starting from x_{-1} = x0:

        x_t = (1 - alpha) * x_{t-1} + alpha * tanh(W @ x_{t-1} + Win * u_t + bias)

    i.e. at each step the reservoir consumes u_t and the resulting state
    x_t (stored at row t of the output) already reflects u_t itself, not
    just the inputs strictly before it. This is the standard convention
    used for memory-capacity tasks: reconstructing u_t from x_t (delay
    k = 0) is the easiest case (x_t contains Win * u_t directly, inside
    the tanh), and reconstructing u_{t-k} gets harder as k grows.

    Parameters
    ----------
    u : (T,) ndarray, scalar input sequence u_0, ..., u_{T-1}.
    W : (n, n) reservoir matrix.
    Win : (n,) input weight vector.
    alpha : leak rate in (0, 1]. alpha = 1 recovers the standard
        (non-leaky) ESN update.
    bias : scalar bias added inside the tanh nonlinearity.
    x0 : optional (n,) state before any input is consumed (i.e. x_{-1}).
        Defaults to the zero vector.

    Returns
    -------
    states : (T, n) ndarray with states[t] = x_t, the state immediately
        after consuming u_0, ..., u_t (inclusive).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    n = W.shape[0]
    T = u.shape[0]
    states = np.empty((T, n), dtype=float)

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    for t in range(T):
        pre_activation = W @ x + Win * u[t] + bias
        x = (1.0 - alpha) * x + alpha * np.tanh(pre_activation)
        states[t] = x
    return states


def _design_matrix(X: np.ndarray) -> np.ndarray:
    """Prepend a constant bias column of ones to a design matrix X."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float, washout: int) -> np.ndarray:
    """Fit a ridge-regression readout Wout from states X to targets Y.

    A bias column of ones is always prepended to X before fitting, and the
    first `washout` rows of X and Y are always discarded before fitting
    (washout is a mandatory, explicit argument -- it is never applied
    silently).

    The readout solves:

        Wout = argmin_W  ||Y - [1, X] @ W||^2 + lam * ||W||^2

    via the closed-form ridge solution
    Wout = (Z^T Z + lam * I)^-1 Z^T Y, where Z = [1, X] is the
    bias-augmented design matrix. The bias row/column of the identity is
    NOT regularized-away specially here (a single lam is used for all
    coefficients including the bias term), which is the common minimal
    convention.

    Parameters
    ----------
    X : (T, n) reservoir states.
    Y : (T,) or (T, k) targets.
    lam : ridge regularization strength (lam >= 0). lam = 0 is ordinary
        least squares (only well posed if Z^T Z is invertible).
    washout : number of initial rows to discard from X and Y before
        fitting. Must be explicitly provided (no default) and satisfy
        0 <= washout < T.

    Returns
    -------
    Wout : (n + 1, k) ndarray (or (n + 1,) if Y is 1-D), where row 0 is the
        bias coefficient and rows 1..n are the coefficients for each
        reservoir unit.
    """
    if washout < 0 or washout >= X.shape[0]:
        raise ValueError("washout must satisfy 0 <= washout < T")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    X_use = X[washout:]
    Y_use = Y[washout:]

    Z = _design_matrix(X_use)
    n_features = Z.shape[1]
    reg = lam * np.eye(n_features)
    Wout = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y_use)
    return Wout


def predict(X: np.ndarray, Wout: np.ndarray) -> np.ndarray:
    """Apply a readout fitted by fit_ridge to (possibly new) states X.

    Prepends the same bias column of ones used during fitting, then
    returns Z @ Wout.

    Parameters
    ----------
    X : (T, n) reservoir states.
    Wout : (n + 1, k) or (n + 1,) readout weights from fit_ridge.

    Returns
    -------
    yhat : (T,) or (T, k) predictions, matching the shape convention of
        Wout.
    """
    Z = _design_matrix(X)
    return Z @ Wout
