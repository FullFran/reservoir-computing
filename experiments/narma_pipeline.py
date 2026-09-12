"""Shared NARMA10 fit/evaluate pipeline used by several figures
(fig_rho_sweep, fig_narma_pred, fig_ridge, fig_heatmap) so the exact same
protocol -- washout, train/test split, ridge fit, NMSE -- is used
everywhere NARMA10 is benchmarked.
"""

import numpy as np
from common import DENSITY, N

from esn import fit_ridge, make_input, make_reservoir, predict, run_esn
from metrics import nmse
from tasks import sample_narma10

NARMA_SEED_OFFSET = 5000


def narma_trial(rho, alpha, lam, seed, T, washout, test_frac=0.2):
    """Fit and evaluate a ridge readout on NARMA10 for one (rho, alpha, lam, seed).

    Returns a dict with train_nmse, test_nmse, the held-out target/prediction
    sequences (Y_test, Yhat_test), and `attempts` (> 1 if NARMA10 diverged
    and had to be resampled).
    """
    u, y, attempts = sample_narma10(T, seed=NARMA_SEED_OFFSET + seed)

    W = make_reservoir(N, DENSITY, rho, seed=seed)
    Win = make_input(N, scale=1.0, seed=seed)
    states = run_esn(u, W, Win, alpha=alpha)

    X = states[washout:]
    Y = y[washout:]
    n = X.shape[0]
    split = int(n * (1.0 - test_frac))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    Wout = fit_ridge(X_train, Y_train, lam, washout=0)
    Yhat_train = predict(X_train, Wout)
    Yhat_test = predict(X_test, Wout)

    return {
        "train_nmse": nmse(Y_train, Yhat_train),
        "test_nmse": nmse(Y_test, Yhat_test),
        "Y_test": Y_test,
        "Yhat_test": Yhat_test,
        "attempts": attempts,
    }
