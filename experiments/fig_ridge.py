"""Figure 7: train/test NMSE on NARMA10 vs ridge regularization lambda.

Shows the overfitting regime at very small lambda (near-perfect train fit,
poor generalization) and the underfitting regime at very large lambda.
Uses the best (rho, alpha) found by fig_heatmap.py. Averaged over >= 10
seeds, shaded +/- 1 std.
"""

import json
import pathlib

import numpy as np
from common import SEEDS, T, WASHOUT
from narma_pipeline import narma_trial
from plotstyle import BLUE, ORANGE, new_figure, save

LAM_GRID = np.logspace(-12, 0, 13)

BEST_JSON = pathlib.Path(__file__).parent / "_heatmap_best.json"
DEFAULT_RHO, DEFAULT_ALPHA = 0.8, 0.9


def main():
    if BEST_JSON.exists():
        best = json.loads(BEST_JSON.read_text())
        rho, alpha = best["rho"], best["alpha"]
    else:
        rho, alpha = DEFAULT_RHO, DEFAULT_ALPHA
    print(f"using (rho, alpha) = ({rho:.3f}, {alpha:.3f})")

    train_mean, train_std, test_mean, test_std = [], [], [], []
    for lam in LAM_GRID:
        train_vals, test_vals = [], []
        for seed in SEEDS:
            result = narma_trial(rho, alpha, lam, seed, T=T, washout=WASHOUT)
            train_vals.append(result["train_nmse"])
            test_vals.append(result["test_nmse"])
        train_mean.append(np.mean(train_vals))
        train_std.append(np.std(train_vals))
        test_mean.append(np.mean(test_vals))
        test_std.append(np.std(test_vals))
        print(f"lam={lam:.1e}  train={train_mean[-1]:.4f}  test={test_mean[-1]:.4f}")

    train_mean, train_std = np.array(train_mean), np.array(train_std)
    test_mean, test_std = np.array(test_mean), np.array(test_std)

    fig, ax = new_figure()
    ax.plot(LAM_GRID, train_mean, color=BLUE, lw=1.8, label="train NMSE")
    ax.fill_between(LAM_GRID, train_mean - train_std, train_mean + train_std,
                     color=BLUE, alpha=0.2, lw=0)
    ax.plot(LAM_GRID, test_mean, color=ORANGE, lw=1.8, label="test NMSE")
    ax.fill_between(LAM_GRID, test_mean - test_std, test_mean + test_std,
                     color=ORANGE, alpha=0.2, lw=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"ridge regularization $\lambda$")
    ax.set_ylabel("NARMA10 NMSE")
    ax.legend(frameon=False)

    save(fig, "fig_ridge.png")

    best_test_idx = int(np.argmin(test_mean))
    print(f"\nBest test NMSE at lambda={LAM_GRID[best_test_idx]:.1e}: "
          f"{test_mean[best_test_idx]:.4f} (train={train_mean[best_test_idx]:.4f})")
    print(f"Smallest lambda={LAM_GRID[0]:.1e}: train={train_mean[0]:.5f}, "
          f"test={test_mean[0]:.4f} (overfitting gap)")


if __name__ == "__main__":
    main()
