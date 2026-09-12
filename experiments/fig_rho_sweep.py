"""Figure 3: memory / nonlinearity tradeoff.

Two panels sharing the x axis (spectral radius rho in [0.1, 1.6]):
  top    -- linear memory capacity (MC)
  bottom -- NARMA10 test NMSE

Averaged over >= 10 seeds, shaded +/- 1 std.
"""

import matplotlib.pyplot as plt
import numpy as np
from common import DENSITY, LAM, N, SEEDS, T, WASHOUT
from narma_pipeline import narma_trial
from plotstyle import BLUE, ORANGE, save, style_axes

from esn import make_input, make_reservoir, run_esn
from metrics import memory_capacity

ALPHA = 1.0
MAX_DELAY = 40
RHO_GRID = np.linspace(0.1, 1.6, 20)


def main():
    mc_mean, mc_std = [], []
    nmse_mean, nmse_std = [], []
    total_resamples = 0

    for rho in RHO_GRID:
        mc_vals = []
        nmse_vals = []
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            u = rng.uniform(-1.0, 1.0, size=T)
            W = make_reservoir(N, DENSITY, rho, seed=seed)
            Win = make_input(N, scale=1.0, seed=seed)
            states = run_esn(u, W, Win, alpha=ALPHA)
            mc_total, _ = memory_capacity(
                states, u, max_delay=MAX_DELAY, washout=WASHOUT, lam=LAM
            )
            mc_vals.append(mc_total)

            result = narma_trial(rho, ALPHA, LAM, seed, T=T, washout=WASHOUT)
            nmse_vals.append(result["test_nmse"])
            if result["attempts"] > 1:
                total_resamples += 1

        mc_mean.append(np.mean(mc_vals))
        mc_std.append(np.std(mc_vals))
        nmse_mean.append(np.mean(nmse_vals))
        nmse_std.append(np.std(nmse_vals))
        print(f"rho={rho:.3f}  MC={mc_mean[-1]:.2f}+-{mc_std[-1]:.2f}  "
              f"NARMA_test_NMSE={nmse_mean[-1]:.4f}+-{nmse_std[-1]:.4f}")

    mc_mean, mc_std = np.array(mc_mean), np.array(mc_std)
    nmse_mean, nmse_std = np.array(nmse_mean), np.array(nmse_std)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6.0), sharex=True, height_ratios=[1, 1]
    )
    style_axes(ax_top)
    style_axes(ax_bot)

    ax_top.plot(RHO_GRID, mc_mean, color=BLUE, lw=1.8)
    ax_top.fill_between(RHO_GRID, mc_mean - mc_std, mc_mean + mc_std,
                         color=BLUE, alpha=0.2, lw=0)
    ax_top.axvline(1.0, color="#8a8a8a", lw=0.7, ls="--", alpha=0.6)
    ax_top.set_ylabel("linear memory capacity $MC$")

    ax_bot.plot(RHO_GRID, nmse_mean, color=ORANGE, lw=1.8)
    ax_bot.fill_between(RHO_GRID, nmse_mean - nmse_std, nmse_mean + nmse_std,
                         color=ORANGE, alpha=0.2, lw=0)
    ax_bot.axvline(1.0, color="#8a8a8a", lw=0.7, ls="--", alpha=0.6)
    ax_bot.set_ylabel("NARMA10 test NMSE")
    ax_bot.set_xlabel(r"spectral radius $\rho$")
    ax_bot.set_yscale("log")

    fig.tight_layout()
    save(fig, "fig_rho_sweep.png")

    best_idx = int(np.argmin(nmse_mean))
    mc_best_idx = int(np.argmax(mc_mean))
    print(f"\nBest NARMA10 rho (mean over seeds): rho={RHO_GRID[best_idx]:.3f}, "
          f"test_NMSE={nmse_mean[best_idx]:.4f}")
    print(f"Best MC rho (mean over seeds): rho={RHO_GRID[mc_best_idx]:.3f}, "
          f"MC={mc_mean[mc_best_idx]:.3f}")
    print(f"NARMA10 resamples needed (divergence): {total_resamples}")


if __name__ == "__main__":
    main()
