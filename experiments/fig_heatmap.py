"""Figure 8: NARMA10 test NMSE heatmap over (rho, alpha).

12x12 grid, averaged over >= 10 seeds per cell. Uses a perceptually
uniform colormap (viridis) with the color scale clipped so structure near
the optimum is visible instead of being washed out by a few very bad
(high-rho, high-alpha) cells.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from common import LAM, SEEDS, T, WASHOUT
from narma_pipeline import narma_trial
from plotstyle import GREY, save

N_GRID = 12
RHO_GRID = np.linspace(0.1, 1.6, N_GRID)
ALPHA_GRID = np.linspace(0.1, 1.0, N_GRID)


def main():
    nmse_grid = np.zeros((N_GRID, N_GRID))  # [alpha_idx, rho_idx]

    for i, alpha in enumerate(ALPHA_GRID):
        for j, rho in enumerate(RHO_GRID):
            vals = []
            for seed in SEEDS:
                result = narma_trial(rho, alpha, LAM, seed, T=T, washout=WASHOUT)
                vals.append(result["test_nmse"])
            nmse_grid[i, j] = np.mean(vals)
        print(f"alpha={alpha:.3f} done: "
              f"row min={nmse_grid[i].min():.4f} max={nmse_grid[i].max():.4f}")

    np.save("experiments/_heatmap_grid.npy", nmse_grid)

    flat_idx = np.argmin(nmse_grid)
    best_i, best_j = np.unravel_index(flat_idx, nmse_grid.shape)
    best_alpha = ALPHA_GRID[best_i]
    best_rho = RHO_GRID[best_j]
    best_nmse = nmse_grid[best_i, best_j]
    print(f"\nBest cell: rho={best_rho:.3f}, alpha={best_alpha:.3f}, "
          f"test_NMSE={best_nmse:.4f}")

    with open("experiments/_heatmap_best.json", "w") as f:
        json.dump({"rho": float(best_rho), "alpha": float(best_alpha),
                   "test_nmse": float(best_nmse)}, f)

    # clip the color scale: NMSE >= 1 means "no better than predicting the
    # mean", so anything at/above that is uninformative -- clip there to
    # keep contrast in the region that matters.
    vmax = min(1.0, np.percentile(nmse_grid, 90))
    vmin = nmse_grid.min()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        nmse_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[RHO_GRID[0], RHO_GRID[-1], ALPHA_GRID[0], ALPHA_GRID[-1]],
    )
    ax.plot(best_rho, best_alpha, marker="*", color="white", markersize=14,
            markeredgecolor=GREY, markeredgewidth=0.8)

    ax.set_xlabel(r"spectral radius $\rho$")
    ax.set_ylabel(r"leak rate $\alpha$")
    ax.tick_params(colors=GREY)
    for spine in ax.spines.values():
        spine.set_color(GREY)

    cbar = fig.colorbar(im, ax=ax, extend="max")
    cbar.set_label("NARMA10 test NMSE (clipped)", color=GREY)
    cbar.ax.yaxis.set_tick_params(color=GREY)
    plt.setp(cbar.ax.get_yticklabels(), color=GREY)
    cbar.outline.set_edgecolor(GREY)

    save(fig, "fig_heatmap.png")


if __name__ == "__main__":
    main()
