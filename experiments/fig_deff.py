"""Figure 6: effective dimension D_eff vs input scaling.

D_eff = participation ratio of the state-covariance eigenvalue spectrum,
for rho in {0.5, 0.95}, as the input weight scale grows from 0.01 to 10
(log-spaced). A horizontal line at N shows the theoretical ceiling.
Strong input drive pushes every neuron toward the tanh saturation
plateaus (+/-1) in a correlated way, collapsing the effective
dimensionality -- weak input lets each neuron's own recurrent dynamics
dominate, keeping the state space closer to full-rank.
"""

import numpy as np
from common import DENSITY, N, SEEDS, T, WASHOUT
from plotstyle import PALETTE, new_figure, save

from esn import make_input, make_reservoir, run_esn
from metrics import effective_dimension

ALPHA = 1.0
RHOS = [0.5, 0.95]
# 13 log-spaced points from 0.01 to 10 (spacing of exactly 0.25 decades)
# chosen so that 0.1 and 10 -- the two values RESULTS.md reports -- land
# exactly on the grid instead of needing interpolation.
SCALE_GRID = np.logspace(-2, 1, 13)


def main():
    fig, ax = new_figure()

    results = {}
    for color, rho in zip(PALETTE, RHOS):
        means, stds = [], []
        for scale in SCALE_GRID:
            vals = []
            for seed in SEEDS:
                rng = np.random.default_rng(4000 + seed)
                u = rng.uniform(-1.0, 1.0, size=T)
                W = make_reservoir(N, DENSITY, rho, seed=seed)
                Win = make_input(N, scale=scale, seed=seed)
                states = run_esn(u, W, Win, alpha=ALPHA)
                d_eff = effective_dimension(states[WASHOUT:])
                vals.append(d_eff)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        results[rho] = (means, stds)

        ax.plot(SCALE_GRID, means, color=color, lw=1.8, label=rf"$\rho={rho}$")
        ax.fill_between(SCALE_GRID, means - stds, means + stds,
                         color=color, alpha=0.2, lw=0)

    ax.axhline(N, color="#8a8a8a", lw=0.8, ls="--", alpha=0.7)
    ax.text(SCALE_GRID[0], N, f"  N = {N}", va="bottom", ha="left",
            color="#8a8a8a", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input weight scale (Win amplitude)")
    ax.set_ylabel(r"effective dimension $D_{eff}$ (log scale)")
    ax.legend(frameon=False)

    save(fig, "fig_deff.png")

    for rho in RHOS:
        means, _ = results[rho]
        idx_01 = int(np.argmin(np.abs(SCALE_GRID - 0.1)))
        idx_10 = int(np.argmin(np.abs(SCALE_GRID - 10.0)))
        print(f"rho={rho}: D_eff at scale~0.1 (actual {SCALE_GRID[idx_01]:.3f}) "
              f"= {means[idx_01]:.2f}; at scale~10 (actual {SCALE_GRID[idx_10]:.3f}) "
              f"= {means[idx_10]:.2f}")


if __name__ == "__main__":
    main()
