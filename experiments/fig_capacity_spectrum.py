"""Figure 9: information processing capacity (IPC) spectrum C(k, p).

Generalises the linear-only forgetting curve of `fig_memory_curve.py` to
Dambre et al. (2012)'s full spectrum: capacity broken down jointly by
polynomial order p (how nonlinear a term is) and temporal depth k (how far
back it reaches), for the same three spectral radii used throughout this
repo (rho = 0.5, 0.95, 1.3), averaged over 10 seeds.

Parameter choice (MAX_DEGREE, MAX_DELAY) and why:
`capacity.enumerate_specs` is combinatorial in these two parameters (see
its docstring), and each spec costs one ridge fit + out-of-sample score.
With N = 300, T = 6000 (this project's shared defaults), a single
`capacity_of` call costs a few milliseconds, so the runtime budget of
"under ~3 minutes total" for 3 rho values x 10 seeds effectively caps the
usable spec count at roughly a couple thousand.

MAX_DEGREE = 4, MAX_DELAY = 8 gives 714 specs (measured directly via
`enumerate_specs(4, 8)`), i.e. 714 x 10 x 3 = 21,420 capacity
evaluations. Benchmarked on this machine this runs in under two minutes
end to end (including building all 30 reservoirs), leaving headroom
under the 3-minute target. A tighter delay window at higher degree would
also work (Dambre et al. show nonlinear terms decay fast in k in
practice), but a uniform box over (p, k) keeps `enumerate_specs`'s single
(max_degree, max_delay) contract simple and the resulting heatmap
rectangular and easy to read.

Two panels are produced, saved together as one figure:
- left: total capacity per polynomial order p (summed over all k),
  grouped by rho, with +/- 1 std error bars over seeds -- the headline
  answer to "how much of the capacity budget is linear vs nonlinear".
- right: the full C(k, p) grid as a heatmap, mean over seeds, for one
  representative rho (0.95).
"""

import matplotlib.pyplot as plt
import numpy as np
from common import DENSITY, LAM, N, SEEDS, T, WASHOUT
from plotstyle import GREY, PALETTE, FIGURES_DIR

from capacity import capacity_spectrum
from esn import make_input, make_reservoir, run_esn

ALPHA = 1.0
RHOS = [0.5, 0.95, 1.3]
MAX_DEGREE = 4
MAX_DELAY = 8
REPRESENTATIVE_RHO = 0.95


def spectrum_for_rho(rho):
    """Return (grids, totals) over SEEDS.

    grids : (len(SEEDS), MAX_DEGREE + 1, MAX_DELAY + 1) ndarray, one
        capacity_spectrum grid per seed.
    totals : (len(SEEDS),) ndarray, one total capacity per seed.
    """
    grids, totals = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(5000 + seed)
        u = rng.uniform(-1.0, 1.0, size=T)
        W = make_reservoir(N, DENSITY, rho, seed=seed)
        Win = make_input(N, scale=1.0, seed=seed)
        states = run_esn(u, W, Win, alpha=ALPHA)

        total, _, grid = capacity_spectrum(
            states, u, max_degree=MAX_DEGREE, max_delay=MAX_DELAY,
            washout=WASHOUT, lam=LAM,
        )
        grids.append(grid)
        totals.append(total)
    return np.array(grids), np.array(totals)


def main():
    n_readout_params = N + 1  # N reservoir units + 1 bias (Dambre bound)
    results = {}
    for rho in RHOS:
        grids, totals = spectrum_for_rho(rho)
        results[rho] = (grids, totals)
        print(
            f"rho={rho}: total capacity = {totals.mean():.2f} +/- "
            f"{totals.std():.2f} (Dambre bound: <= {n_readout_params})"
        )
        by_degree_mean = grids.sum(axis=2)[:, 1:].mean(axis=0)  # (MAX_DEGREE,)
        by_degree_std = grids.sum(axis=2)[:, 1:].std(axis=0)
        for p in range(1, MAX_DEGREE + 1):
            print(
                f"  p={p}: capacity = {by_degree_mean[p - 1]:.3f} "
                f"+/- {by_degree_std[p - 1]:.3f}"
            )

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- left panel: capacity by polynomial order p, grouped by rho ---
    degrees = np.arange(1, MAX_DEGREE + 1)
    bar_width = 0.8 / len(RHOS)
    for i, (color, rho) in enumerate(zip(PALETTE, RHOS)):
        grids, _ = results[rho]
        by_degree = grids.sum(axis=2)[:, 1:]  # (n_seeds, MAX_DEGREE), drop p=0
        means = by_degree.mean(axis=0)
        stds = by_degree.std(axis=0)
        offset = (i - (len(RHOS) - 1) / 2) * bar_width
        ax_bar.bar(
            degrees + offset, means, width=bar_width, color=color,
            yerr=stds, capsize=3, label=rf"$\rho={rho}$",
        )
    ax_bar.set_xlabel("polynomial order $p$")
    ax_bar.set_ylabel("capacity (summed over delay $k$)")
    ax_bar.set_xticks(degrees)
    ax_bar.legend(frameon=False)

    # --- right panel: C(k, p) heatmap for the representative rho ---
    grids, _ = results[REPRESENTATIVE_RHO]
    mean_grid = grids.mean(axis=0)[1:, :]  # drop p=0 row (always zero)

    im = ax_heat.imshow(
        mean_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[-0.5, MAX_DELAY + 0.5, 0.5, MAX_DEGREE + 0.5],
    )
    ax_heat.set_xlabel("temporal depth $k$ (largest delay in the term)")
    ax_heat.set_ylabel("polynomial order $p$")
    ax_heat.set_yticks(degrees)
    ax_heat.set_title(rf"$\rho = {REPRESENTATIVE_RHO}$", color=GREY, fontsize=10)
    ax_heat.tick_params(colors=GREY)
    for spine in ax_heat.spines.values():
        spine.set_color(GREY)

    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label(f"$C(k, p)$, mean over {len(SEEDS)} seeds", color=GREY)
    cbar.ax.yaxis.set_tick_params(color=GREY)
    plt.setp(cbar.ax.get_yticklabels(), color=GREY)
    cbar.outline.set_edgecolor(GREY)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_capacity_spectrum.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
