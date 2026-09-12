"""Figure 2 (the money figure): forgetting curve.

R^2 of linearly reconstructing u_{t-k} from the reservoir state, as a
function of delay k, for three spectral radii. Averaged over >= 10 seeds
with shaded +/- 1 std.
"""

import numpy as np
from common import DENSITY, LAM, N, SEEDS, T, WASHOUT
from plotstyle import PALETTE, new_figure, save

from esn import make_input, make_reservoir, run_esn
from metrics import memory_capacity

ALPHA = 1.0
MAX_DELAY = 40
RHOS = [0.5, 0.95, 1.3]


def curves_for_rho(rho):
    """Return (mean_r2, std_r2) arrays of shape (MAX_DELAY + 1,) over SEEDS."""
    all_curves = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        u = rng.uniform(-1.0, 1.0, size=T)

        W = make_reservoir(N, DENSITY, rho, seed=seed)
        Win = make_input(N, scale=1.0, seed=seed)
        states = run_esn(u, W, Win, alpha=ALPHA)

        _, per_delay_r2 = memory_capacity(
            states, u, max_delay=MAX_DELAY, washout=WASHOUT, lam=LAM
        )
        all_curves.append(per_delay_r2)
    all_curves = np.array(all_curves)  # (n_seeds, MAX_DELAY+1)
    return all_curves.mean(axis=0), all_curves.std(axis=0), all_curves


def main():
    fig, ax = new_figure()
    k = np.arange(MAX_DELAY + 1)

    results = {}
    for color, rho in zip(PALETTE, RHOS):
        mean_r2, std_r2, _ = curves_for_rho(rho)
        results[rho] = (mean_r2, std_r2)
        ax.plot(k, mean_r2, color=color, lw=1.8, label=f"$\\rho = {rho}$")
        ax.fill_between(
            k, mean_r2 - std_r2, mean_r2 + std_r2, color=color, alpha=0.2, lw=0
        )

    ax.axhline(0.0, color="#8a8a8a", lw=0.7, alpha=0.5)
    ax.axhline(0.5, color="#8a8a8a", lw=0.7, alpha=0.5, ls="--")
    ax.set_xlabel("delay $k$")
    ax.set_ylabel("test $R^2$ reconstructing $u_{t-k}$")
    ax.set_xlim(0, MAX_DELAY)
    ax.legend(frameon=False)

    save(fig, "fig_memory_curve.png")

    # print numbers used in RESULTS.md
    for rho in RHOS:
        mean_r2, _ = results[rho]
        mc_total = float(np.sum(np.clip(mean_r2, 0.0, None)))
        below_half = np.where(mean_r2 < 0.5)[0]
        k_half = int(below_half[0]) if len(below_half) else None
        print(f"rho={rho}: MC(from mean curve)={mc_total:.3f}, "
              f"first k with R2<0.5: {k_half}")


if __name__ == "__main__":
    main()
