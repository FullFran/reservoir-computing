"""Figure 4: perturbation growth / contraction (a Lyapunov-exponent proxy).

Two identical-input trajectories starting eps apart are run for each rho;
the log10 of their distance over time reveals whether nearby trajectories
converge (contraction, echo state property holds) or diverge (expansion,
chaotic / not echo-state).  Averaged over >= 10 seeds (shaded +/- 1 std).
"""

import numpy as np
from common import DENSITY, N, SEEDS
from plotstyle import PALETTE, new_figure, save

from esn import make_input, make_reservoir
from metrics import perturbation_growth

ALPHA = 1.0
EPS = 1e-9
T_PERT = 300
RHOS = [0.5, 0.95, 1.3]
FLOOR = -13.0  # below this, float64 cancellation noise dominates -- exclude from slope fit
FIT_START = 2  # skip the very first couple of steps (initial-condition noise)


def fit_slope(t, logd_mean):
    mask = (t >= FIT_START) & (logd_mean > FLOOR)
    if mask.sum() < 5:
        mask = t >= FIT_START
    slope, intercept = np.polyfit(t[mask], logd_mean[mask], 1)
    return slope, mask


def main():
    fig, ax = new_figure()
    t = np.arange(T_PERT)
    slopes = {}

    for color, rho in zip(PALETTE, RHOS):
        curves = []
        for seed in SEEDS:
            rng = np.random.default_rng(3000 + seed)
            u = rng.uniform(-1.0, 1.0, size=T_PERT)
            W = make_reservoir(N, DENSITY, rho, seed=seed)
            Win = make_input(N, scale=1.0, seed=seed)
            dist = perturbation_growth(u, W, Win, alpha=ALPHA, eps=EPS, seed=seed)
            curves.append(np.log10(dist + 1e-300))
        curves = np.array(curves)
        logd_mean = curves.mean(axis=0)
        logd_std = curves.std(axis=0)

        slope, mask = fit_slope(t, logd_mean)
        slopes[rho] = slope

        ax.plot(t, logd_mean, color=color, lw=1.6,
                label=f"$\\rho={rho}$ (slope={slope:+.4f}/step)")
        ax.fill_between(t, logd_mean - logd_std, logd_mean + logd_std,
                         color=color, alpha=0.2, lw=0)
        fit_line = slope * t[mask] + np.polyfit(t[mask], logd_mean[mask], 1)[1]
        ax.plot(t[mask], fit_line, color=color, lw=1.0, ls="--", alpha=0.9)

    ax.axhline(np.log10(EPS), color="#8a8a8a", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"$\log_{10} \|x_t - x_t'\|$")
    ax.set_xlim(0, T_PERT)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    save(fig, "fig_perturbation.png")

    for rho in RHOS:
        regime = "expanding" if slopes[rho] > 0 else "contracting"
        print(f"rho={rho}: fitted slope={slopes[rho]:+.5f} log10-units/step ({regime})")


if __name__ == "__main__":
    main()
