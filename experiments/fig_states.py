"""Figure 1: a handful of reservoir neuron traces settling out of the
washout transient into an input-driven regime.
"""

import numpy as np
from common import DENSITY, N
from plotstyle import GREY, PALETTE, new_figure, save

from esn import make_input, make_reservoir, run_esn

SEED = 0
RHO = 0.9
ALPHA = 1.0
N_STEPS_SHOWN = 300
N_TRACES = 8

# NOTE: this is an illustrative transient window, chosen small enough to be
# visible by eye in a 300-step plot. It is NOT the conservative washout=500
# used for fitting elsewhere in this project (that longer washout is a
# safety margin, not the actual settling time of this particular rho=0.9
# reservoir, which settles out of its initial-condition transient in well
# under 50 steps).
WASHOUT_ILLUSTRATION = 40


def main():
    rng = np.random.default_rng(SEED)
    u = rng.uniform(-1.0, 1.0, size=N_STEPS_SHOWN)

    W = make_reservoir(N, DENSITY, RHO, seed=SEED)
    Win = make_input(N, scale=1.0, seed=SEED)
    states = run_esn(u, W, Win, alpha=ALPHA)

    neuron_idx = np.linspace(0, N - 1, N_TRACES, dtype=int)
    t = np.arange(N_STEPS_SHOWN)

    fig, ax = new_figure()
    ax.axvspan(0, WASHOUT_ILLUSTRATION, color=GREY, alpha=0.15, lw=0)
    ax.text(WASHOUT_ILLUSTRATION / 2, 0.98, "washout", ha="center",
            va="top", color=GREY, fontsize=9, transform=ax.get_xaxis_transform())

    colors = (PALETTE * ((N_TRACES // len(PALETTE)) + 1))[:N_TRACES]
    for i, idx in enumerate(neuron_idx):
        ax.plot(t, states[:, idx], color=colors[i], lw=0.9, alpha=0.85)

    ax.set_xlabel("time step $t$")
    ax.set_ylabel("reservoir state $x_i(t)$")
    ax.set_xlim(0, N_STEPS_SHOWN)

    save(fig, "fig_states.png")


if __name__ == "__main__":
    main()
