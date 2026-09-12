"""Figure 5: NARMA10 target vs prediction on a 200-step test window.

Uses the best (rho, alpha) found by the fig_heatmap.py sweep (falls back to
a sensible default if the heatmap has not been run yet).
"""

import json
import pathlib

import numpy as np
from common import LAM, T, WASHOUT
from narma_pipeline import narma_trial
from plotstyle import BLUE, ORANGE, new_figure, save

N_STEPS_SHOWN = 200
SEED = 0

BEST_JSON = pathlib.Path(__file__).parent / "_heatmap_best.json"
DEFAULT_RHO, DEFAULT_ALPHA = 0.8, 0.9


def main():
    if BEST_JSON.exists():
        best = json.loads(BEST_JSON.read_text())
        rho, alpha = best["rho"], best["alpha"]
        print(f"using best (rho, alpha) from heatmap: {rho}, {alpha}")
    else:
        rho, alpha = DEFAULT_RHO, DEFAULT_ALPHA
        print(f"heatmap result not found, using default (rho, alpha): {rho}, {alpha}")

    result = narma_trial(rho, alpha, LAM, SEED, T=T, washout=WASHOUT)
    Y_test = result["Y_test"][:N_STEPS_SHOWN]
    Yhat_test = result["Yhat_test"][:N_STEPS_SHOWN]
    test_nmse_window = result["test_nmse"]

    fig, ax = new_figure()
    t = np.arange(N_STEPS_SHOWN)
    ax.plot(t, Y_test, color=BLUE, lw=1.6, label="target $y_t$")
    ax.plot(t, Yhat_test, color=ORANGE, lw=1.3, ls="--", label=r"prediction $\hat{y}_t$")
    ax.set_xlabel("test time step")
    ax.set_ylabel("NARMA10 output")
    ax.set_xlim(0, N_STEPS_SHOWN)
    ax.legend(frameon=False)
    ax.set_title(
        rf"$\rho={rho:.3f}$, $\alpha={alpha:.3f}$ -- test NMSE = {test_nmse_window:.4f}",
        fontsize=10,
    )

    save(fig, "fig_narma_pred.png")
    print(f"test NMSE (full test set): {test_nmse_window:.5f}")


if __name__ == "__main__":
    main()
