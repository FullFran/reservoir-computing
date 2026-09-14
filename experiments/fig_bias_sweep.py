"""Figure 10: bias sensitivity of the three headline results.

Background
----------
`esn.run_esn` takes a `bias` argument added inside the tanh nonlinearity,
and every experiment script in this project (`fig_rho_sweep.py`,
`fig_perturbation.py`, `fig_deff.py`, `fig_capacity_spectrum.py`, ...) has
used the implicit default `bias = 0.0`. At `bias = 0.0`, driven by
symmetric `u ~ Uniform[-1, 1]` from `x0 = 0`, the reservoir state is an
exactly odd functional of the input history (flipping `u -> -u` flips
every state `x_t -> -x_t` exactly, by induction through the tanh update),
which structurally zeroes all even-order information processing capacity
(see RESULTS.md, "Information processing capacity (IPC) spectrum").

This script re-measures the quantities behind this project's three
headline results at BIAS_GRID = [0.0, 0.1, 0.3, 0.6, 1.0]:

  (a) linear memory capacity (MC) vs rho, argmax rho              -- headline 1
  (b) NARMA10 test NMSE vs rho, argmin rho                        -- headline 1
  (c) perturbation-growth slope at rho = 0.5, 0.95, 1.3           -- headline 2
  (d) effective dimension D_eff at input scale 0.1 and 10.0       -- headline 3
  (e) capacity by polynomial order p = 1, 2, 3 at rho = 0.95      -- the mechanism

reusing the exact protocols of `fig_rho_sweep.py`, `fig_perturbation.py`,
`fig_deff.py` and `fig_capacity_spectrum.py` (same N, density, T, washout,
lam, alpha = 1.0, and the same 10 seeds, seeds 0-9) but threading `bias`
through every `run_esn` call.

None of `narma_pipeline.narma_trial`, `metrics.perturbation_growth`,
`metrics.memory_capacity`, or `capacity.capacity_spectrum` accept a `bias`
argument, and this script must not modify `src/`, `tests/`, or any
existing file in `experiments/`. So the trial functions below
re-implement the surrounding measurement code from `fig_rho_sweep.py`,
`fig_perturbation.py` and `fig_deff.py` locally, line-for-line identical
except for the added `bias` parameter passed into `run_esn`; wherever a
sibling fig_*.py module already exposes a constant or function that does
not itself need `bias` threaded through it (`RHOS`, `EPS`, `T_PERT`,
`fit_slope`, `NARMA_SEED_OFFSET`), it is imported here, not copied.

`memory_capacity`, `capacity_spectrum` and `effective_dimension` are used
unmodified: bias only ever enters a measurement via the state trajectory
`run_esn` produces, never via these functions' own math.

Correctness check: the bias = 0.0 column below MUST reproduce the numbers
already published in RESULTS.md (same seeds, same protocol, bias = 0.0
recovers the current default exactly). This was verified directly before
writing this docstring: with this script's trial functions, bias = 0.0
reproduces fig_perturbation.py's slopes (-0.3605, -0.1171, -0.0268 at rho
= 0.5, 0.95, 1.3) and fig_deff.py's D_eff (9.28 at scale 0.1, 1.35 at
scale 10.0, rho = 0.95) to the reported precision, and the MC/NARMA10
argmax/argmin land at rho = 1.27 (MC = 16.69) and rho = 0.84 (NMSE =
0.0614) on this script's coarser 12-point grid -- consistent with (not
identical to, since the published values come from a finer 20-point grid)
`fig_rho_sweep.py`'s rho ~= 1.20-1.21 (MC ~= 16.7) and rho ~= 0.78-0.81.

NARMA10 caveat: NARMA10 REQUIRES u ~ Uniform[0, 0.5] (see `tasks.py`),
an input distribution that is already NOT symmetric about zero. The
odd-symmetry argument for why bias = 0.0 kills even-order information
(sign flip u -> -u forces x_t -> -x_t, which cannot correlate with an
even-degree target) never applied to NARMA10 in the first place -- its
input distribution already breaks the symmetry on its own. So NARMA10's
optimum is expected to be far less bias-sensitive than the two
Uniform[-1, 1]-driven memory measurements (MC and D_eff). This is stated
again in this script's stdout output, not just here.

Runtime budget: under 15 minutes for the full 5-bias sweep, all 10 seeds.
Benchmarked directly on this machine before finalizing the grid: the
12-point rho grid used here (vs. fig_rho_sweep.py's 20-point grid) keeps
the MC+NARMA10 panels (the dominant cost, since they run the full T =
6000 ESN loop for every (rho, bias, seed) combination) to roughly 8
minutes total across all 5 biases, leaving comfortable headroom for
panels (c), (d) and (e). If runtime budget ever needs to shrink further,
shrink RHO_GRID below 12 points before ever reducing N_SEEDS below 10 --
this project's standard dispersion is always the seed count, never the
sample count.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from common import DENSITY, LAM, N, SEEDS, T, WASHOUT
from fig_perturbation import EPS, RHOS as PERT_RHOS, T_PERT, fit_slope
from narma_pipeline import NARMA_SEED_OFFSET
from plotstyle import GREY, PALETTE, save, style_axes
from tasks import sample_narma10

from capacity import capacity_spectrum
from esn import fit_ridge, make_input, make_reservoir, predict, run_esn
from metrics import effective_dimension, memory_capacity, nmse

ALPHA = 1.0
BIAS_GRID = [0.0, 0.1, 0.3, 0.6, 1.0]
RHO_GRID = np.linspace(0.4, 1.6, 12)
MC_MAX_DELAY = 40  # same as fig_rho_sweep.py
NARMA_TEST_FRAC = 0.2  # same as narma_pipeline.narma_trial's default

DEFF_RHO = 0.95
DEFF_SCALES = (0.1, 10.0)

CAP_RHO = 0.95
CAP_MAX_DEGREE = 3
CAP_MAX_DELAY = 6


def mc_trial(rho: float, bias: float, seed: int) -> float:
    """One (rho, bias, seed) linear memory capacity measurement.

    Identical protocol to fig_rho_sweep.py's MC panel (same input draw
    `rng = default_rng(2000 + seed)`, same reservoir construction, same
    `metrics.memory_capacity` call with MAX_DELAY = 40), with `bias`
    threaded through `run_esn`.
    """
    rng = np.random.default_rng(2000 + seed)
    u = rng.uniform(-1.0, 1.0, size=T)
    W = make_reservoir(N, DENSITY, rho, seed=seed)
    Win = make_input(N, scale=1.0, seed=seed)
    states = run_esn(u, W, Win, alpha=ALPHA, bias=bias)
    mc_total, _ = memory_capacity(states, u, max_delay=MC_MAX_DELAY, washout=WASHOUT, lam=LAM)
    return mc_total


def narma_trial_biased(rho: float, bias: float, seed: int):
    """One (rho, bias, seed) NARMA10 test-NMSE measurement.

    Reimplements `narma_pipeline.narma_trial`'s exact protocol (same
    NARMA_SEED_OFFSET, same reservoir/seed convention, same 80/20
    train/test split via test_frac = 0.2) since `narma_trial` itself does
    not expose a `bias` parameter. bias = 0.0 reproduces `narma_trial`'s
    numbers exactly.

    Returns (test_nmse, attempts).
    """
    u, y, attempts = sample_narma10(T, seed=NARMA_SEED_OFFSET + seed)
    W = make_reservoir(N, DENSITY, rho, seed=seed)
    Win = make_input(N, scale=1.0, seed=seed)
    states = run_esn(u, W, Win, alpha=ALPHA, bias=bias)

    X = states[WASHOUT:]
    Y = y[WASHOUT:]
    n = X.shape[0]
    split = int(n * (1.0 - NARMA_TEST_FRAC))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    Wout = fit_ridge(X_train, Y_train, LAM, washout=0)
    Yhat_test = predict(X_test, Wout)
    return nmse(Y_test, Yhat_test), attempts


def perturbation_slope(rho: float, bias: float) -> float:
    """Fitted contraction/expansion slope at one (rho, bias), averaged
    over all SEEDS, using the exact protocol of fig_perturbation.py
    (same input draw `rng = default_rng(3000 + seed)`, same EPS, same
    T_PERT, same direction-perturbation construction, same `fit_slope`
    fit window) with `bias` threaded through both trajectories' run_esn
    calls -- `metrics.perturbation_growth` itself has no bias parameter.
    """
    t = np.arange(T_PERT)
    curves = []
    for seed in SEEDS:
        rng = np.random.default_rng(3000 + seed)
        u = rng.uniform(-1.0, 1.0, size=T_PERT)
        W = make_reservoir(N, DENSITY, rho, seed=seed)
        Win = make_input(N, scale=1.0, seed=seed)

        n = W.shape[0]
        rng_dir = np.random.default_rng(seed)
        direction = rng_dir.standard_normal(n)
        direction = direction / np.linalg.norm(direction)
        x0_a = np.zeros(n)
        x0_b = EPS * direction

        states_a = run_esn(u, W, Win, alpha=ALPHA, bias=bias, x0=x0_a)
        states_b = run_esn(u, W, Win, alpha=ALPHA, bias=bias, x0=x0_b)
        dist = np.linalg.norm(states_a - states_b, axis=1)
        curves.append(np.log10(dist + 1e-300))

    logd_mean = np.array(curves).mean(axis=0)
    slope, _ = fit_slope(t, logd_mean)
    return float(slope)


def deff_trial(rho: float, scale: float, bias: float, seed: int) -> float:
    """One (rho, scale, bias, seed) D_eff measurement, exact protocol of
    fig_deff.py (same input draw `rng = default_rng(4000 + seed)`) with
    `bias` threaded through run_esn.
    """
    rng = np.random.default_rng(4000 + seed)
    u = rng.uniform(-1.0, 1.0, size=T)
    W = make_reservoir(N, DENSITY, rho, seed=seed)
    Win = make_input(N, scale=scale, seed=seed)
    states = run_esn(u, W, Win, alpha=ALPHA, bias=bias)
    return effective_dimension(states[WASHOUT:])


def capacity_by_order_trial(rho: float, bias: float, seed: int) -> np.ndarray:
    """capacity_spectrum grid for one (rho, bias, seed), exact protocol of
    fig_capacity_spectrum.py (same input draw `rng = default_rng(5000 +
    seed)`) with `bias` threaded through run_esn. Returns the per-degree
    totals for p = 1, 2, ..., CAP_MAX_DEGREE.
    """
    rng = np.random.default_rng(5000 + seed)
    u = rng.uniform(-1.0, 1.0, size=T)
    W = make_reservoir(N, DENSITY, rho, seed=seed)
    Win = make_input(N, scale=1.0, seed=seed)
    states = run_esn(u, W, Win, alpha=ALPHA, bias=bias)
    _, _, grid = capacity_spectrum(
        states, u, max_degree=CAP_MAX_DEGREE, max_delay=CAP_MAX_DELAY,
        washout=WASHOUT, lam=LAM,
    )
    return grid.sum(axis=1)[1:]  # drop the always-zero p=0 row


def measure_bias(bias: float) -> dict:
    """Measure all five quantities at one bias value."""
    mc_mean = np.zeros(len(RHO_GRID))
    mc_std = np.zeros(len(RHO_GRID))
    narma_mean = np.zeros(len(RHO_GRID))
    narma_std = np.zeros(len(RHO_GRID))
    resamples = 0

    for i, rho in enumerate(RHO_GRID):
        mc_vals, narma_vals = [], []
        for seed in SEEDS:
            mc_vals.append(mc_trial(rho, bias, seed))
            nmse_val, attempts = narma_trial_biased(rho, bias, seed)
            narma_vals.append(nmse_val)
            if attempts > 1:
                resamples += 1
        mc_mean[i] = np.mean(mc_vals)
        mc_std[i] = np.std(mc_vals)
        narma_mean[i] = np.mean(narma_vals)
        narma_std[i] = np.std(narma_vals)

    pert_slopes = {rho: perturbation_slope(rho, bias) for rho in PERT_RHOS}

    deff = {}
    for scale in DEFF_SCALES:
        vals = [deff_trial(DEFF_RHO, scale, bias, seed) for seed in SEEDS]
        deff[scale] = (float(np.mean(vals)), float(np.std(vals)))

    cap_vals = np.array([capacity_by_order_trial(CAP_RHO, bias, seed) for seed in SEEDS])
    cap_mean = cap_vals.mean(axis=0)
    cap_std = cap_vals.std(axis=0)

    return {
        "mc_mean": mc_mean,
        "mc_std": mc_std,
        "narma_mean": narma_mean,
        "narma_std": narma_std,
        "resamples": resamples,
        "pert_slopes": pert_slopes,
        "deff": deff,
        "cap_mean": cap_mean,
        "cap_std": cap_std,
    }


def make_figure(results: dict):
    bias_arr = np.array(BIAS_GRID)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_mc, ax_narma, ax_pert, ax_deff = axes.flat
    for ax in axes.flat:
        style_axes(ax)

    cmap = plt.get_cmap("viridis")
    bias_colors = [cmap(x) for x in np.linspace(0.15, 0.9, len(BIAS_GRID))]

    for color, bias in zip(bias_colors, BIAS_GRID):
        r = results[bias]
        ax_mc.plot(RHO_GRID, r["mc_mean"], color=color, lw=1.6, label=f"bias={bias}")
        ax_mc.fill_between(RHO_GRID, r["mc_mean"] - r["mc_std"], r["mc_mean"] + r["mc_std"],
                            color=color, alpha=0.15, lw=0)

        ax_narma.plot(RHO_GRID, r["narma_mean"], color=color, lw=1.6, label=f"bias={bias}")
        ax_narma.fill_between(RHO_GRID, r["narma_mean"] - r["narma_std"], r["narma_mean"] + r["narma_std"],
                               color=color, alpha=0.15, lw=0)

    ax_mc.axvline(1.0, color=GREY, lw=0.7, ls="--", alpha=0.6)
    ax_mc.set_xlabel(r"spectral radius $\rho$")
    ax_mc.set_ylabel("linear memory capacity $MC$")
    ax_mc.legend(frameon=False, fontsize=8)

    ax_narma.axvline(1.0, color=GREY, lw=0.7, ls="--", alpha=0.6)
    ax_narma.set_xlabel(r"spectral radius $\rho$")
    ax_narma.set_ylabel("NARMA10 test NMSE")
    ax_narma.set_yscale("log")
    ax_narma.legend(frameon=False, fontsize=8)

    for color, rho in zip(PALETTE, PERT_RHOS):
        slopes = [results[bias]["pert_slopes"][rho] for bias in BIAS_GRID]
        ax_pert.plot(bias_arr, slopes, color=color, marker="o", lw=1.6, label=rf"$\rho={rho}$")
    ax_pert.axhline(0.0, color=GREY, lw=0.8, ls="--", alpha=0.7)
    ax_pert.set_xlabel("bias")
    ax_pert.set_ylabel("perturbation slope (log10-units/step)")
    ax_pert.legend(frameon=False, fontsize=8)

    for color, scale in zip(PALETTE, DEFF_SCALES):
        means = np.array([results[bias]["deff"][scale][0] for bias in BIAS_GRID])
        stds = np.array([results[bias]["deff"][scale][1] for bias in BIAS_GRID])
        ax_deff.plot(bias_arr, means, color=color, marker="o", lw=1.6, label=f"scale={scale}")
        ax_deff.fill_between(bias_arr, means - stds, means + stds, color=color, alpha=0.15, lw=0)
    ax_deff.set_xlabel("bias")
    ax_deff.set_ylabel(rf"$D_{{eff}}$ ($\rho={DEFF_RHO}$)")
    ax_deff.set_yscale("log")
    ax_deff.legend(frameon=False, fontsize=8)

    fig.suptitle("Bias sensitivity of the headline results", color=GREY)
    fig.tight_layout()
    save(fig, "fig_bias_sweep.png")


def main():
    t_start = time.time()
    print(
        "Bias sensitivity sweep: N=%d, density=%.2f, T=%d, washout=%d, lam=%.0e, "
        "alpha=%.1f, %d seeds (seeds %d-%d)"
        % (N, DENSITY, T, WASHOUT, LAM, ALPHA, len(SEEDS), SEEDS[0], SEEDS[-1])
    )
    print(f"bias grid: {BIAS_GRID}")
    print(f"rho grid ({len(RHO_GRID)} points, MC/NARMA panels): "
          + ", ".join(f"{r:.3f}" for r in RHO_GRID))
    print(
        "NOTE: NARMA10 requires u ~ Uniform[0, 0.5], an input distribution that is "
        "already asymmetric about zero -- the odd-symmetry argument for why bias = "
        "0.0 kills even-order information never applied to NARMA10 in the first "
        "place, so its bias-sensitivity is expected to be much weaker than MC, "
        "D_eff and the capacity spectrum, which are all driven by symmetric "
        "u ~ Uniform[-1, 1]."
    )

    results = {}
    for bias in BIAS_GRID:
        t0 = time.time()
        r = measure_bias(bias)
        results[bias] = r

        mc_idx = int(np.argmax(r["mc_mean"]))
        narma_idx = int(np.argmin(r["narma_mean"]))
        gap = RHO_GRID[mc_idx] - RHO_GRID[narma_idx]

        print(f"\n--- bias = {bias:.2f} ({time.time() - t0:.1f}s) ---")
        print(f"  MC:      argmax rho = {RHO_GRID[mc_idx]:.3f}  "
              f"(MC = {r['mc_mean'][mc_idx]:.2f} +/- {r['mc_std'][mc_idx]:.2f})")
        print(f"  NARMA10: argmin rho = {RHO_GRID[narma_idx]:.3f}  "
              f"(test NMSE = {r['narma_mean'][narma_idx]:.4f} +/- {r['narma_std'][narma_idx]:.4f})")
        print(f"  rho(MC argmax) - rho(NARMA argmin) = {gap:+.3f}")
        for rho in PERT_RHOS:
            slope = r["pert_slopes"][rho]
            regime = "expanding" if slope > 0 else "contracting"
            print(f"  perturbation rho={rho}: slope={slope:+.5f} log10-units/step ({regime})")
        d01_mean, d01_std = r["deff"][0.1]
        d10_mean, d10_std = r["deff"][10.0]
        print(f"  D_eff (rho={DEFF_RHO}): scale=0.1 -> {d01_mean:.2f}+/-{d01_std:.2f}; "
              f"scale=10.0 -> {d10_mean:.2f}+/-{d10_std:.2f}")
        cap_str = ", ".join(
            f"p={p}: {m:.2f}+/-{s:.2f}"
            for p, m, s in zip(range(1, CAP_MAX_DEGREE + 1), r["cap_mean"], r["cap_std"])
        )
        print(f"  capacity by order (rho={CAP_RHO}, max_delay={CAP_MAX_DELAY}): {cap_str}")
        print(f"  NARMA10 divergence resamples: {r['resamples']}")

    # ------------------------------------------------------------------
    # Summary tables, one per headline result
    # ------------------------------------------------------------------
    print("\n=== Summary: headline result 1 (MC argmax vs NARMA argmin rho, opposite sides of 1) ===")
    print(f"{'bias':>6} | {'MC argmax rho':>14} | {'NARMA argmin rho':>17} | {'gap':>7}")
    for bias in BIAS_GRID:
        r = results[bias]
        mc_idx = int(np.argmax(r["mc_mean"]))
        narma_idx = int(np.argmin(r["narma_mean"]))
        gap = RHO_GRID[mc_idx] - RHO_GRID[narma_idx]
        print(f"{bias:>6.2f} | {RHO_GRID[mc_idx]:>14.3f} | {RHO_GRID[narma_idx]:>17.3f} | {gap:>+7.3f}")

    print("\n=== Summary: headline result 2 (perturbation contraction, rho=0.5/0.95/1.3) ===")
    print(f"{'bias':>6} | " + " | ".join(f"rho={rho:>4}" for rho in PERT_RHOS))
    any_flip = False
    for bias in BIAS_GRID:
        r = results[bias]
        cells = []
        for rho in PERT_RHOS:
            slope = r["pert_slopes"][rho]
            if slope > 0:
                any_flip = True
            cells.append(f"{slope:>+8.5f}")
        print(f"{bias:>6.2f} | " + " | ".join(cells))
    print(f"contraction verdict flips at any tested bias: {any_flip}")

    print("\n=== Summary: headline result 3 (D_eff collapse at rho=0.95, scale 0.1 -> 10.0) ===")
    print(f"{'bias':>6} | {'D_eff@0.1':>10} | {'D_eff@10.0':>11} | {'ratio':>8}")
    for bias in BIAS_GRID:
        r = results[bias]
        d01_mean, _ = r["deff"][0.1]
        d10_mean, _ = r["deff"][10.0]
        ratio = d01_mean / d10_mean
        print(f"{bias:>6.2f} | {d01_mean:>10.2f} | {d10_mean:>11.2f} | {ratio:>8.2f}")

    print("\n=== Summary: even-order capacity switching on (rho=0.95) ===")
    print(f"{'bias':>6} | {'p=1':>10} | {'p=2':>10} | {'p=3':>10}")
    for bias in BIAS_GRID:
        r = results[bias]
        print(f"{bias:>6.2f} | {r['cap_mean'][0]:>10.2f} | {r['cap_mean'][1]:>10.2f} | {r['cap_mean'][2]:>10.2f}")

    make_figure(results)

    print(f"\nTotal wall-clock: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
