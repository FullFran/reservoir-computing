# Results

All numbers below come from actually running the scripts in `experiments/`
against the from-scratch NumPy implementation in `src/`, with
N = 300, density = 0.1, T = 6000, washout = 500, lam = 1e-6 (except where a
figure explicitly sweeps one of these), averaged over 10 seeds (seeds
0-9) wherever a curve or scalar is reported. Leak rate alpha = 1.0 unless
a figure sweeps alpha explicitly (fig_heatmap).

## Headline numbers

**Linear memory capacity (MC), delay k = 0..40, 10-seed mean** (`fig_memory_curve.py`):

| rho | MC | first k with R^2 < 0.5 |
|------|-------|------|
| 0.50 | 11.31 | k = 12 |
| 0.95 | 15.10 | k = 16 |
| 1.30 | 16.51 | k = 17 |

**NARMA10, best test NMSE over the 12x12 (rho, alpha) grid, 10-seed mean** (`fig_heatmap.py`):

- Best cell: **rho = 0.782, alpha = 0.918 -> test NMSE = 0.0527**
- With alpha fixed at 1.0 (`fig_rho_sweep.py`), the best rho alone is **rho = 0.811 -> test NMSE = 0.0605** -- allowing the leak rate to adapt only shaves off another ~13% of the error, it does not move the optimum much.
- No NARMA10 divergence was observed in any of the 10 seeds x 20 rho values tested (`fig_rho_sweep.py` reports "NARMA10 resamples needed (divergence): 0"). The retry-on-divergence path in `tasks.sample_narma10` exists and is exercised in the test suite's random inputs, but never triggered in these particular experiments' Uniform[0, 0.5] draws.

**Perturbation growth, fitted slope of log10||x_t - x'_t|| vs t** (`fig_perturbation.py`, eps=1e-9, 10-seed mean curve, fit over the pre-floor linear region):

| rho | slope (log10-units / step) | regime |
|------|------|------|
| 0.50 | -0.3605 | contracting |
| 0.95 | -0.1171 | contracting |
| 1.30 | -0.0268 | contracting (barely) |

**Effective dimension D_eff, rho = 0.95** (`fig_deff.py`, N = 300, 10-seed mean):

| input (Win) scale | D_eff |
|------|------|
| 0.1  | 9.28 |
| 10.0 | 1.35 |

For comparison, rho = 0.5 stays essentially flat and low across the whole scale range (D_eff = 1.59 at scale 0.1, D_eff = 1.12 at scale 10) -- there is much less complexity to collapse in the first place.

## Does "rho just under 1 is optimal" hold?

**No, and the two tasks disagree with each other, not just with the folklore.**

- Memory capacity is *maximized above* rho = 1: the 20-point sweep in
  `fig_rho_sweep.png` peaks at **rho ~= 1.20-1.21 (MC ~= 16.7)**, and MC is
  still higher at rho = 1.3 (16.51, from the memory-curve run) than at
  rho = 0.95 (15.10). This is not a fluke of one run -- the two independent
  scripts (`fig_memory_curve.py` and `fig_rho_sweep.py`) agree on this within
  ~1%.
- NARMA10 is *minimized below* rho = 1: the same sweep's NMSE panel bottoms
  out at **rho ~= 0.78-0.81**, and error rises on both sides of that point,
  including monotonically for every rho > 1 tested up to 1.6.
- So the two "optimal rho" values are on **opposite sides of 1** and about
  0.4 apart -- there is no single rho that is simultaneously best for
  linear memory and for a nonlinear task like NARMA10. This is the
  classic memory/nonlinearity tradeoff, but it is sharper than "just under
  1 is a good compromise": 1 itself is neither optimum, it sits on the
  NARMA10 side of the memory-capacity peak and past the NARMA10 minimum.

## Other things that surprised us

1. **rho = 1.3 does not blow up here -- it is still contracting.**
   `fig_perturbation.png` shows all three tested rho values (0.5, 0.95, 1.3)
   as net-contracting under identical-input driving with Win scale 1.0: the
   fitted slope is negative for all three, just progressively closer to
   zero. Naively, spectral radius > 1 should mean chaos/expansion, but the
   *driven* reservoir's conditional Lyapunov exponent is not the same thing
   as the spectral radius of W alone -- a sufficiently informative input
   signal synchronizes (or "entrains") nearby trajectories even when the
   autonomous map would be unstable. This is consistent with the classic
   Jaeger result that rho(W) < 1 is neither necessary nor sufficient for
   the echo state property once you're actually driving the reservoir. It
   also explains, mechanistically, why MC keeps rising past rho = 1 in this
   setup: the network isn't chaotic there yet, so linear reconstructability
   is not being destroyed by sensitivity to initial conditions.
2. **Effective dimension never gets close to N, even in the best case.**
   With density = 0.1 and N = 300, the best D_eff we ever measured
   (rho = 0.95, Win scale ~ 0.01-0.1) was only about 10 -- roughly 3% of N.
   A sparse random 300-unit reservoir is nowhere near a 300-dimensional
   representation even before any input-driven collapse; the "collapse"
   documented above is a collapse from ~10 effective dimensions down to
   ~1.1-1.35, not from 300.
3. **The ridge overfitting effect is real but mild here, because T >> N.**
   `fig_ridge.png` shows train and test NMSE staying close together (a gap
   of roughly 0.005) across nine full decades of lambda (1e-12 to 1e-6)
   before both rise together as lambda keeps growing past that. With
   ~4,400 post-washout training samples and only 301 readout parameters
   (300 units + bias), ordinary least squares is nowhere near the
   interpolation threshold, so the classic sharp overfitting blowup at
   lam -> 0 that shows up in small-N/small-T regimes is only a small,
   fairly flat gap here rather than a dramatic spike.
4. **The leaky-integrator leak rate alpha barely moves the NARMA10
   optimum.** The best cell in the 12x12 heatmap (rho=0.782, alpha=0.918)
   is close to alpha = 1 (i.e., close to no leaky integration at all), and
   the rho-only sweep at alpha = 1 (NMSE = 0.0605) is barely worse than the
   full 2D optimum (NMSE = 0.0527). The heatmap does show a much sharper
   penalty for *low* alpha (< 0.3) than for rho being off by a similar
   relative amount, i.e. this reservoir is far more sensitive to
   under-leaking (alpha too small) than to spectral radius alone.

## Figures

All saved at `dpi=160`, `bbox_inches='tight'`, `transparent=True`, with all
text/spines/grid in `#8a8a8a` and data colors from the fixed
blue/orange/green/purple palette, so they read correctly on both light and
dark backgrounds.

1. `/home/franblakia/fullfran/reservoir-computing/figures/fig_states.png`
2. `/home/franblakia/fullfran/reservoir-computing/figures/fig_memory_curve.png`
3. `/home/franblakia/fullfran/reservoir-computing/figures/fig_rho_sweep.png`
4. `/home/franblakia/fullfran/reservoir-computing/figures/fig_perturbation.png`
5. `/home/franblakia/fullfran/reservoir-computing/figures/fig_narma_pred.png`
6. `/home/franblakia/fullfran/reservoir-computing/figures/fig_deff.png`
7. `/home/franblakia/fullfran/reservoir-computing/figures/fig_ridge.png`
8. `/home/franblakia/fullfran/reservoir-computing/figures/fig_heatmap.png`

## Implementation notes relevant to reproducing these numbers

- `esn.run_esn` stores `states[t]` as the state produced *after* consuming
  `u_t` (not before). This matters for memory-capacity-style tasks: delay
  k = 0 asks "can a linear readout of x_t recover u_t itself", which is
  the easy/near-1 end of the forgetting curve, and delay grows from there.
  The first version of this code stored the pre-update state instead,
  which shifted the whole forgetting curve by one step and made it
  non-monotonic near k = 0; this was caught by manually inspecting
  `fig_memory_curve.png` before finalizing it (see git history / session
  notes) and fixed before any figure was finalized.
- `esn.fit_ridge` takes `washout` as a required (no-default) argument, and
  always prepends the bias column internally, per the task's explicit
  "never silent" requirement.
- `metrics.memory_capacity` fits a *separate* ridge readout per delay k on
  a training slice and scores R^2 out-of-sample on a held-out slice (50/50
  split of the post-washout data) -- MC is never fit and evaluated on the
  same data.
