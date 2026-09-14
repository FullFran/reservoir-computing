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

## Information processing capacity (IPC) spectrum

`src/capacity.py` generalises linear memory capacity to the full Dambre et
al. (2012) spectrum C(k, p): capacity broken down jointly by polynomial
order p (nonlinearity) and temporal depth k (how far back a term reaches),
using the normalised Legendre family Ptilde_d as the orthonormal basis for
u ~ Uniform[-1, 1]. Measured with `experiments/fig_capacity_spectrum.py`,
N = 300, T = 6000, washout = 500, lam = 1e-6, alpha = 1.0, 10 seeds, for the
same three rho values used throughout this project.

**Parameter choice.** `enumerate_specs` is combinatorial in (max_degree,
max_delay); `enumerate_specs(4, 8)` already yields 714 specs (9 at p=1, 45
at p=2, 165 at p=3, 495 at p=4). At ~3-4 ms per capacity evaluation, 714
specs x 10 seeds x 3 rho (21,420 evaluations) ran in 1 min 38 s wall clock
(measured), comfortably inside the ~3-minute budget.

**Headline numbers (total capacity, mean +/- std over 10 seeds; conservative
readout-parameter reference N + 1 = 301):**

| rho | total capacity | p=1 | p=2 | p=3 | p=4 |
|------|------|------|------|------|------|
| 0.50 | 101.46 +/- 1.62 | 8.953 +/- 0.009 | 0.000 +/- 0.000 | 92.509 +/- 1.615 | 0.000 +/- 0.000 |
| 0.95 | 102.41 +/- 1.65 | 8.845 +/- 0.018 | 0.000 +/- 0.000 | 93.568 +/- 1.638 | 0.000 +/- 0.000 |
| 1.30 | 47.56 +/- 2.80 | 8.646 +/- 0.018 | 0.000 +/- 0.000 | 38.917 +/- 2.806 | 0.000 +/- 0.000 |

The measured totals (~48-102) are well under the conservative readout-parameter
reference of 301 for all three rho. Dambre's bound applies to population
projections onto an orthonormal target family; a constant intercept adds no
population capacity for these centred targets. Finite-sample ridge fits and
clipped held-out R^2 sums need not obey the bound exactly. These totals are
sanity checks within a finite (p <= 4, k <= 8) window, not proofs of the bound.

**All measured even-order estimates are exactly zero after clipping,
consistent with population symmetry.** Instrument control: a
synthetic states array containing `u_t^2` as an explicit column gives
`capacity_of` = 1.0 for the even target `Ptilde_2(u_t)`, so the instrument
correctly detects even-order structure when it is actually present (see
session notes / test development). The zero is specific to *this*
reservoir: `esn.run_esn` uses `tanh` (an odd function) with `bias = 0.0`
(this project's default, used in this spectrum experiment) and
`x0 = 0`, driven by symmetric `u ~ Uniform[-1, 1]`. Flipping the entire
input sequence's sign (`u -> -u`) therefore flips the sign of every
reservoir state exactly (`x_t -> -x_t`, by induction from `x_{-1} = 0`
through the update equation), for any leak rate alpha. So every linear
readout of the state, excluding its constant intercept, is an *odd* functional
of the input history, while a degree-p basis function picks up a factor
`(-1)^p` under that same flip
(`Ptilde_d(-u) = (-1)^d Ptilde_d(u)`). An odd predictor cannot correlate
with a centred even target in the population under a sign-symmetric input
distribution; the intercept does not change that covariance. Thus p=2 and
p=4 population capacity is zero. Finite-sample correlations need not be zero;
the exact zeros reported here are the observed clipped test-R^2 estimates.
A nonzero bias or asymmetric input can break the symmetry. In particular,
NARMA10 already uses asymmetric input, and the bias sweep below explicitly
uses nonzero bias.

**Odd-order capacity is heavily concentrated at p=3, not p=1, and rho
suppresses it.** Even though the *average* capacity per spec drops with
p (p=1 averages ~0.85-0.99 per spec across 9 specs; p=3 averages
~0.24-0.57 per spec across 165 specs), there are 165/9 ~ 18x more p=3
specs than p=1 specs in this window, so p=3 dominates the total by more
than 9:1. Raising rho from 0.95 to 1.3 barely changes the p=1 total
(8.845 -> 8.646, consistent with `fig_memory_curve.png`'s forgetting
curves being nearly identical at these small delays, k <= 8) but roughly
halves the p=3 total (93.568 -> 38.917) and with it the overall total
(102.41 -> 47.56) -- higher spectral radius trades away higher-order
nonlinear capacity in this delay window much more than it trades away
linear memory.

**Regression tie to the existing linear instrument.** The degree-1 row of
`capacity_spectrum` reproduces `metrics.memory_capacity`'s per-delay R^2
values to within 8e-5 (measured directly; the test asserts atol=1e-3),
confirming the new IPC instrument agrees with the old linear-only one
where they overlap, as required.

9. `/home/franblakia/fullfran/reservoir-computing/figures/fig_capacity_spectrum.png`

### Two controls that make the numbers above trustworthy

**1. Noise floor (surrogate targets).** A degree-3 capacity of ~90 against a
degree-1 capacity of ~9 is exactly the shape a finite-sample artifact would
take: there are 165 degree-3 basis functions against only 9 linear ones, so a
small positive bias per basis function, accumulated, would manufacture a large
fake total. Control: keep the same reservoir states, but build the basis
functions from an **independent** input sequence `u'` drawn from the same
distribution, which the reservoir has never seen. Any capacity measured there
is pure inflation.

| p | # specs | capacity (real `u`) | capacity (surrogate `u'`) |
|---|---|---|---|
| 1 | 9   |  8.82 | 0.00 |
| 2 | 45  |  0.00 | 0.00 |
| 3 | 165 | 90.37 | 0.00 |
| 4 | 495 |  0.00 | 0.00 |
| **total** | 714 | **99.18** | **0.00** |

The surrogate total is exactly zero in this run with 301 readout parameters
and ~2,750 test rows: no positive held-out R^2 survived clipping.
**This control supports the degree-3 signal**, but finite-sample surrogate
correlations can yield positive estimates in other runs.
(rho = 0.95, single seed, max_degree = 4, max_delay = 8.)

**2. The odd-symmetry claim, tested by its own prediction.** The explanation
for zero even-order capacity predicts that a nonzero bias should break the
symmetry and unlock it. Both halves were measured directly (rho = 0.95,
max_degree = 3, max_delay = 6, single seed):

```
max |x(-u) + x(u)|,  bias = 0.0  :  0.000e+00     state is exactly odd
max |x(-u) + x(u)|,  bias = 0.3  :  1.712e+00     symmetry broken
```

| bias | p=1 | p=2 | p=3 | total |
|---|---|---|---|---|
| 0.0 | 6.93 | **0.00**  | 59.93 | 66.86 |
| 0.3 | 6.94 | **24.93** | 52.56 | 84.43 |

The sign-flip identity holds to exactly zero floating-point error at
`bias = 0`, and even-order capacity appears the moment it is broken.

**The practical consequence is larger than it looks.** `bias = 0.0` is this
project's default in `esn.run_esn` and was used by the original figure
scripts. **The even-order population restriction applies to their symmetric-input
measurements, not to NARMA10's asymmetric input.** Turning on bias
raised total measured capacity by 26% here. For the connectome work this is a
protocol requirement, not a curiosity: comparing a fly network against null
models at `bias = 0` would be blind to any difference that lives in the even
orders.

## Bias sensitivity of the headline results

Every number in this repo up to this point was measured at `bias = 0.0`
(this project's implicit default). Measured with
`experiments/fig_bias_sweep.py`, N = 300, density = 0.1, T = 6000,
washout = 500, lam = 1e-6, alpha = 1.0, 10 seeds (seeds 0-9), sweeping
`bias` over `[0.0, 0.1, 0.3, 0.6, 1.0]` and re-running the exact
protocols of `fig_rho_sweep.py`, `fig_perturbation.py`, `fig_deff.py`
and `fig_capacity_spectrum.py` with `bias` threaded through every
`run_esn` call. **Reproduction check:** the `bias = 0.0` column
reproduces the published numbers exactly for the two quantities that
don't depend on grid resolution -- perturbation slopes -0.3605, -0.1171,
-0.0268 (rho = 0.5, 0.95, 1.3) and D_eff 9.28 / 1.35 (rho = 0.95, scale
0.1 / 10.0) -- and lands within the resolution of this script's coarser
12-point rho grid (0.4 to 1.6) for MC/NARMA10 (argmax rho = 1.273, MC =
16.69; argmin rho = 0.836, NMSE = 0.0614) against the published
20-point-grid values (rho ~= 1.20-1.21, MC ~= 16.7; rho ~= 0.78-0.81).
Full wall clock: 8 min 30 s.

**NARMA10 caveat, confirmed empirically below:** NARMA10 requires
`u ~ Uniform[0, 0.5]`, an input distribution that is already asymmetric
about zero, so the odd-symmetry mechanism that makes `bias = 0` special
never applied to it in the first place. NARMA10's optimum does move with
bias, but the *reason* is different from MC's -- it is not "even-order
information got unlocked", it is the ordinary effect of adding a
constant drive to an already-asymmetric-input reservoir.

### Headline result 1 -- optimal rho for MC vs NARMA10 on opposite sides of 1: MOVES, and breaks down

| bias | MC argmax rho | NARMA10 argmin rho | gap |
|------|------|------|------|
| 0.0 | 1.273 | 0.836 | +0.436 |
| 0.1 | 1.273 | 0.836 | +0.436 |
| 0.3 | 1.273 | 1.055 | +0.218 |
| 0.6 | 1.382 | 1.273 | +0.109 |
| 1.0 | 1.600 | 1.491 | +0.109 |

At `bias = 0.0` this reproduces the published picture (opposite sides of
1, about 0.4 apart). But the gap **shrinks monotonically to a quarter of
its original size** as bias grows, and by `bias = 0.6` **both optima have
moved to the same side of rho = 1** (NARMA10's argmin at rho = 1.273 is
no longer below 1). The "opposite sides of 1" framing does not survive a
nonzero bias much past 0.1-0.3 -- it is a `bias = 0` artifact, not a
robust structural property of this task pair. Both optima also drift
upward with bias (MC's argmax reaches rho = 1.600, the edge of the grid
we swept, at `bias = 1.0`, so the true argmax there may lie even higher
-- this grid was not widened past 1.6 to stay inside the runtime budget).
Peak MC itself falls substantially as bias grows (16.69 -> 10.95): a
nonzero bias does not just move the optimum, it lowers the ceiling.

### Headline result 2 -- rho = 1.3 still contracts: SURVIVES, and strengthens

| bias | rho=0.5 | rho=0.95 | rho=1.3 |
|------|------|------|------|
| 0.0 | -0.36053 | -0.11711 | -0.02684 |
| 0.1 | -0.36232 | -0.12076 | -0.02963 |
| 0.3 | -0.38199 | -0.14197 | -0.04335 |
| 0.6 | -0.45037 | -0.19666 | -0.08805 |
| 1.0 | -0.47895 | -0.28839 | -0.15616 |

The contraction verdict **never flips** across any tested bias for any
of the three rho values -- every slope stays negative. If anything, a
nonzero bias makes the contraction *stronger*: the rho = 1.3 slope goes
from -0.0268 (bias 0.0, "barely" contracting) to -0.1562 (bias 1.0),
almost a 6x increase in contraction strength, and the same monotonic
strengthening holds at rho = 0.5 and 0.95. This is unchanged, and this
is stated plainly: this headline result is a genuine, bias-independent
property of this driven reservoir, not an artifact of `bias = 0`.

### Headline result 3 -- D_eff collapses from 9.28 to 1.35 (rho=0.95): MOVES, and the effect nearly disappears

| bias | D_eff @ scale 0.1 | D_eff @ scale 10.0 | ratio |
|------|------|------|------|
| 0.0 | 9.28 | 1.35 | 6.89 |
| 0.1 | 6.80 | 1.35 | 5.04 |
| 0.3 | 3.87 | 1.36 | 2.85 |
| 0.6 | 2.42 | 1.39 | 1.74 |
| 1.0 | 1.69 | 1.45 | 1.16 |

The high-input-scale endpoint (D_eff @ 10.0) is essentially flat across
bias (1.35 -> 1.45): a strong enough input still saturates the reservoir
regardless of bias, same as before. But the low-input-scale endpoint
(D_eff @ 0.1) **collapses on its own as bias grows** -- from 9.28 down to
1.69 -- because a large constant bias is itself a saturating drive inside
the tanh, even before any input arrives. The two endpoints converge, and
the "collapse" ratio shrinks from 6.89x at `bias = 0.0` to just 1.16x at
`bias = 1.0`: at high bias there is almost nothing left to collapse,
because bias has already done most of the collapsing by itself. This
headline result is real at `bias = 0.0` but is substantially a `bias = 0`
artifact in its *magnitude* -- the mechanism ("strong input saturates the
reservoir") is still present, but a nonzero bias pre-empts most of it.

### The mechanism: even-order capacity switching on (rho = 0.95, max_degree = 3, max_delay = 6)

| bias | p=1 | p=2 | p=3 |
|------|------|------|------|
| 0.0 | 6.95 +/- 0.01 | 0.00 +/- 0.00 | 62.12 +/- 0.68 |
| 0.1 | 6.94 +/- 0.01 | 22.26 +/- 0.52 | 56.85 +/- 0.91 |
| 0.3 | 6.96 +/- 0.01 | 25.91 +/- 0.21 | 55.46 +/- 0.89 |
| 0.6 | 6.98 +/- 0.00 | 26.91 +/- 0.07 | 61.25 +/- 0.81 |
| 1.0 | 6.98 +/- 0.00 | 26.96 +/- 0.12 | 62.31 +/- 0.71 |

p=2 capacity is exactly zero at `bias = 0.0` (the odd-symmetry result)
and jumps to ~22-27 the moment bias becomes nonzero, then stays roughly
flat from `bias = 0.1` onward -- almost all of the even-order unlock
happens with just a small bias. p=1 is essentially unaffected by bias
(6.94-6.98 throughout); p=3 dips slightly at `bias = 0.1-0.3` before
recovering, with no clear monotonic trend. This confirms that bias opens an
even-order channel under symmetric input while also changing odd-order
capacity. It does not establish that this channel causes the shifts in
headline results 1 and 3; in particular, NARMA10's asymmetric input means
that the parity explanation does not apply to its optimum.

### Bottom line

Of the three headline results, only **one survives unchanged** (rho =
1.3 still contracts -- and contracts harder as bias grows). The other
two are **substantially bias = 0 artifacts**: the "opposite sides of
rho = 1" framing collapses onto the same side by `bias = 0.6`, and the
D_eff "collapse" shrinks from a 6.89x effect to a 1.16x effect as bias
grows, because bias itself pre-empts most of the collapse that used to
require a large input scale. This is a null result for two of three
headline claims as originally stated, and it is reported as such.

10. `/home/franblakia/fullfran/reservoir-computing/figures/fig_bias_sweep.png`

### Resolution limits of the bias sweep

Two of the numbers above are grid artifacts and must be read as such.

**1. The MC/NARMA gap is quantised by the rho grid.** The sweep uses
`linspace(0.4, 1.6, 12)`, so the step is **0.1091** and `rho = 1.0` is not
even a grid point (nearest neighbours: 0.945 and 1.055). Every reported gap is
an exact integer multiple of that step — 0.436, 0.436, 0.218, 0.109, 0.109 are
4, 4, 2, 1, 1 steps. So the final "gap = 0.109" means **one grid cell**, which
does not resolve whether the continuous optima coincide. The
*direction* of the trend (the gap closes monotonically as bias rises) is a real
measurement; the *value* at the bottom is a resolution floor.

Refining around the minimum at bias = 0.6 (10 seeds, mean +/- std):

| rho | NARMA10 test NMSE |
|---|---|
| 0.945 | 0.0648 +/- 0.0151 |
| 1.055 | 0.0552 +/- 0.0146 |
| **1.164** | **0.0496 +/- 0.0121** |
| **1.273** | **0.0489 +/- 0.0108** |
| 1.382 | 0.0609 +/- 0.0156 |
| 1.491 | 0.0999 +/- 0.0295 |

The two best sample means are 0.0007 apart against a +/- 0.011 spread.
These marginal standard deviations establish neither statistical equivalence
nor a significant difference: the matched seeds require paired differences
for inference. Descriptively, at bias = 0.6 the lowest tested means sit
**above** rho = 1, while rho = 0.945 has a higher mean. The sampled minimum
region moves above 1; neither a precise continuous argmin nor statistical
equivalence of the two best cells is established here.

**2. The MC optimum at bias = 1.0 is not running off the edge.** The sweep
reported `argmax = 1.600`, the right edge of the grid, which normally means the
optimum lies outside the window. Extending past it (10 seeds, bias = 1.0):

| rho | MC |
|---|---|
| 1.40 | 10.73 +/- 0.48 |
| **1.60** | **11.02 +/- 0.49** |
| 1.80 | 10.86 +/- 0.28 |
| 2.00 | 10.10 +/- 0.33 |
| 2.40 |  7.38 +/- 0.38 |

There is a genuine, broad maximum near rho ~ 1.6-1.8 and MC falls away after
it. The peak is flat enough that 1.4, 1.6 and 1.8 are within about one standard
deviation of each other, so "the MC optimum moves up with bias" is supported,
while the specific value 1.600 is not.

**Protocol note.** `experiments/fig_bias_sweep.py` does not modify
`narma_pipeline.py`, `metrics.py` or `capacity.py`; it reimplements each trial
locally so that `bias` can be threaded into `run_esn`. The faithfulness check
is that its `bias = 0.0` column reproduces the published record: perturbation
slopes -0.3605 / -0.1171 / -0.0268 and D_eff 9.28 / 1.35 match this file
exactly. The MC/NARMA optima differ from the published 20-point sweep only as
the coarser 12-point grid allows.

## Exploratory topology discrimination: full grid versus order totals

**The full grid classified these five ensembles better than its order totals in
this eight-block exploration. This is not a bias-selection result.** Historical
measurements and figures above are unchanged.

Run `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
experiments/fig_topology_discrimination.py --pilot` for a forecast; omit `--pilot`
for the experiment. The entry point caps these thread counts at four before
scientific imports, preserving lower positive settings. It refuses to overwrite
either new output. To reproduce, use a clean checkout without those generated
outputs, or explicitly preserve/remove only those two new outputs first.

### Matched protocol and ensemble definitions

N=300, exactly **9,000 directed edges**, no self-loops, rho=.95, input scale=1,
alpha=1, ridge lambda=1e-6, degree p<=3, maximum delay k<=6. Edge density is
9000/(300*299), approximately .10033; degree 30 permits an exactly regular
circulant rather than rounding a nominal .1 density differently by family.
`W[target, source]` is the orientation convention.

| Family | Definition |
|---|---|
| ER | Uniform fixed-edge-count graph, not independent Bernoulli edges |
| Regular circulant | Each node connects to offsets +/-1 through +/-15; a degree-30 local lattice, **not a simple ring** |
| Small-world | Rewire each circulant edge with probability .1; preserve source out-degree, prohibit loops and duplicate edges |
| Heavy-tailed fitness | Independent incoming/outgoing Pareto(shape=2)+1 node propensities; sample edges without replacement weighted by their product; **no fitted scale-free degree-law claim** |
| Modular | Four equal contiguous blocks; within/between edge-sampling propensity 8:1, fixed total edges |

Eight independent seed blocks pair the input-weight vector, input trajectories,
and pre-normalization Gaussian edge-weight multiset across all families and
biases 0, .1, .3. Separate RNG streams handle graph, weights, input weights,
fit input, and evaluation input. Each matrix is normalized to rho=.95:
**final weight variance and strengths need not match**, so these are controlled
ensemble comparisons, not pure topology effects at matched final weight statistics.

Each trial uses **two independent T=6000 input trajectories**, each with 500
washout rows, zero initial state and IID Uniform[-1,1] inputs. Fit on one,
evaluate on the other. This differs from the historical single-trajectory
50/50 split: totals are not directly comparable. Existing `fit_ridge`/`predict`
support all 119 targets as matrix columns, avoiding repeated matrix solves;
per-target held-out R² is clipped before aggregation. `capacity.py` is unchanged.

Flatten `grid[1:, :]` to 21 features or sum its delay axis to three order totals.
Both representations use exactly the same grids. Fixed nearest-centroid
classification uses training-only feature means/scales and holds out all five
families of one seed block. Each bias is evaluated separately; no tuning or bias
selection is performed. Chance accuracy is 20% (five balanced classes).

### Observed results and uncertainty

| Bias | Order totals | Full grid | Paired gain | Descriptive 95% gain interval | Blocks better / tied / worse |
|---|---|---|---|---|---|
| 0 | 32.5% | 52.5% | +20.0 percentage points | +10.0 to +30.0 | 6 / 2 / 0 |
| .1 | 35.0% | 52.5% | +17.5 percentage points | +2.5 to +30.0 | 6 / 1 / 1 |
| .3 | 30.0% | 42.5% | +12.5 percentage points | -5.0 to +27.5 | 5 / 2 / 1 |

Intervals resample the eight fixed out-of-fold block scores 2,000 times;
they are descriptive, not independent-fold confidence guarantees, because
training folds overlap. There are only 40 held-out predictions per bias.

199 within-block label permutations refit centroids, using the same permutations
for both representations and all biases. Max-statistic adjusted accuracy p-values
across all six comparisons are .130/.070/.190 for order totals and
.005/.005/.015 for the full grid (bias order 0/.1/.3). Adjusting the paired-gain
statistic across three biases gives .015/.020/.100. These test the **no-label-
association null**, not equality of model performance when a real signal exists;
the gain p-values are not proof of general full-grid superiority. Permutation
inference assumes labels are exchangeable within blocks under that null.
No ANOVA over grid cells, independent-fold claim, or selected best bias is used.

### Outputs, cost, and next step

- `experiments/results/topology_discrimination.json`: all per-seed grids, trial
  RNG seeds, normalization factors, configuration/definitions, versions, thread
  counts, predictions, block scores, permutation scores and resource measurements.
- `figures/fig_topology_discrimination.png`: mean accuracies and paired gains,
  with individual held-out block scores rather than hidden dispersion.
- Full experiment executed once: **40.69 s wall**, including **39.71 s measurement**;
  **82.62 s user CPU + 2.81 s system CPU**, peak Linux VmHWM **140.57 MiB**,
  four-thread caps. Matrix-target fitting makes this faster than the initial forecast.

The result supports retaining delay structure in this finite-window instrument,
but eight blocks and selected synthetic ensembles are not broad validation.
Replicate the fixed protocol on fresh blocks, inspect per-family confusions,
and then test sensitivity to final weight-statistic controls. Do not choose a
bias by total capacity or treat this result as a resolved bias recommendation.
