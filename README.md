# reservoir-computing

**I built the smallest possible recurrent network I could train, and it disagreed with the rule everybody repeats.**

A from-scratch Echo State Network in pure NumPy — no reservoir-computing library, no
deep-learning framework, ~200 lines of actual algorithm. Then eight experiments to find
out what the thing is really doing.

This is a curiosity project. The goal was never a benchmark number; it was to watch
memory *appear* out of iterating a matrix, and to check the folklore against measurement.

---

## The idea in one paragraph

Training a recurrent network is expensive and fragile: gradients have to travel backwards
through time and they explode or vanish on the way. Reservoir computing refuses to play.
Take a **random, fixed** recurrent network. Never train it. Use it purely as an expander
that turns the history of an input signal into a high-dimensional state. Train **only a
linear readout** on top.

Because the only trainable part is linear, training stops being gradient descent and
becomes **ridge regression** — one linear solve, closed form, no learning rate, no local
minima.

```
x[t+1] = (1-α)·x[t] + α·tanh(W·x[t] + W_in·u[t] + b)      ← fixed, random, never trained
ŷ[t]   = W_out · x[t]                                      ← the only thing that learns
```

---

## Three things the measurements say that the tutorials don't

### 1. "Set spectral radius to 0.99 because edge of chaos" is wrong in both directions

<img src="figures/fig_rho_sweep.png" width="100%" alt="Memory capacity and NARMA10 error versus spectral radius">

Linear memory capacity peaks at **ρ ≈ 1.20–1.21**. The NARMA10 error bottoms out at
**ρ ≈ 0.80**. The two optima sit on *opposite sides* of ρ = 1, about 0.4 apart.
ρ = 1 is optimal for neither task.

There is no good value of ρ. There is a good value *per task*, because memory and
nonlinearity compete for the same budget.

### 2. ρ = 1.3 is not chaotic — it contracts

Twin-trajectory test: same input, initial states separated by ε = 10⁻⁹, measure whether
the gap grows.

| ρ | fitted slope (log₁₀ / step) | verdict |
|---|---|---|
| 0.50 | −0.361 | contracts |
| 0.95 | −0.117 | contracts |
| 1.30 | −0.027 | **still contracts** |

All three converge. The input drives neurons into the flat regions of the `tanh`, the
effective contraction rises, and the driven system keeps its Echo State Property well
past ρ = 1 — which is exactly why memory capacity was still climbing there in the plot
above. The spectral radius of `W` is a hint, not a criterion. Measure it instead.

This is the empirical version of Yildiz, Jaeger & Kiebel (2012).

### 3. A 300-neuron reservoir was using nine dimensions

<img src="figures/fig_deff.png" width="100%" alt="Effective dimension versus input scaling">

Effective dimension (participation ratio of the state covariance) collapses from
**9.28** at input scaling 0.1 to **1.35** at scaling 10 — out of **N = 300**.
Strong input saturates the `tanh`, every neuron synchronises, and the reservoir
flattens to essentially one dimension.

Input scaling is the least-tuned knob in the field and it silently decides how much of
your network exists.

---

## Why this matters more than it looks

Dambre et al. (2012) proved that the total information processing capacity of a fading-memory
dynamical system is bounded by the number of linearly independent state variables:

```
C_total = Σ C[z]  ≤  N
```

Measured peak here: **MC ≈ 16.7 against a ceiling of N = 300**. The reservoir is using
roughly **5% of the capacity theory allows it**.

That bound reframes the question this project is heading towards. Running the *Drosophila*
connectome (FlyWire, 139,255 neurons; MaleCNS v1.0, 166,700) as a reservoir cannot beat a
degree-preserving null model *on total capacity* at matched N — topology can only
**redistribute** capacity between linear memory and nonlinearities of different order, and
get closer to or further from the ceiling.

So the interesting question is not "does the fly win?" but **"where does the fly spend its
capacity budget, and how close to N does it get?"**

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Nothing else.

```bash
git clone https://github.com/FullFran/reservoir-computing
cd reservoir-computing

uv run pytest -q                                  # 5 tests, ~0.5 s
uv run python experiments/fig_memory_curve.py     # regenerate one figure
for f in experiments/fig_*.py; do uv run python "$f"; done   # regenerate all eight
```

Every experiment fixes its seeds. Curves are averaged over 10 seeds and plotted with
their spread — a result without dispersion is not a result.

---

## What's in here

```
src/esn.py           reservoir construction, leaky-integrator update, ridge readout
src/tasks.py         delayed memory, delayed product, NARMA10
src/metrics.py       NMSE, R², memory capacity, effective dimension, perturbation growth
experiments/         eight scripts, one per figure
figures/             the eight figures, regenerable from scratch
tests/               spectral radius exactness, NARMA10 ground truth, ridge recovery
RESULTS.md           every measured number, with the conditions that produced it
notes/index.html     illustrated study notes (Spanish) — open it in a browser
```

### `notes/index.html`

Thirteen sections from the minimal ESN to the connectome question, with rendered
equations, hand-drawn diagrams and every figure in this repo. Fully self-contained:
fonts embedded, MathJax vendored, **works offline with no network access**.
Written in Spanish; the code and this README are English.

---

## A bug worth writing down

The first forgetting curve came out non-monotonic, which is physically impossible for a
fading-memory system. Cause: `run_esn` stored the state *before* consuming `u[t]`, so the
k = 0 memory task was being asked to reconstruct an input the state had never seen.

`states[t]` must reflect `u[t]`. An off-by-one in an index convention, invisible in the
tests that existed, and it produced a plot that looked plausible enough to publish.
That's the interesting kind of bug.

---

## Where this goes next

1. Synthetic topologies at matched N and density — Erdős–Rényi, small-world, scale-free, modular.
2. A *Drosophila* subnetwork (mushroom body) from FlyWire, as `W`.
3. Degree-preserving directed null models, tightened step by step: same degrees → same
   weight distribution → same strength → same spatial structure → same cell-type composition.
4. If the difference survives to the tight controls, that is a real result about
   connectome organisation. If it dies at the first one, it was the degree distribution —
   also a result.

---

## References

1. **Jaeger, H.** (2001). *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148.
2. **Maass, W., Natschläger, T. & Markram, H.** (2002). *Real-time computing without stable states.* Neural Computation 14(11).
3. **Lukoševičius, M.** (2012). *A practical guide to applying echo state networks.* In *Neural Networks: Tricks of the Trade*.
4. **Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S.** (2012). *Information processing capacity of dynamical systems.* Scientific Reports 2, 514.
5. **Yildiz, I. B., Jaeger, H. & Kiebel, S. J.** (2012). *Re-visiting the echo state property.* Neural Networks 35.
6. **Appeltant, L. et al.** (2011). *Information processing using a single dynamical node as complex system.* Nature Communications 2, 468.

---

## Licensing

- **Code, figures and notes text** — MIT, see [LICENSE](LICENSE).
- **Photographs in `notes/img/`** — third-party, CC BY 2.0 / CC BY-SA 2.5 / CC BY-SA 4.0.
  Per-image source, licence and required attribution in [`notes/img/CREDITS.md`](notes/img/CREDITS.md).
- **`notes/vendor/tex-svg.js`** — MathJax 3.2.2, Apache-2.0, vendored so the notes render offline.
- **Fonts embedded in `notes/index.html`** — Bricolage Grotesque, Newsreader and JetBrains Mono,
  SIL Open Font License 1.1.
