"""Reproducible exploratory full-grid versus order-total discrimination.

Run --pilot first; the full run refuses to overwrite either output.
Only this entry point sets thread limits, before scientific imports.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        requested = int(os.environ.get(variable, "4"))
    except ValueError:
        requested = 4
    os.environ[variable] = str(min(4, max(1, requested)))

import argparse
import json
from pathlib import Path
import resource
import time

import common  # Adds src to the import path.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topology_discrimination import FAMILIES, evaluate, measure_grid

CONFIG = dict(n=300, degree=30, rho=.95, t=6000, washout=500,
              scale=1., alpha=1., lam=1e-6, max_degree=3, max_delay=6)
BIASES = [0., .1, .3]
ROOT = Path(__file__).resolve().parent.parent


def peak_rss_mib():
    # Linux ru_maxrss can retain the launcher's pre-exec high-water mark.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("Linux VmHWM unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        grid, _ = measure_grid(FAMILIES[0], 0, .1, CONFIG)
        elapsed = time.perf_counter() - started
        print(json.dumps(dict(pilot_seconds=elapsed, projected_trials_seconds=elapsed * 120,
                              peak_rss_mib=peak_rss_mib(),
                              grid_total=float(grid.sum()))))
        return
    output = ROOT / "experiments/results/topology_discrimination.json"
    figure = ROOT / "figures/fig_topology_discrimination.png"
    if output.exists() or figure.exists():
        raise SystemExit("Refusing to overwrite existing experiment outputs")
    grids = np.empty((3, 8, 5, 4, 7))
    metadata = []
    for b, bias in enumerate(BIASES):
        for block in range(8):
            for f, family in enumerate(FAMILIES):
                grids[b, block, f], details = measure_grid(family, block, bias, CONFIG)
                metadata.append(dict(bias=bias, block=block, family=family, **details))
            print(f"bias={bias} block={block} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    measurement_seconds = time.perf_counter() - started
    statistics = evaluate(grids)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scores = np.array(statistics["block_accuracy"])
    for r, label in enumerate(("Order totals (3 features)", "Full grid (21 features)")):
        axes[0].plot(BIASES, scores[r].mean(axis=-1), "o-", label=label)
        for b, bias in enumerate(BIASES):
            axes[0].scatter(np.full(8, bias) + (r - .5) * .007, scores[r, b], s=10, alpha=.4)
    axes[0].axhline(.2, color="gray", linestyle="--", label="Chance (20%)")
    axes[0].set(xlabel="Bias", ylabel="Held-out accuracy", ylim=(0, 1.05))
    axes[0].legend(fontsize=8)
    delta = scores[1] - scores[0]
    axes[1].plot(BIASES, delta.mean(axis=-1), "o-", color="black")
    for b, bias in enumerate(BIASES):
        axes[1].scatter(np.full(8, bias), delta[b], s=16, alpha=.5)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set(xlabel="Bias", ylabel="Paired full-grid minus totals accuracy")
    fig.suptitle("Exploratory topology discrimination: 8 held-out seed blocks")
    fig.tight_layout()
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    payload = dict(schema_version=1, config=CONFIG, biases=BIASES, families=FAMILIES,
                   blocks=list(range(8)), edges=9000, self_loops=False,
                   graph_definitions=dict(
                       erdos_renyi="Uniform fixed 9000 directed edges without replacement",
                       regular_circulant="30 neighbours: offsets +/-1 through +/-15; not a simple ring",
                       small_world="Circulant with edge rewiring probability .1; source out-degree preserved",
                       heavy_tailed_fitness="Independent Pareto(shape=2)+1 incoming/outgoing node propensities; weighted edge sampling without replacement; no power-law degree claim",
                       modular="Four equal contiguous blocks, within/between propensity 8:1; fixed edge count"),
                   weight_control="Shared Gaussian multiset before topology-dependent spectral normalization; final variance/strength not matched",
                   rng_protocol="SeedSequence([20260914, block]).generate_state(5): graph, weights, input weights, fit input, evaluation input; graph stream paired across families",
                   trajectory_protocol="Independent fit/evaluation inputs: 6000 rows each, 500 washout each",
                   grid_axes=["bias", "block", "family", "order", "maximum_delay"],
                   representation_order=["order_totals", "full_grid"],
                   grids=grids.tolist(), trials=metadata, statistics=statistics,
                   classifier="Train-only standardized nearest centroid; leave-one-seed-block-out",
                   permutations=199, permutation_control="Within-block labels; max over 6 accuracy tests and 3 paired-delta tests separately",
                   permutation_null="No topology-label association within seed blocks; delta p-values are not tests of equal predictive performance under an existing signal",
                   uncertainty="Descriptive percentile resampling of 8 fixed out-of-fold block scores; overlapping training folds are dependent",
                   versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
                   threads={v: os.environ[v] for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                   resources=dict(measurement_seconds=measurement_seconds,
                                  wall_seconds=time.perf_counter()-started,
                                  user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                                  peak_rss_mib=peak_rss_mib()))
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(accuracy=statistics["accuracy"], delta=statistics["full_minus_totals"],
                         p_delta=statistics["adjusted_p_delta"], resources=payload["resources"])))


if __name__ == "__main__":
    main()
