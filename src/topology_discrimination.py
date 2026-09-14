"""Matched graph ensembles and seed-block-held-out spectrum classification.

W[target, source] is directed; diagonal entries are always absent. The
circulant is a degree-d local lattice, NOT a simple ring. The fitness family
uses Pareto node propensities, not a claim of a fitted scale-free degree law.
"""

import numpy as np

from capacity import basis_function, enumerate_specs
from esn import fit_ridge, make_input, predict, run_esn

FAMILIES = ("erdos_renyi", "regular_circulant", "small_world",
            "heavy_tailed_fitness", "modular")


def make_mask(family, n, degree, seed):
    """Exactly n*degree directed edges, no loops; n divisible by four.

    ER: uniform fixed-edge-count sampling. Circulant: degree/2 nearest
    neighbours on either side. Small-world: independently rewire each
    circulant edge with probability .1, preserving source out-degree.
    Fitness: weighted sampling without replacement, Pareto(shape=2)+1
    independent incoming/outgoing propensities. Modular: four equal blocks,
    within-block sampling propensity 8 versus 1 between blocks.
    """
    if family not in FAMILIES or degree % 2 or not 0 < degree < n or n % 4:
        raise ValueError("Require a known family, even degree < n, and n divisible by four")
    rng = np.random.default_rng(seed)
    mask = np.zeros((n, n), dtype=bool)
    if family in ("regular_circulant", "small_world"):
        for source in range(n):
            for offset in range(1, degree // 2 + 1):
                mask[(source + offset) % n, source] = True
                mask[(source - offset) % n, source] = True
        if family == "small_world":
            for target, source in np.argwhere(mask):
                if rng.random() < .1:
                    mask[target, source] = False
                    candidates = np.flatnonzero(~mask[:, source] & (np.arange(n) != source))
                    mask[rng.choice(candidates), source] = True
        return mask
    propensity = np.ones((n, n))
    if family == "heavy_tailed_fitness":
        incoming, outgoing = rng.pareto(2., size=(2, n)) + 1
        propensity = incoming[:, None] * outgoing[None, :]
    elif family == "modular":
        blocks = np.arange(n) // (n // 4)
        propensity = np.where(blocks[:, None] == blocks[None, :], 8., 1.)
    # Exponential race implements weighted sampling without replacement.
    priority = rng.exponential(size=(n, n)) / propensity
    np.fill_diagonal(priority, np.inf)
    indices = np.argpartition(priority.ravel(), n * degree - 1)[:n * degree]
    mask.ravel()[indices] = True
    return mask


def matched_weights(mask, values, rho):
    """Same pre-normalization Gaussian multiset; one scalar radius correction.

    Final weight variance/strength need not match: spectral normalization is
    topology dependent. This is an ensemble comparison, not topology isolated
    from every weight statistic.
    """
    w = np.zeros(mask.shape)
    w[mask] = values
    radius = np.max(np.abs(np.linalg.eigvals(w)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Cannot normalize zero/nonfinite spectral radius")
    factor = float(rho / radius)
    return w * factor, factor


def measure_grid(family, block, bias, config):
    """Independent fit/evaluation runs, T rows EACH, washout on EACH.

    Reuse existing multi-target ridge API rather than modifying capacity.py.
    Per-column test R2 is clipped before aggregation, matching capacity_of.
    Inputs, input weights, and Gaussian weight multiset are paired across
    families/biases within a block. RNG streams are separate by purpose.
    """
    c = config
    streams = np.random.SeedSequence([20260914, block]).generate_state(5)
    graph_seed, weight_seed, input_seed, fit_seed, eval_seed = map(int, streams)
    mask = make_mask(family, c["n"], c["degree"], graph_seed)
    values = np.random.default_rng(weight_seed).normal(size=int(mask.sum()))
    w, factor = matched_weights(mask, values, c["rho"])
    win = make_input(c["n"], c["scale"], input_seed)
    specs = enumerate_specs(c["max_degree"], c["max_delay"])
    if c["washout"] < c["max_delay"]:
        raise ValueError("washout must cover target delays")
    trajectories = []
    for seed in (fit_seed, eval_seed):
        u = np.random.default_rng(seed).uniform(-1, 1, c["t"])
        states = run_esn(u, w, win, c["alpha"], bias=bias)
        targets = np.column_stack([basis_function(u, spec) for spec in specs])
        trajectories.append((states[c["washout"]:], targets[c["washout"]:]))
    (x_train, y_train), (x_test, y_test) = trajectories
    readout = fit_ridge(x_train, y_train, c["lam"], washout=0)
    residual = np.sum((y_test - predict(x_test, readout)) ** 2, axis=0)
    variance = np.sum((y_test - y_test.mean(axis=0)) ** 2, axis=0)
    scores = np.maximum(1 - residual / variance, 0)
    grid = np.zeros((c["max_degree"] + 1, c["max_delay"] + 1))
    for spec, score in zip(specs, scores):
        grid[sum(d for _, d in spec), max(k for k, _ in spec)] += score
    metadata = dict(graph_seed=graph_seed, weight_seed=weight_seed,
                    input_weight_seed=input_seed, fit_input_seed=fit_seed,
                    eval_input_seed=eval_seed, spectral_scale=factor,
                    final_weight_std=float(w[mask].std()))
    return grid, metadata


def features(grids):
    """Input (..., p+1, k+1); omit empty p=0, flatten p-major or sum k."""
    active = np.asarray(grids)[..., 1:, :]
    return active.reshape(*active.shape[:-2], -1), active.sum(axis=-1)


def classify(x, labels=None):
    """Leave one entire seed block out; scaler/centroids use train only.

    Fixed nearest centroid, Euclidean distance, equal class priors, no tuning.
    Zero-variance training features use scale=1; ties select first class.
    """
    blocks, classes, _ = x.shape
    if labels is None:
        labels = np.tile(np.arange(classes), (blocks, 1))
    predictions = np.empty((blocks, classes), dtype=int)
    diagnostics = []
    for held in range(blocks):
        train_blocks = [i for i in range(blocks) if i != held]
        train = x[train_blocks].reshape(-1, x.shape[-1])
        y = labels[train_blocks].ravel()
        mean, scale = train.mean(axis=0), train.std(axis=0)
        scale = np.where(scale > 0, scale, 1.)
        train = (train - mean) / scale
        centroids = np.array([train[y == label].mean(axis=0) for label in range(classes)])
        test = (x[held] - mean) / scale
        predictions[held] = np.argmin(((test[:, None] - centroids) ** 2).sum(axis=-1), axis=1)
        diagnostics.append(dict(train_blocks=train_blocks, mean=mean,
                                scale=scale, centroids=centroids))
    return predictions, diagnostics


def permutation_scores(blocks, classes, count, seed):
    """Within-block label permutations, shared across representations/biases."""
    rng = np.random.default_rng(seed)
    return np.array([[rng.permutation(classes) for _ in range(blocks)] for _ in range(count)])


def evaluate(grids, permutations=199):
    """grids[bias, block, family, p, k]; max-statistic multiplicity controls."""
    full, totals = features(grids)
    blocks, classes = grids.shape[1:3]
    labels = np.tile(np.arange(classes), (blocks, 1))
    representations = (totals, full)
    predictions = np.array([[classify(x)[0] for x in representation]
                            for representation in representations])
    block_scores = (predictions == labels).mean(axis=-1)
    observed = block_scores.mean(axis=-1)
    permuted_labels = permutation_scores(blocks, classes, permutations, 701)
    null = np.array([[[np.mean(classify(x, y)[0] == y) for x in representation]
                      for representation in representations] for y in permuted_labels])
    # Same label permutation across biases and both feature representations.
    max_accuracy = null.max(axis=(1, 2))
    delta = observed[1] - observed[0]
    max_delta = (null[:, 1] - null[:, 0]).max(axis=1)
    p_accuracy = (1 + (max_accuracy[:, None, None] >= observed).sum(axis=0)) / (permutations + 1)
    p_delta = (1 + (max_delta[:, None] >= delta).sum(axis=0)) / (permutations + 1)
    rng = np.random.default_rng(702)
    indices = rng.integers(blocks, size=(2000, blocks))
    # Descriptive resampling of fixed out-of-fold block scores, not independent-fold inference.
    resampled = block_scores[..., indices].mean(axis=-1)
    return dict(accuracy=observed.tolist(), full_minus_totals=delta.tolist(),
                block_accuracy=block_scores.tolist(), predictions=predictions.tolist(),
                accuracy_interval=np.quantile(resampled, [.025, .975], axis=-1).tolist(),
                delta_interval=np.quantile(resampled[1] - resampled[0], [.025, .975], axis=-1).tolist(),
                adjusted_p_accuracy=p_accuracy.tolist(), adjusted_p_delta=p_delta.tolist(),
                permutation_accuracy=null.tolist(), permutation_seed=701,
                bootstrap_seed=702, bootstrap_replicates=2000)
