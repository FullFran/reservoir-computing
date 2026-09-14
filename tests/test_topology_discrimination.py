"""Controls for the exploratory topology comparison."""

import numpy as np
import pytest

from esn import fit_ridge, predict

from topology_discrimination import (
    FAMILIES, classify, evaluate, features, make_mask, matched_weights, measure_grid,
    permutation_scores,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_graph_invariants_and_reproducibility(family):
    mask = make_mask(family, 40, 6, 12)
    assert mask.shape == (40, 40)
    assert mask.sum() == 240
    assert not np.diag(mask).any()
    np.testing.assert_array_equal(mask, make_mask(family, 40, 6, 12))
    if family in ("regular_circulant", "small_world"):
        np.testing.assert_array_equal(mask.sum(axis=0), 6)


def test_weight_controls_and_radius():
    values = np.random.default_rng(3).normal(size=240)
    for family in FAMILIES:
        mask = make_mask(family, 40, 6, 12)
        w, factor = matched_weights(mask, values, .95)
        np.testing.assert_allclose(np.sort(w[mask] / factor), np.sort(values))
        assert np.max(np.abs(np.linalg.eigvals(w))) == pytest.approx(.95)


def test_feature_mapping():
    grids = np.arange(2 * 5 * 4 * 7).reshape(2, 5, 4, 7)
    full, totals = features(grids)
    assert full.shape == (2, 5, 21)
    np.testing.assert_array_equal(totals, full.reshape(2, 5, 3, 7).sum(axis=-1))


def test_fold_training_excludes_held_out_block():
    x = np.random.default_rng(4).normal(size=(8, 5, 3))
    _, original = classify(x)
    x[0] += 10000
    _, changed = classify(x)
    for key in ("mean", "scale", "centroids"):
        np.testing.assert_array_equal(original[0][key], changed[0][key])
    assert original[0]["train_blocks"] == list(range(1, 8))
    labels = np.tile(np.arange(5), (8, 1))
    labels[0] = labels[0, ::-1]
    _, relabeled = classify(x, labels)
    np.testing.assert_array_equal(changed[0]["centroids"], relabeled[0]["centroids"])


def test_classifier_positive_and_null_controls():
    positive = np.tile(np.eye(5), (8, 1, 1))
    predicted, _ = classify(positive)
    np.testing.assert_array_equal(predicted, np.tile(np.arange(5), (8, 1)))
    predicted, _ = classify(np.zeros((8, 5, 3)))
    assert np.mean(predicted == np.arange(5)) == .2
    labels = permutation_scores(8, 5, 19, 31)
    np.testing.assert_array_equal(labels, permutation_scores(8, 5, 19, 31))
    for permutation in labels:
        for block in permutation:
            np.testing.assert_array_equal(np.sort(block), np.arange(5))
    null_scores = [np.mean(classify(positive, y)[0] == y) for y in labels]
    assert np.mean(null_scores) < .4


def test_independent_trajectory_measurement_is_deterministic():
    config = dict(n=20, degree=4, rho=.95, t=200, washout=30,
                  max_degree=2, max_delay=2, lam=1e-6, alpha=1., scale=1.)
    a, metadata = measure_grid("modular", 0, .1, config)
    b, _ = measure_grid("modular", 0, .1, config)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, 3) and np.isfinite(a).all()
    assert metadata["fit_input_seed"] != metadata["eval_input_seed"]
    assert (a >= 0).all()


def test_permutation_statistics_and_pairing_controls():
    # Identical all-zero representations: ties give exactly chance and no advantage.
    grids = np.zeros((3, 8, 5, 4, 7))
    result = evaluate(grids, permutations=9)
    np.testing.assert_allclose(result["accuracy"], .2)
    np.testing.assert_allclose(result["full_minus_totals"], 0)
    np.testing.assert_allclose(result["adjusted_p_accuracy"], 1)
    np.testing.assert_allclose(result["adjusted_p_delta"], 1)
    np.testing.assert_array_equal(result["delta_interval"], np.zeros((2, 3)))


def test_multi_target_readout_matches_separate_fits():
    rng = np.random.default_rng(23)
    fit_x, test_x = rng.normal(size=(2, 100, 8))
    targets = rng.normal(size=(100, 5))
    batched = predict(test_x, fit_ridge(fit_x, targets, 1e-6, 0))
    separate = np.column_stack([
        predict(test_x, fit_ridge(fit_x, target, 1e-6, 0)) for target in targets.T
    ])
    np.testing.assert_allclose(batched, separate, atol=1e-12)


def test_saved_results_statistics_reproduce_without_simulation():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments/results/topology_discrimination.json"
    result = json.loads(path.read_text())
    grids = np.asarray(result["grids"])
    assert grids.shape == (3, 8, 5, 4, 7)
    assert np.isfinite(grids).all()
    np.testing.assert_array_equal(grids[..., 0, :], 0)
    actual = evaluate(grids, result["permutations"])
    for key in ("accuracy", "predictions", "full_minus_totals", "adjusted_p_accuracy",
                "adjusted_p_delta", "delta_interval"):
        np.testing.assert_allclose(actual[key], result["statistics"][key])
    assert len(result["trials"]) == 120
    for block in range(8):
        trials = [trial for trial in result["trials"] if trial["block"] == block]
        for key in ("weight_seed", "input_weight_seed", "fit_input_seed", "eval_input_seed"):
            assert len({trial[key] for trial in trials}) == 1


def test_entrypoint_caps_threads_and_preserves_lower_setting():
    import os
    from pathlib import Path
    import subprocess
    import sys

    directory = Path(__file__).resolve().parents[1] / "experiments"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="32", MKL_NUM_THREADS="bad")
    command = "import fig_topology_discrimination; import os; print(*(os.environ[v] for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')))"
    output = subprocess.check_output([sys.executable, "-c", command], cwd=directory, env=env, text=True)
    assert output.strip() == "1 4 4"
