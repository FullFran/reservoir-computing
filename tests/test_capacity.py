"""Correctness tests for the information processing capacity (IPC) spectrum.

Run with: uv run pytest -q

These tests are written before `src/capacity.py` exists (strict TDD): the
first run of this file is expected to fail on import, then pass once the
implementation is in place.
"""

import numpy as np
import pytest

from capacity import (
    basis_function,
    capacity_of,
    capacity_spectrum,
    enumerate_specs,
    legendre_normalized,
)
from esn import make_input, make_reservoir, run_esn
from metrics import memory_capacity
from tasks import delayed_memory


# ---------------------------------------------------------------------------
# 1. legendre_normalized: hand-computed low-degree values
# ---------------------------------------------------------------------------


def test_legendre_normalized_degree1_hand_computed():
    u = np.array([-1.0, -0.5, 0.0, 0.3, 0.9])
    p1 = legendre_normalized(u, 1)
    expected = np.sqrt(3.0) * u
    np.testing.assert_allclose(p1, expected, atol=1e-12)


def test_legendre_normalized_degree2_hand_computed():
    u = np.array([-1.0, -0.5, 0.0, 0.3, 0.9])
    p2 = legendre_normalized(u, 2)
    expected = np.sqrt(5.0) * (3.0 * u**2 - 1.0) / 2.0
    np.testing.assert_allclose(p2, expected, atol=1e-12)


def test_legendre_normalized_degree0_is_ones():
    u = np.linspace(-1.0, 1.0, 9)
    p0 = legendre_normalized(u, 0)
    np.testing.assert_allclose(p0, np.ones_like(u))


def test_legendre_normalized_rejects_negative_degree():
    with pytest.raises(ValueError):
        legendre_normalized(np.array([0.1, -0.2]), -1)


# ---------------------------------------------------------------------------
# 2. Empirical orthonormality of the normalised Legendre family
# ---------------------------------------------------------------------------


def test_legendre_normalized_empirical_orthonormality():
    # Sample size and tolerance stated explicitly: with T = 200_000 i.i.d.
    # Uniform[-1, 1] draws, the standard error of a mean of a bounded,
    # order-1 product Ptilde_a * Ptilde_b is O(1/sqrt(T)) ~ 0.002-0.003
    # for the degrees tested here; atol = 0.01 gives a >3-sigma margin
    # around the true value (0 or 1) without hiding a loose tolerance.
    T = 200_000
    rng = np.random.default_rng(0)
    u = rng.uniform(-1.0, 1.0, size=T)

    polys = {d: legendre_normalized(u, d) for d in range(5)}
    for a in range(5):
        for b in range(5):
            product_mean = float(np.mean(polys[a] * polys[b]))
            expected = 1.0 if a == b else 0.0
            assert product_mean == pytest.approx(expected, abs=0.01), (a, b)


# ---------------------------------------------------------------------------
# 3. basis_function with a single (k, 1) spec == sqrt(3) * delayed_memory
# ---------------------------------------------------------------------------


def test_basis_function_single_linear_factor_matches_delayed_memory():
    rng = np.random.default_rng(1)
    u = rng.uniform(-1.0, 1.0, size=50)
    for k in (0, 1, 7, 20):
        z = basis_function(u, [(k, 1)])
        expected = np.sqrt(3.0) * delayed_memory(u, k)
        np.testing.assert_allclose(z, expected, atol=1e-12)


def test_basis_function_rejects_duplicate_delays():
    u = np.linspace(-1.0, 1.0, 10)
    with pytest.raises(ValueError):
        basis_function(u, [(1, 1), (1, 2)])


def test_basis_function_rejects_degree_below_one():
    u = np.linspace(-1.0, 1.0, 10)
    with pytest.raises(ValueError):
        basis_function(u, [(0, 0)])


def test_basis_function_rejects_empty_spec():
    u = np.linspace(-1.0, 1.0, 10)
    with pytest.raises(ValueError):
        basis_function(u, [])


# ---------------------------------------------------------------------------
# 4. capacity_of: perfectly reachable target vs. independent noise
# ---------------------------------------------------------------------------


def test_capacity_of_is_one_for_a_state_column_itself():
    rng = np.random.default_rng(11)
    T, n = 2000, 20
    states = rng.standard_normal((T, n))
    washout = 100

    target = states[:, 3].copy()
    capacity = capacity_of(states, target, washout=washout, lam=1e-8)
    assert capacity == pytest.approx(1.0, abs=1e-6)


def test_capacity_of_is_near_zero_for_independent_noise():
    rng = np.random.default_rng(12)
    T, n = 4000, 20
    states = rng.standard_normal((T, n))
    washout = 200

    noise_target = rng.standard_normal(T)
    capacity = capacity_of(states, noise_target, washout=washout, lam=1e-6)
    assert capacity < 0.05


# ---------------------------------------------------------------------------
# 5. enumerate_specs: exact hand-checkable set for a small case
# ---------------------------------------------------------------------------


def test_enumerate_specs_small_case_exact_set():
    specs = enumerate_specs(max_degree=2, max_delay=2)

    # Hand enumeration for max_degree=2, max_delay=2 (delays 0, 1, 2):
    #   1 factor,  degree 1: (0,1) (1,1) (2,1)               -> 3
    #   1 factor,  degree 2: (0,2) (1,2) (2,2)               -> 3
    #   2 factors, degrees (1,1) summing to 2, distinct delays:
    #       {0,1} {0,2} {1,2}                                -> 3
    # total = 9
    expected = {
        ((0, 1),),
        ((1, 1),),
        ((2, 1),),
        ((0, 2),),
        ((1, 2),),
        ((2, 2),),
        ((0, 1), (1, 1)),
        ((0, 1), (2, 1)),
        ((1, 1), (2, 1)),
    }
    assert set(specs) == expected
    assert len(specs) == len(expected) == 9


def test_enumerate_specs_rejects_bad_params():
    with pytest.raises(ValueError):
        enumerate_specs(max_degree=0, max_delay=2)
    with pytest.raises(ValueError):
        enumerate_specs(max_degree=2, max_delay=-1)


# ---------------------------------------------------------------------------
# 6. Regression tie: degree-1 row of capacity_spectrum == memory_capacity
# ---------------------------------------------------------------------------


def test_capacity_spectrum_degree1_matches_memory_capacity():
    # Degree-1 normalised Legendre is Ptilde_1(u) = sqrt(3) * u, and R^2
    # is invariant to a positive rescaling of the target (ridge/OLS
    # coefficients rescale linearly, so predictions rescale the same way
    # and the ratio SS_res / SS_tot is unchanged). So capacity_spectrum's
    # degree-1 row must reproduce metrics.memory_capacity's per-delay R^2
    # values, up to the fact that capacity_of does not additionally
    # truncate the trailing `k` rows the way memory_capacity's index-based
    # slicing does -- the two use very slightly different but heavily
    # overlapping post-washout windows (differing by at most max_delay
    # rows out of ~2900). Measured directly, the largest discrepancy this
    # produces here is ~8e-5; atol = 1e-3 keeps more than 10x margin
    # while still being a tight, non-hidden tolerance.
    n, T, washout, max_delay = 150, 3200, 300, 6
    Win = make_input(n, scale=1.0, seed=3)
    rng = np.random.default_rng(55)
    u = rng.uniform(-1.0, 1.0, size=T)
    W = make_reservoir(n, density=0.1, rho=0.9, seed=3)
    states = run_esn(u, W, Win, alpha=1.0)

    _, per_delay_r2 = memory_capacity(
        states, u, max_delay=max_delay, washout=washout, lam=1e-6
    )

    _, _, grid = capacity_spectrum(
        states, u, max_degree=1, max_delay=max_delay, washout=washout, lam=1e-6
    )

    expected = np.clip(per_delay_r2, 0.0, None)
    np.testing.assert_allclose(grid[1, :], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# 7. Dambre bound: total capacity <= number of readout parameters
# ---------------------------------------------------------------------------


def test_dambre_bound_holds_on_small_reservoir():
    n, T, washout = 30, 3000, 200
    Win = make_input(n, scale=1.0, seed=9)
    rng = np.random.default_rng(99)
    u = rng.uniform(-1.0, 1.0, size=T)
    W = make_reservoir(n, density=0.3, rho=0.9, seed=9)
    states = run_esn(u, W, Win, alpha=1.0)

    total, _, _ = capacity_spectrum(
        states, u, max_degree=3, max_delay=8, washout=washout, lam=1e-6
    )

    n_readout_params = n + 1  # N reservoir units + 1 bias
    assert total <= n_readout_params + 1e-6
