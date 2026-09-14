"""Information processing capacity (IPC) spectrum, per Dambre et al. (2012).

`esn.py` and `metrics.py` measure only *linear* memory capacity: how well
a linear readout reconstructs a delayed copy of the scalar input. This
module generalises that to the full IPC spectrum C(k, p), where `k` is a
temporal depth and `p` is a polynomial order, following Dambre, Verstraeten,
Schrauwen & Massar, "Information Processing Capacity of Dynamical
Systems" (2012).

Setup
-----
The input is u[t] ~ Uniform[-1, 1], i.i.d. The orthogonal polynomial
family for that measure is the Legendre family P_d, orthogonal on
[-1, 1]:

    E[ P_a(u) P_b(u) ] = delta_ab / (2a + 1)

so the *normalised* family used everywhere below is

    Ptilde_d(u) = sqrt(2d + 1) * P_d(u)      =>   E[Ptilde_a Ptilde_b] = delta_ab

A *basis function* is indexed by a set of (delay, degree) pairs with
distinct delays and degrees >= 1:

    z_A[t] = prod_i Ptilde_{d_i}( u[t - k_i] )        total degree p = sum_i d_i

Because the u[t] are independent and each factor with d >= 1 is
mean-zero and unit-variance, these products are orthonormal:
E[z_A z_B] = delta_AB. That orthonormality is what makes the per-basis-
function capacities additive, and it is what makes the Dambre bound
(total capacity <= number of linearly independent readout parameters)
apply.

The capacity of one basis function is the out-of-sample R^2 of the best
linear readout of the reservoir state onto it, clipped at zero:

    C_A = max( R^2( z_A , Wout^T x ) , 0 )

No `scipy.special` is used anywhere in this module: the Legendre
recurrence is written out explicitly (see `legendre_normalized`) so it
can be checked by hand, matching the rest of this repository.
"""

from __future__ import annotations

import itertools

import numpy as np

from esn import fit_ridge, predict
from metrics import r2


def legendre_normalized(u: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate the normalised Legendre polynomial Ptilde_degree at u.

    The (unnormalised) Legendre polynomials P_d are defined on [-1, 1] by
    the three-term recurrence

        P_0(u) = 1
        P_1(u) = u
        (d + 1) P_{d+1}(u) = (2d + 1) u P_d(u) - d P_{d-1}(u)

    and are orthogonal under u ~ Uniform[-1, 1]:

        E[ P_a(u) P_b(u) ] = delta_ab / (2a + 1)

    so the *normalised* family evaluated here is

        Ptilde_d(u) = sqrt(2d + 1) * P_d(u)      =>   E[Ptilde_a Ptilde_b] = delta_ab

    The recurrence is run explicitly (no `scipy.special`), keeping only
    the last two polynomials needed at each step.

    Parameters
    ----------
    u : ndarray of values expected in [-1, 1] (not range-checked here;
        this module is designed around u ~ Uniform[-1, 1] inputs, per
        `tasks.py`'s convention for `delayed_memory` and `delayed_product`).
    degree : non-negative integer polynomial degree.

    Returns
    -------
    ndarray, same shape as u, equal to Ptilde_degree(u). degree = 0
    returns an array of ones (Ptilde_0(u) = sqrt(1) * 1 = 1).

    Raises
    ------
    ValueError
        if degree is negative.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")

    u = np.asarray(u, dtype=float)
    p_prev2 = np.ones_like(u)  # P_0(u)
    if degree == 0:
        p_degree = p_prev2
    else:
        p_prev1 = u.copy()  # P_1(u)
        if degree == 1:
            p_degree = p_prev1
        else:
            for d in range(1, degree):
                # (d + 1) P_{d+1} = (2d + 1) u P_d - d P_{d-1}
                p_next = ((2 * d + 1) * u * p_prev1 - d * p_prev2) / (d + 1)
                p_prev2, p_prev1 = p_prev1, p_next
            p_degree = p_prev1

    norm = np.sqrt(2 * degree + 1)
    return norm * p_degree


def basis_function(u: np.ndarray, spec) -> np.ndarray:
    """Evaluate one orthonormal basis function z_A driven by input u.

    A basis function is indexed by a set A of (delay, degree) pairs with
    distinct delays and degree >= 1 for every pair:

        z_A[t] = prod_i Ptilde_{d_i}( u[t - k_i] )

    with total degree p = sum_i d_i. Because the u[t] are i.i.d. and each
    factor Ptilde_{d_i} (d_i >= 1) is mean-zero and unit-variance under
    Uniform[-1, 1], distinct basis functions are orthonormal:
    E[z_A z_B] = delta_AB.

    Entries where any t - k_i < 0 (the delayed value is not yet
    available) are filled with zero -- the same convention used by
    `tasks.delayed_memory`. Those entries fall inside the mandatory
    washout region of any caller that respects washout >= the largest
    delay used, so they are never fit or scored against.

    Parameters
    ----------
    u : (T,) ndarray, the driving input sequence.
    spec : sequence of (delay, degree) pairs, e.g. [(0, 2), (3, 1)] for
        Ptilde_2(u_t) * Ptilde_1(u_{t-3}). Delays must be distinct across
        the sequence and every degree must be >= 1 (a degree-0 factor
        contributes 1 everywhere and is meaningless as part of a basis
        function index, so it is rejected).

    Returns
    -------
    z : (T,) ndarray.

    Raises
    ------
    ValueError
        if spec is empty, any delay is negative, delays are not all
        distinct, or any degree is < 1.
    """
    spec = list(spec)
    if not spec:
        raise ValueError("spec must contain at least one (delay, degree) pair")

    delays = [k for k, _ in spec]
    if len(set(delays)) != len(delays):
        raise ValueError("delays in spec must be distinct")
    for k, d in spec:
        if k < 0:
            raise ValueError("delay must be non-negative")
        if d < 1:
            raise ValueError("degree must be >= 1 for every factor in spec")

    u = np.asarray(u, dtype=float)
    T = u.shape[0]
    z = np.ones(T, dtype=float)
    valid = np.ones(T, dtype=bool)
    for k, d in spec:
        shifted = np.zeros(T, dtype=float)
        if k == 0:
            shifted[:] = u
        else:
            shifted[k:] = u[:-k]
            valid[:k] = False
        z = z * legendre_normalized(shifted, d)

    z[~valid] = 0.0
    return z


def _compositions(total: int, parts: int):
    """Yield every tuple of `parts` positive integers summing to `total`."""
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def enumerate_specs(max_degree: int, max_delay: int):
    """Enumerate every basis-function spec up to a degree and delay bound.

    A spec is a tuple of (delay, degree) pairs, sorted by increasing
    delay, with:

    - distinct delays drawn from {0, ..., max_delay},
    - degree >= 1 for every pair,
    - total degree p = sum(degrees) satisfying 1 <= p <= max_degree.

    Combinatorial size: for m factors (1 <= m <= min(max_degree,
    max_delay + 1)), there are C(max_delay + 1, m) ways to choose which
    delays appear, and for each choice, sum_{p=m}^{max_degree} C(p - 1,
    m - 1) ways to distribute total degree p among the m chosen delays
    (compositions of p into m positive parts). The full count sums that
    product over m = 1 .. min(max_degree, max_delay + 1). This grows
    quickly -- e.g. max_degree=4, max_delay=20 already gives thousands of
    specs -- so callers (`capacity_spectrum`, and in particular the
    `fig_capacity_spectrum.py` experiment) must bound both parameters
    deliberately to keep runtime under control.

    Parameters
    ----------
    max_degree : largest total polynomial degree p to include (>= 1).
    max_delay : largest delay k to make available (delays 0..max_delay).

    Returns
    -------
    list of specs (tuples of (delay, degree) pairs), sorted
    deterministically by (total degree, largest delay in the spec, the
    spec tuple itself) so results are reproducible.

    Raises
    ------
    ValueError
        if max_degree < 1 or max_delay < 0.
    """
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    if max_delay < 0:
        raise ValueError("max_delay must be >= 0")

    delays = list(range(max_delay + 1))
    max_factors = min(max_degree, len(delays))

    specs = []
    for m in range(1, max_factors + 1):
        for delay_subset in itertools.combinations(delays, m):
            for total_p in range(m, max_degree + 1):
                for degrees in _compositions(total_p, m):
                    specs.append(tuple(zip(delay_subset, degrees)))

    specs.sort(
        key=lambda spec: (sum(d for _, d in spec), max(k for k, _ in spec), spec)
    )
    return specs


def capacity_of(
    states: np.ndarray,
    target: np.ndarray,
    washout: int,
    lam: float,
    test_frac: float = 0.5,
) -> float:
    """Out-of-sample capacity of the reservoir readout for one target.

    Fits a ridge-regression readout on a training slice of the
    post-washout data and scores R^2 out-of-sample on a held-out test
    slice -- the target is never fit and scored on the same rows. This
    mirrors the train/test split convention of `metrics.memory_capacity`
    exactly: the post-washout data is split once, in order, at
    `int(n * (1 - test_frac))`, with the first part used for training and
    the remainder for testing.

        C = max( R^2( target_test , Wout^T x_test ) , 0 )

    Parameters
    ----------
    states : (T, n) reservoir states.
    target : (T,) ndarray, the target sequence to reconstruct (e.g. one
        basis function z_A from `basis_function`). Must already be
        time-aligned with `states` (target[t] corresponds to states[t]).
    washout : number of initial steps to discard before splitting into
        train/test. Must be >= every delay used to build `target`, so
        that no zero-padded (unavailable-delay) entries leak past the
        washout into the scored data.
    lam : ridge regularization strength (see `esn.fit_ridge`).
    test_frac : fraction of the post-washout data held out for testing
        (default 0.5, i.e. a 50/50 train/test split).

    Returns
    -------
    capacity : float, max(R^2, 0) on the held-out test slice.
    """
    states_use = states[washout:]
    target_use = target[washout:]
    n = states_use.shape[0]
    split = int(n * (1.0 - test_frac))

    X_train, X_test = states_use[:split], states_use[split:]
    y_train, y_test = target_use[:split], target_use[split:]

    Wout = fit_ridge(X_train, y_train, lam, washout=0)
    yhat_test = predict(X_test, Wout)
    score = r2(y_test, yhat_test)
    return float(max(score, 0.0))


def capacity_spectrum(
    states: np.ndarray,
    u: np.ndarray,
    max_degree: int,
    max_delay: int,
    washout: int,
    lam: float,
    test_frac: float = 0.5,
):
    """Measure the full information processing capacity (IPC) spectrum.

    Enumerates every basis-function spec up to (max_degree, max_delay)
    via `enumerate_specs`, evaluates each one with `basis_function`
    (driven by `u`), and measures its capacity with `capacity_of` on the
    given reservoir `states`. Because distinct basis functions are
    orthonormal (see `basis_function`), their capacities are additive,
    and the Dambre et al. (2012) bound applies: the total is bounded by
    the number of linearly independent readout parameters (reservoir
    units + bias).

    Temporal depth convention: a spec's "temporal depth" k, used to
    aggregate `grid`, is defined here as the LARGEST delay appearing in
    the spec (max_i k_i). This is a modeling convention adopted for this
    module, not a theorem -- a spec such as [(0, 1), (5, 2)] genuinely
    needs history back to t-5, so it is bucketed at k=5 even though it
    also involves a k=0 factor.

    Parameters
    ----------
    states : (T, n) reservoir states.
    u : (T,) input sequence that drove the reservoir.
    max_degree : largest total polynomial degree to include (see
        `enumerate_specs`).
    max_delay : largest delay to make available (see `enumerate_specs`).
    washout : mandatory washout, forwarded to `capacity_of`; must be >=
        max_delay so that no zero-padded entries are ever scored.
    lam : ridge regularization strength, forwarded to `capacity_of`.
    test_frac : train/test split fraction, forwarded to `capacity_of`
        (default 0.5).

    Returns
    -------
    total : float, sum of all per-spec capacities.
    per_spec : list of (spec, capacity) pairs, one per spec returned by
        `enumerate_specs`, in that same order.
    grid : (max_degree + 1, max_delay + 1) ndarray, grid[p, k] = sum of
        capacities of every spec with total degree p and temporal depth
        k (as defined above). Row p = 0 is always zero (no spec has total
        degree 0); it is kept so `grid[p, k]` can be indexed directly by
        the polynomial order p without an off-by-one.

    Raises
    ------
    ValueError
        if washout < max_delay (a zero-padded entry could otherwise leak
        into the scored data).
    """
    if washout < max_delay:
        raise ValueError(
            "washout must be >= max_delay so that zero-padded delayed "
            "entries never leak past washout into the scored data"
        )

    specs = enumerate_specs(max_degree, max_delay)
    per_spec = []
    grid = np.zeros((max_degree + 1, max_delay + 1))

    for spec in specs:
        target = basis_function(u, spec)
        capacity = capacity_of(states, target, washout, lam, test_frac)
        per_spec.append((spec, capacity))

        p = sum(d for _, d in spec)
        k = max(k for k, _ in spec)
        grid[p, k] += capacity

    total = float(sum(c for _, c in per_spec))
    return total, per_spec, grid
