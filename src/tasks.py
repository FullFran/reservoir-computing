"""Benchmark tasks for testing reservoir computing systems.

Each task takes an input sequence `u` and returns a target sequence `y`
that a reservoir readout must learn to reconstruct from the reservoir
states driven by `u`.

Input distributions matter and are NOT interchangeable:

- `delayed_memory` and `delayed_product` are driven by u ~ Uniform[-1, 1].
- `narma10` REQUIRES u ~ Uniform[0, 0.5]; the recursion is only stable
  (in the "usually does not blow up" sense) for inputs in that range.
"""

from __future__ import annotations

import numpy as np


def delayed_memory(u: np.ndarray, k: int) -> np.ndarray:
    """Return the k-step-delayed copy of u: y_t = u_{t-k}.

    For t < k, u_{t-k} is not available; those entries are filled with
    zeros (the standard convention for memory-capacity tasks, since they
    fall inside the mandatory washout region anyway and are never fit or
    scored against).

    Parameters
    ----------
    u : (T,) input sequence.
    k : non-negative integer delay.

    Returns
    -------
    y : (T,) ndarray with y[t] = u[t - k] for t >= k, else 0.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    y = np.zeros_like(u)
    if k == 0:
        return u.copy()
    y[k:] = u[:-k]
    return y


def delayed_product(u: np.ndarray, k1: int = 1, k2: int = 5) -> np.ndarray:
    """Return y_t = u_{t-k1} * u_{t-k2}, a simple nonlinear memory task.

    Entries where either delayed value is unavailable (t < max(k1, k2))
    are filled with zeros, same convention as `delayed_memory`.

    Parameters
    ----------
    u : (T,) input sequence.
    k1, k2 : non-negative integer delays.

    Returns
    -------
    y : (T,) ndarray.
    """
    if k1 < 0 or k2 < 0:
        raise ValueError("k1 and k2 must be non-negative")
    u_k1 = delayed_memory(u, k1)
    u_k2 = delayed_memory(u, k2)
    y = u_k1 * u_k2
    m = max(k1, k2)
    y[:m] = 0.0
    return y


def narma10(u: np.ndarray) -> np.ndarray:
    """Compute the NARMA10 target sequence driven by input u.

    Recursion (defined for t >= 9, since it needs y_{t}, ..., y_{t-9} and
    u_{t-9}, u_t):

        y_{t+1} = 0.3 y_t + 0.05 y_t * sum_{i=0}^{9} y_{t-i}
                  + 1.5 u_{t-9} u_t + 0.1

    y_0, ..., y_9 are initialized to 0 (standard convention). The
    recursion is run for t = 9, ..., T-2, producing y_10, ..., y_{T-1}.

    NARMA10 REQUIRES u to be drawn from Uniform[0, 0.5]; using a different
    range (e.g. the Uniform[-1, 1] used for the memory tasks) very
    frequently makes the recursion diverge.

    Parameters
    ----------
    u : (T,) input sequence, expected to be drawn from Uniform[0, 0.5].

    Returns
    -------
    y : (T,) ndarray, the NARMA10 target sequence.

    Raises
    ------
    FloatingPointError
        if the recursion diverges (a non-finite value is produced). The
        caller is expected to catch this, resample u, and retry -- NARMA10
        is a known-divergent recursion and this must never be silently
        swallowed.
    """
    T = u.shape[0]
    y = np.zeros(T, dtype=float)
    for t in range(9, T - 1):
        window_sum = np.sum(y[t - 9 : t + 1])
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * window_sum
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
        if not np.isfinite(y[t + 1]):
            raise FloatingPointError(
                f"NARMA10 recursion diverged at t={t + 1} "
                f"(y={y[t + 1]}); resample the input and retry"
            )
    return y


def sample_narma10(T: int, seed: int, max_attempts: int = 50):
    """Draw a valid (non-diverging) NARMA10 (u, y) pair.

    Resamples u ~ Uniform[0, 0.5] and recomputes narma10(u) until the
    recursion does not diverge, up to `max_attempts` times. Every retry is
    reported via the returned `attempts` count so callers can log it
    instead of silently hiding divergence.

    Parameters
    ----------
    T : sequence length.
    seed : base seed; attempt i uses seed + i so retries are still
        reproducible.
    max_attempts : maximum number of resample attempts before giving up.

    Returns
    -------
    u : (T,) ndarray, Uniform[0, 0.5] input.
    y : (T,) ndarray, NARMA10 target.
    attempts : int, number of attempts used (1 means it worked on the
        first try).
    """
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        u = rng.uniform(0.0, 0.5, size=T)
        try:
            y = narma10(u)
            return u, y, attempt + 1
        except FloatingPointError:
            continue
    raise RuntimeError(
        f"NARMA10 failed to produce a stable sequence in {max_attempts} attempts"
    )
