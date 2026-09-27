"""Gaussian hidden semi-Markov model with explicit Poisson state durations.

A standard HMM implies geometric dwell times, whose mode is the shortest possible
bout. An HSMM instead attaches an explicit duration distribution to each state, so
the characteristic timescale of a behaviour becomes a model parameter rather than a
by-product of the self-transition probability.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from loguru import logger
from scipy import stats


def _run_lengths(states: np.ndarray) -> Dict[int, List[int]]:
    """Return the observed run lengths for each state."""
    runs: Dict[int, List[int]] = {}
    if states.size == 0:
        return runs

    boundaries = np.concatenate(
        ([0], np.where(np.diff(states) != 0)[0] + 1, [states.size])
    )
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        runs.setdefault(int(states[start]), []).append(int(end - start))
    return runs


def _duration_penalty(lambdas: np.ndarray, max_duration: int) -> np.ndarray:
    """Log-probability of each duration under each state's Poisson model."""
    durations = np.arange(1, max_duration + 1)
    with np.errstate(divide="ignore"):
        return np.array(
            [stats.poisson.logpmf(durations, lam) for lam in lambdas]
        )


def fit_gaussian_hsmm(
    x: np.ndarray,
    n_states: int = 4,
    n_iter: int = 30,
    max_duration: int = 96,
    random_state: int = 42,
) -> Dict:
    """
    Fit a Gaussian HSMM with Poisson durations by segmental EM.

    The E-step decodes the most likely state sequence with a duration-aware Viterbi
    recursion; the M-step re-estimates the Gaussian emission parameters from the
    assigned observations and the Poisson rates from the resulting run lengths.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional observation sequence.
    n_states : int, default=4
        Number of hidden states.
    n_iter : int, default=30
        Number of EM iterations.
    max_duration : int, default=96
        Longest bout considered by the decoder, in time steps.
    random_state : int, default=42

    Returns
    -------
    dict
        Fitted means, standard deviations, Poisson rates, decoded state sequence
        and the log-likelihood trace.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        raise ValueError("Empty observation sequence")

    rng = np.random.default_rng(random_state)

    # Initialise emissions on quantiles so states start ordered by activity level.
    quantiles = np.linspace(0, 100, n_states + 2)[1:-1]
    means = np.percentile(x, quantiles)
    sds = np.full(n_states, max(x.std(), 1e-3))
    lambdas = np.full(n_states, 4.0)

    loglik_trace: List[float] = []
    states = np.zeros(n, dtype=int)

    for iteration in range(n_iter):
        # --- E-step: duration-aware Viterbi over (state, duration) segments ---
        emission_ll = np.array(
            [stats.norm.logpdf(x, loc=m, scale=max(s, 1e-6)) for m, s in zip(means, sds)]
        )
        cumulative = np.concatenate(
            [np.zeros((n_states, 1)), np.cumsum(emission_ll, axis=1)], axis=1
        )
        duration_ll = _duration_penalty(lambdas, max_duration)

        score = np.full((n_states, n + 1), -np.inf)
        back_state = np.zeros((n_states, n + 1), dtype=int)
        back_dur = np.zeros((n_states, n + 1), dtype=int)
        score[:, 0] = 0.0

        for t in range(1, n + 1):
            max_d = min(max_duration, t)
            durations = np.arange(1, max_d + 1)
            starts = t - durations

            for k in range(n_states):
                # Total emission log-likelihood of the segment [start, t).
                seg = cumulative[k, t] - cumulative[k, starts]
                cand = seg + duration_ll[k, durations - 1]

                prev = score[:, starts]
                if n_states > 1:
                    # A segment must be preceded by a different state.
                    prev = prev.copy()
                    prev[k, :] = -np.inf
                    best_prev = prev.max(axis=0)
                    best_prev_state = prev.argmax(axis=0)
                else:
                    best_prev = prev[0]
                    best_prev_state = np.zeros_like(durations)

                # The first segment has no predecessor.
                first = starts == 0
                best_prev = np.where(first, 0.0, best_prev)

                total = cand + best_prev
                idx = int(np.argmax(total))
                if total[idx] > score[k, t]:
                    score[k, t] = total[idx]
                    back_state[k, t] = best_prev_state[idx]
                    back_dur[k, t] = durations[idx]

        # Backtrace the optimal segmentation.
        t = n
        k = int(np.argmax(score[:, n]))
        loglik = float(score[k, n])
        new_states = np.zeros(n, dtype=int)
        while t > 0:
            d = int(back_dur[k, t])
            if d <= 0:
                break
            new_states[t - d : t] = k
            k_prev = int(back_state[k, t])
            t -= d
            k = k_prev

        states = new_states
        loglik_trace.append(loglik)

        # --- M-step ---
        for k in range(n_states):
            mask = states == k
            if mask.sum() > 1:
                means[k] = x[mask].mean()
                sds[k] = max(x[mask].std(), 1e-3)
            elif mask.sum() == 0:
                # Re-seed an empty state to keep the model at n_states.
                means[k] = float(rng.choice(x))

        runs = _run_lengths(states)
        for k in range(n_states):
            if runs.get(k):
                lambdas[k] = max(float(np.mean(runs[k])), 1e-3)

        if len(loglik_trace) > 1 and abs(loglik_trace[-1] - loglik_trace[-2]) < 1e-6:
            logger.debug(f"HSMM converged after {iteration + 1} iterations")
            break

    order = np.argsort(means)
    remap = {int(old): new for new, old in enumerate(order)}

    return {
        "means": means[order],
        "sds": sds[order],
        "lambdas": lambdas[order],
        "states": np.array([remap[int(s)] for s in states]),
        "loglik": loglik_trace,
        "n_states": n_states,
    }


def state_sequence_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of time steps on which two decoded sequences agree."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float((a[:n] == b[:n]).mean())


def mean_bout_duration(states: np.ndarray, bin_hours: float = 0.25) -> float:
    """Mean bout duration of a decoded sequence, in hours."""
    runs = _run_lengths(np.asarray(states))
    lengths = [length for state_runs in runs.values() for length in state_runs]
    return float(np.mean(lengths) * bin_hours) if lengths else float("nan")
