"""Uncertainty of classical estimators.

Two tools, chosen by the kind of estimator:

- Maximum-likelihood fits: standard errors from the observed information, the
  inverse Hessian of the negative log-likelihood at the optimum (numerical,
  central differences). A Hessian that is not positive definite means the data
  do not pin the parameters down; the errors are then reported as infinite
  rather than as a confident number.
- Moment estimators (no likelihood): a parametric bootstrap. Simulate many
  datasets of the same length from the fitted model, refit each, and report
  the spread of the refits (a standard error) and their mean shift from the
  truth used to simulate (the estimator's bias at this point).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray,
                      rel_step: float = 1e-4, abs_step: float = 1e-7) -> np.ndarray:
    """Central-difference Hessian of a scalar function at ``x``."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.maximum(np.abs(x) * rel_step, abs_step)
    hess = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        hess[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            value = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4.0 * h[i] * h[j])
            hess[i, j] = hess[j, i] = value
    return hess


def observed_information_std_errors(neg_log_likelihood: Callable[[np.ndarray], float],
                                    x_hat: np.ndarray) -> np.ndarray:
    """Standard errors of an MLE from the inverse observed information.

    Returns ``inf`` for every parameter when the Hessian is not positive
    definite (the likelihood is flat or saddle-shaped there).
    """
    hess = numerical_hessian(neg_log_likelihood, x_hat)
    hess = 0.5 * (hess + hess.T)
    try:
        np.linalg.cholesky(hess)
    except np.linalg.LinAlgError:
        return np.full(np.size(x_hat), np.inf)
    return np.sqrt(np.diag(np.linalg.inv(hess)))


@dataclass(frozen=True)
class BootstrapResult:
    """Parametric-bootstrap summary of an estimator, per parameter."""

    std_errors: dict[str, float]
    bias: dict[str, float]      # mean refit minus the truth it was simulated from
    n_boot: int                 # refits that succeeded


def parametric_bootstrap(
    estimate: dict[str, float],
    simulate: Callable[[np.random.Generator, int], np.ndarray],
    refit: Callable[[np.ndarray], dict[str, float]],
    n_boot: int = 200,
    seed: int = 0,
) -> BootstrapResult:
    """Spread and bias of an estimator at ``estimate``.

    Args:
        estimate: the fitted parameters (the truth the bootstrap simulates from).
        simulate: ``(rng, n_boot) -> (n_boot, n_prices)`` price paths drawn from
            the model at ``estimate``, each as long as the data.
        refit: the estimator applied to one price path.
        n_boot: number of simulated datasets.
        seed: seed of the generator handed to ``simulate``.
    """
    paths = simulate(np.random.default_rng(seed), n_boot)
    fits = []
    for path in paths:
        try:
            fits.append(refit(path))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if len(fits) < 2:
        raise ValueError("the bootstrap refits failed; no uncertainty estimate")
    names = list(estimate)
    values = np.array([[fit[k] for k in names] for fit in fits], dtype=float)
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    return BootstrapResult(
        std_errors={k: float(np.std(values[:, i], ddof=1)) for i, k in enumerate(names)},
        bias={k: float(np.mean(values[:, i]) - estimate[k]) for i, k in enumerate(names)},
        n_boot=int(values.shape[0]),
    )
