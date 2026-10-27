"""Posterior predictive checks: does the fitted model reproduce the history it was fitted to?

A posterior says which parameters fit a history *if the model is right*; it
cannot say whether the model is. The check simulates replicate histories of
the same length from posterior draws and places each of the observed
history's summaries among the replicates'. A summary the model cannot
reproduce (the history's kurtosis above every replicate's, say) is a
specific, readable misfit: what this model leaves out of this stock.

The predictive p-value of a summary is the share of replicates below the
observed value (ties count half): near 0 or 1, the model rarely produces a
history like this one in that respect.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mjollnir.inference.artifact import NPEArtifact
from mjollnir.inference.posterior import MAX_LEN, Posterior
from mjollnir.inference.simulate import replicate_summaries
from mjollnir.inference.summaries import FEATURE_NAMES, pad_returns, summary_features

__all__ = ["PredictiveCheck", "predictive_check", "predictive_p_values"]

# summaries the check leaves out: the history's length is not a property of the model
NOT_CHECKED = {"log_n"}
EXTREME = 0.025  # outside the replicates' central 95%


@dataclass
class PredictiveCheck:
    """Each checked summary's observed value, its replicates' 5/50/95% quantiles, and its p-value."""

    names: tuple[str, ...]
    observed: np.ndarray
    replicate_quantiles: np.ndarray  # (3, F)
    p_values: np.ndarray
    n_replicates: int

    @property
    def misfits(self) -> list[str]:
        """Summaries outside the replicates' central 95%, worst first."""
        tail = np.minimum(self.p_values, 1.0 - self.p_values)
        order = np.argsort(tail)
        return [self.names[i] for i in order if tail[i] < EXTREME]


def predictive_p_values(observed: np.ndarray, replicates: np.ndarray) -> np.ndarray:
    """(F,) share of replicates (R, F) below the observed (F,) summaries, ties counting half."""
    below = np.mean(replicates < observed, axis=0)
    ties = np.mean(replicates == observed, axis=0)
    return below + 0.5 * ties


def predictive_check(artifact: NPEArtifact, prices, posterior: Posterior, *, n_replicates: int = 64,
                     seed: int = 0) -> PredictiveCheck:
    """Replicate the history ``n_replicates`` times from ``posterior``'s draws and compare summaries."""
    model = artifact.model
    returns = np.diff(np.log(np.asarray(prices, dtype=float)))[-MAX_LEN:]
    padded, mask = pad_returns([returns], MAX_LEN)
    observed = np.asarray(summary_features(jnp.asarray(padded), jnp.asarray(mask)))[0]
    rng = np.random.default_rng(seed)
    natural = posterior.draws[rng.choice(len(posterior.draws), size=n_replicates, replace=False)]
    u = model.to_unconstrained(jnp.asarray(natural))
    replicates = np.asarray(replicate_summaries(model, jax.random.PRNGKey(seed), u, returns.size))
    keep = np.array([n not in NOT_CHECKED for n in FEATURE_NAMES])
    finite = np.all(np.isfinite(replicates), axis=1)
    replicates = replicates[finite][:, keep]
    return PredictiveCheck(
        names=tuple(n for n in FEATURE_NAMES if n not in NOT_CHECKED),
        observed=observed[keep],
        replicate_quantiles=np.quantile(replicates, [0.05, 0.5, 0.95], axis=0),
        p_values=predictive_p_values(observed[keep], replicates),
        n_replicates=int(finite.sum()),
    )
