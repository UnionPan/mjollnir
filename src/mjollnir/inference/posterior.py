"""The posterior of a real history, under the training prior or a narrower one.

``fit`` turns a daily price history into posterior draws in one forward pass
of the trained network: its cost does not depend on the data beyond the
summaries, which is what makes it safe to serve publicly.

A narrower prior (a sector's, from empirical Bayes over a universe) enters by
importance reweighting: the network's posterior is under the training prior,
which is flat over its box in unconstrained coordinates, so a Gaussian prior
reweights each draw by its density (draws outside the box get weight 0). The
effective sample size says how much of the draw set the reweighting kept.

Contraction is measured against the prior in use: 1 - posterior variance /
prior variance per unconstrained coordinate, so under a sector prior it says
what the history added to what the sector already said.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from mjollnir.inference.artifact import NPEArtifact
from mjollnir.inference.summaries import pad_returns, summary_features
from mjollnir.inference.train import posterior_draws

__all__ = ["GaussianPrior", "Posterior", "fit"]

MAX_LEN = 1260


@dataclass(frozen=True)
class GaussianPrior:
    """A Gaussian prior over unconstrained parameters (e.g. a sector's)."""

    mean: np.ndarray
    cov: np.ndarray
    label: str = "prior"

    def log_density(self, u: np.ndarray) -> np.ndarray:
        diff = u - self.mean
        solved = np.linalg.solve(self.cov, diff.T).T
        return -0.5 * np.sum(diff * solved, axis=1)


@dataclass
class Posterior:
    """Posterior summaries in natural units, with what produced them."""

    model: str
    names: tuple[str, ...]
    mean: dict[str, float]
    sd: dict[str, float]
    q05: dict[str, float]
    q95: dict[str, float]
    contraction: dict[str, float]           # 1 - posterior var / prior var (unconstrained)
    effective_samples: float                # after reweighting; = draws inside the box without one
    in_prior_share: float                   # share of the network's draws inside the training box
    prior: str
    n_days: int
    draws: np.ndarray = field(repr=False)   # (n, d) natural, equally weighted after resampling


def fit(artifact: NPEArtifact, prices, *, n_draws: int = 4096, seed: int = 0,
        prior: GaussianPrior | None = None) -> Posterior:
    """Posterior of the model's dynamics given daily closes (the last five years are read)."""
    model = artifact.model
    returns = np.diff(np.log(np.asarray(prices, dtype=float)))
    padded, mask = pad_returns([returns], MAX_LEN)
    features = np.asarray(summary_features(jnp.asarray(padded), jnp.asarray(mask)))
    if not np.all(np.isfinite(features)):
        raise ValueError("the history's summaries are not finite (too short or constant?)")
    u = posterior_draws(artifact, features, jax.random.PRNGKey(seed), n_draws)[0]

    inside = model.in_prior(u)
    log_w = np.where(inside, 0.0, -np.inf)
    if prior is not None:
        log_w = log_w + prior.log_density(u)
    if not np.any(np.isfinite(log_w)):
        raise ValueError("no posterior draw lies inside the training prior")
    w = np.exp(log_w - np.max(log_w[np.isfinite(log_w)]))
    w /= w.sum()
    ess = float(1.0 / np.sum(w**2))
    rng = np.random.default_rng(seed)
    u_eq = u[rng.choice(len(u), size=len(u), p=w)]  # resampled: equally weighted
    x = np.asarray(model.to_natural(jnp.asarray(u_eq)))

    prior_var = (np.asarray(model.high) - np.asarray(model.low)) ** 2 / 12.0
    if prior is not None:  # a Gaussian truncated to the box is no wider than either
        prior_var = np.minimum(prior_var, np.diag(prior.cov))
    names = model.params
    return Posterior(
        model=model.name, names=names,
        mean=dict(zip(names, x.mean(0).tolist(), strict=True)),
        sd=dict(zip(names, x.std(0).tolist(), strict=True)),
        q05=dict(zip(names, np.quantile(x, 0.05, axis=0).tolist(), strict=True)),
        q95=dict(zip(names, np.quantile(x, 0.95, axis=0).tolist(), strict=True)),
        contraction=dict(zip(names, (1.0 - u_eq.var(0) / prior_var).tolist(), strict=True)),
        effective_samples=ess, in_prior_share=float(inside.mean()),
        prior=prior.label if prior is not None else "training prior",
        n_days=int(min(returns.size, MAX_LEN)), draws=x,
    )
