"""What amortized inference estimates for each model, and the broad prior it trains under.

Each :class:`InferenceModel` names the model's *dynamics* parameters, maps
them to unconstrained coordinates (where the prior is a box and the posterior
network works), and turns a batch of them into the world parameters
:mod:`mjollnir.worlds` simulates. Today's state (Heston's v0, rough Bergomi's
xi0, GARCH's h0, the regime) is not inferred: training histories start in
the model's stationary regime, and a fit reads the state off the data's end.

Training priors are deliberately broad boxes in unconstrained coordinates (a
log-uniform volatility, a uniform correlation, ...): the network learns the
likelihood's shape over everything plausible, and narrower priors (a sector's)
enter later by importance reweighting, without retraining.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["MODELS", "InferenceModel"]

DT = 1.0 / 252


def _logit(p):
    return jnp.log(p) - jnp.log1p(-p)


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


@dataclass(frozen=True)
class InferenceModel:
    """A model as amortized inference sees it.

    Attributes:
        name: the world's model name (mjollnir.worlds).
        params: names of the natural parameters, in order.
        low, high: the training prior's box in unconstrained coordinates.
        to_natural: (n, d) unconstrained -> (n, d) natural.
        to_unconstrained: inverse of ``to_natural``.
        world: (n, d) natural parameters -> dict of (n,) world parameters,
            including the stationary initial state the history starts in.
    """

    name: str
    params: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    to_natural: Callable
    to_unconstrained: Callable
    world: Callable

    @property
    def dim(self) -> int:
        return len(self.params)

    def sample_prior(self, key, n: int):
        """(n, d) unconstrained draws from the training prior (uniform box)."""
        low, high = jnp.asarray(self.low), jnp.asarray(self.high)
        return low + jax.random.uniform(key, (n, self.dim)) * (high - low)

    def in_prior(self, unconstrained) -> np.ndarray:
        """Which rows lie inside the training prior's box."""
        u = np.asarray(unconstrained)
        return np.all((u >= np.asarray(self.low)) & (u <= np.asarray(self.high)), axis=-1)


# ── Heston: (kappa, theta, sigma_v, rho) ────────────────────────────────────

def _heston_natural(u):
    return jnp.stack([jnp.exp(u[:, 0]), jnp.exp(u[:, 1]), jnp.exp(u[:, 2]), 0.99 * jnp.tanh(u[:, 3])], axis=1)


def _heston_unconstrained(x):
    return jnp.stack([jnp.log(x[:, 0]), jnp.log(x[:, 1]), jnp.log(x[:, 2]), jnp.arctanh(x[:, 3] / 0.99)], axis=1)


def _heston_world(x):
    kappa, theta, sigma_v, rho = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return {"v0": theta, "kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho}


# ── Merton: (sigma, lambda_j, mu_j, sigma_j) ────────────────────────────────

def _merton_natural(u):
    return jnp.stack([jnp.exp(u[:, 0]), jnp.exp(u[:, 1]), u[:, 2], jnp.exp(u[:, 3])], axis=1)


def _merton_unconstrained(x):
    return jnp.stack([jnp.log(x[:, 0]), jnp.log(x[:, 1]), x[:, 2], jnp.log(x[:, 3])], axis=1)


def _merton_world(x):
    return {"sigma": x[:, 0], "lambda_j": x[:, 1], "mu_j": x[:, 2], "sigma_j": x[:, 3]}


# ── Two-regime Markov-switching volatility: (sigma_0, sigma_1, p_00, p_11) ────
# sigma_1 > sigma_0 by construction (no label switching): sigma_1 = sigma_0 (1 + e^u)

def _regime_natural(u):
    sigma_0 = jnp.exp(u[:, 0])
    return jnp.stack([sigma_0, sigma_0 * (1.0 + jnp.exp(u[:, 1])), _sigmoid(u[:, 2]), _sigmoid(u[:, 3])], axis=1)


def _regime_unconstrained(x):
    return jnp.stack([jnp.log(x[:, 0]), jnp.log(x[:, 1] / x[:, 0] - 1.0), _logit(x[:, 2]), _logit(x[:, 3])], axis=1)


def _regime_world(x):
    """The arrays mjollnir.worlds reads (see worlds.market_params), per draw."""
    p00, p11 = x[:, 2], x[:, 3]
    transition = jnp.stack([jnp.stack([p00, 1.0 - p00], axis=1), jnp.stack([1.0 - p11, p11], axis=1)], axis=1)
    # start in the stationary distribution's likelier regime
    calm_share = (1.0 - p11) / ((1.0 - p00) + (1.0 - p11))
    return {"sigmas": x[:, :2], "transition": transition, "regime0": jnp.where(calm_share >= 0.5, 0.0, 1.0)}


# ── GARCH(1,1) with Student-t innovations: (long-run vol, persistence, alpha share, nu) ──
# alpha = share * persistence, beta = (1 - share) * persistence, omega from the long-run variance.
# Gaussian GARCH can only produce a stock's earnings gaps through near-explosive
# variance dynamics, so the innovations carry their own tails: 1/nu = MAX_INV_NU * sigmoid(u).

MAX_INV_NU = 0.4  # nu > 2.5


def _garch_natural(u):
    return jnp.stack([jnp.exp(u[:, 0]), _sigmoid(u[:, 1]), _sigmoid(u[:, 2]),
                      1.0 / (MAX_INV_NU * _sigmoid(u[:, 3]))], axis=1)


def _garch_unconstrained(x):
    return jnp.stack([jnp.log(x[:, 0]), _logit(x[:, 1]), _logit(x[:, 2]), _logit(1.0 / (MAX_INV_NU * x[:, 3]))],
                     axis=1)


def _garch_world(x):
    long_run_var = x[:, 0] ** 2 * DT
    persistence, share = x[:, 1], x[:, 2]
    return {"omega": long_run_var * (1.0 - persistence), "alpha": share * persistence,
            "beta": (1.0 - share) * persistence, "h0": long_run_var, "inv_nu": 1.0 / x[:, 3]}


# ── Rough Bergomi: (H, eta, rho, xi0) ───────────────────────────────────────
# E[v_t] = xi0 at every t, so xi0 is the forward-variance level of the window;
# the lab still starts from today's realized variance.

def _rough_natural(u):
    return jnp.stack([0.02 + 0.48 * _sigmoid(u[:, 0]), jnp.exp(u[:, 1]), 0.99 * jnp.tanh(u[:, 2]),
                      jnp.exp(u[:, 3])], axis=1)


def _rough_unconstrained(x):
    return jnp.stack([_logit((x[:, 0] - 0.02) / 0.48), jnp.log(x[:, 1]), jnp.arctanh(x[:, 2] / 0.99),
                      jnp.log(x[:, 3])], axis=1)


def _rough_world(x):
    return {"H": x[:, 0], "eta": x[:, 1], "rho": x[:, 2], "xi0": x[:, 3]}


_LOG = np.log
MODELS: dict[str, InferenceModel] = {
    "heston": InferenceModel(
        "heston", ("kappa", "theta", "sigma_v", "rho"),
        low=(_LOG(0.2), _LOG(0.005), _LOG(0.05), float(np.arctanh(-0.95 / 0.99))),
        high=(_LOG(20.0), _LOG(0.5), _LOG(3.0), float(np.arctanh(0.5 / 0.99))),
        to_natural=_heston_natural, to_unconstrained=_heston_unconstrained, world=_heston_world),
    "merton": InferenceModel(
        "merton", ("sigma", "lambda_j", "mu_j", "sigma_j"),
        low=(_LOG(0.05), _LOG(0.5), -0.15, _LOG(0.005)),
        high=(_LOG(0.8), _LOG(200.0), 0.05, _LOG(0.3)),
        to_natural=_merton_natural, to_unconstrained=_merton_unconstrained, world=_merton_world),
    "regime_switching_gbm": InferenceModel(
        "regime_switching_gbm", ("sigma_0", "sigma_1", "p_00", "p_11"),
        low=(_LOG(0.05), _LOG(0.1), float(np.log(0.9 / 0.1)), float(np.log(0.9 / 0.1))),
        high=(_LOG(0.6), _LOG(5.0), float(np.log(0.999 / 0.001)), float(np.log(0.999 / 0.001))),
        to_natural=_regime_natural, to_unconstrained=_regime_unconstrained, world=_regime_world),
    "garch": InferenceModel(
        "garch", ("long_run_vol", "persistence", "alpha_share", "nu"),
        low=(_LOG(0.05), float(np.log(0.8 / 0.2)), float(np.log(0.01 / 0.99)), -4.0),   # nu up to 140
        high=(_LOG(0.8), float(np.log(0.999 / 0.001)), float(np.log(0.5 / 0.5)), 4.0),  # nu down to 2.5
        to_natural=_garch_natural, to_unconstrained=_garch_unconstrained, world=_garch_world),
    "rough_bergomi": InferenceModel(
        "rough_bergomi", ("H", "eta", "rho", "xi0"),
        low=(-4.0, _LOG(0.3), float(np.arctanh(-0.95 / 0.99)), _LOG(0.005)),
        high=(4.0, _LOG(4.0), float(np.arctanh(0.3 / 0.99)), _LOG(0.5)),
        to_natural=_rough_natural, to_unconstrained=_rough_unconstrained, world=_rough_world),
}
