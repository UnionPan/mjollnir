"""The posterior network: a conditional mixture density network.

Maps a history's standardized summaries to a mixture of diagonal Gaussians
over the model's standardized unconstrained parameters. Mixtures capture what
daily data leave ambiguous (frequent small jumps against rare large ones) and
correlations between parameters, with exact densities and cheap sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

__all__ = ["MixtureDensityNetwork", "mixture_log_prob", "sample_mixture"]

LOG_SCALE_RANGE = (-7.0, 3.0)


class MixtureDensityNetwork(nn.Module):
    hidden: tuple[int, ...] = (256, 256, 256)
    n_components: int = 10
    n_outputs: int = 4

    @nn.compact
    def __call__(self, s):
        x = s
        for width in self.hidden:
            x = nn.gelu(nn.Dense(width)(x))
        logits = nn.Dense(self.n_components)(x)
        means = nn.Dense(self.n_components * self.n_outputs)(x).reshape(-1, self.n_components, self.n_outputs)
        log_scales = nn.Dense(self.n_components * self.n_outputs)(x).reshape(-1, self.n_components, self.n_outputs)
        return logits, means, jnp.clip(log_scales, *LOG_SCALE_RANGE)


def mixture_log_prob(logits, means, log_scales, z):
    """log p(z | s) under the mixture; z is (B, d)."""
    log_weights = jax.nn.log_softmax(logits, axis=-1)
    diff = (z[:, None, :] - means) / jnp.exp(log_scales)
    component = -0.5 * jnp.sum(diff**2 + 2.0 * log_scales + jnp.log(2.0 * jnp.pi), axis=-1)
    return jax.nn.logsumexp(log_weights + component, axis=-1)


def sample_mixture(key, logits, means, log_scales, n: int):
    """(B, n, d) draws from each row's mixture."""
    k_comp, k_noise = jax.random.split(key)
    batch, _, dim = means.shape
    comp = jax.random.categorical(k_comp, logits[:, None, :], axis=-1, shape=(batch, n))
    chosen_means = jnp.take_along_axis(means, comp[..., None], axis=1)
    chosen_scales = jnp.exp(jnp.take_along_axis(log_scales, comp[..., None], axis=1))
    return chosen_means + chosen_scales * jax.random.normal(k_noise, (batch, n, dim))
