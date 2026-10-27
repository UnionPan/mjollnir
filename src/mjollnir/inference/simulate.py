"""Training data for amortized inference: parameter draws and the histories they produce.

Histories come from the same worlds the hedging lab runs (mjollnir.worlds),
vmapped over parameter draws, so the network learns exactly the dynamics the
lab simulates. Each history starts in its model's stationary regime, is
``max_len`` days long, and keeps a random leading stretch of 1 to
``max_len / 252`` years as the data (the rest masked), so the posterior learns
how its width depends on the length of the window.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from mjollnir.inference.models import InferenceModel
from mjollnir.inference.summaries import summary_features
from mjollnir.worlds import world_for

__all__ = ["MIN_DAYS", "replicate_summaries", "simulate_batch"]

DT = 1.0 / 252
MIN_DAYS = 250


def _returns(model: InferenceModel, u, keys, max_len: int):
    """(n, max_len) daily log returns of the histories of unconstrained draws ``u``."""
    params = model.world(model.to_natural(u))
    world = world_for(model.name)

    def history(p, k):
        state, path, k = world.init(p, k, max_len, DT)

        def step(carry, t):
            spot, state, k = carry
            new_spot, state, k = world.step(spot, state, t, p, path, DT, 0.0, k)
            return (new_spot, state, k), jnp.log(new_spot / spot)

        _, returns = jax.lax.scan(step, (jnp.asarray(1.0, u.dtype), jnp.asarray(state, u.dtype), k),
                                  jnp.arange(max_len))
        return returns

    return jax.vmap(history)(params, keys)


@partial(jax.jit, static_argnames=("model", "n", "max_len"))
def _simulate(model: InferenceModel, key, n: int, max_len: int):
    key_prior, key_len, key_paths = jax.random.split(key, 3)
    u = model.sample_prior(key_prior, n)
    returns = _returns(model, u, jax.random.split(key_paths, n), max_len)
    lengths = jax.random.randint(key_len, (n,), MIN_DAYS, max_len + 1)
    mask = (jnp.arange(max_len)[None, :] < lengths[:, None]).astype(returns.dtype)
    return u, summary_features(returns * mask, mask)


def simulate_batch(model: InferenceModel, key, n: int, max_len: int = 1260):
    """``n`` prior draws (unconstrained) and the summaries of the histories they produce."""
    return _simulate(model, key, n, max_len)


@partial(jax.jit, static_argnames=("model", "max_len", "chunk"))
def _replicate(model: InferenceModel, key, u, length, max_len: int, chunk: int):
    keys = jax.random.split(key, u.shape[0])

    def one_chunk(args):
        u_c, k_c = args
        returns = _returns(model, u_c, k_c, max_len)
        mask = jnp.broadcast_to(jnp.arange(max_len) < length, returns.shape).astype(returns.dtype)
        return summary_features(returns * mask, mask)

    n_chunks = u.shape[0] // chunk
    features = jax.lax.map(one_chunk, (u.reshape(n_chunks, chunk, -1), keys.reshape(n_chunks, chunk, -1)))
    return features.reshape(u.shape[0], -1)


def replicate_summaries(model: InferenceModel, key, u, length: int, max_len: int = 1260, chunk: int = 16):
    """Summaries of one history of ``length`` days per draw in ``u`` (posterior
    predictive replicates). Draws are simulated ``chunk`` at a time, which
    bounds memory (rough Bergomi's kernel is max_len^2 per draw)."""
    if u.shape[0] % chunk:
        raise ValueError(f"the number of draws must be a multiple of chunk = {chunk}")
    return _replicate(model, key, jnp.asarray(u), jnp.asarray(length), max_len, chunk)
