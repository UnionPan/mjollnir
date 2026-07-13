"""The multi-agent market contract: ``MarketState`` + ``make_market_step``.

The extension point for multi-agent simulation is functional composition,
not class inheritance: a :class:`MarketState` PyTree threaded through a pure
step function that fuses **impact** (how aggregate order flow moves the mid)
with **dynamics** (how the market evolves on its own). Everything inside the
market boundary is library-owned and versioned; everything inside an agent's
head (policies, observations, rewards) is user code.

Canonical usage::

    from mjollnir.jax import make_market_step, MarketState, sqrt_impact

    step = make_market_step(impact=lambda s, q: sqrt_impact(s, q, adv=1e6, sigma=0.2))

    def episode_body(carry, _):
        state, agent_carry = carry
        actions = jax.vmap(policy)(agent_params, observe(state))   # your research
        state = step(state, actions.sum(axis=0))                    # library physics
        ...

``step`` is jittable, vmappable over batched states, and differentiable —
gradients flow through both the impact function and the QE dynamics, which
is what couples agents to each other in training.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mjollnir.processes._jax_qe import qe_heston_step

__all__ = ["MarketState", "make_market_step"]


class MarketState(NamedTuple):
    """Market-side state threaded through a rollout (a JAX PyTree).

    ``spot``/``variance`` may be scalars or path-batched arrays; ``key`` is
    the PRNG key consumed (and re-split) each step; ``t`` is the step index.
    Agent-side state (inventories, cash, beliefs) deliberately does NOT live
    here — it belongs to the user's carry.
    """

    spot: jax.Array
    variance: jax.Array
    key: jax.Array
    t: jax.Array

    @classmethod
    def create(cls, spot, variance, key) -> MarketState:
        return cls(
            spot=jnp.asarray(spot),
            variance=jnp.asarray(variance),
            key=key,
            t=jnp.asarray(0),
        )


def make_market_step(
    *,
    dt: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    impact=None,
    dynamics=qe_heston_step,
):
    """Build a pure ``(MarketState, net_flow) -> MarketState`` transition.

    Fuses two library-owned pieces into one jittable function:

    1. ``impact(spot, net_flow) -> spot`` — how this step's aggregate order
       flow moves the mid before diffusion (default: no impact, so the
       single-agent case degrades gracefully);
    2. ``dynamics`` — a QE-family step kernel ``(spot, var, dt, mu, kappa,
       theta, sigma_v, rho, key) -> (spot, var, key)``; defaults to Heston.

    Heston parameters are closed over (they are frozen for an experiment —
    typically ``**ParamSet.as_kwargs()``); order flow is per-step data.
    """

    def market_step(state: MarketState, net_flow) -> MarketState:
        spot = state.spot
        if impact is not None:
            spot = impact(spot, net_flow)
        spot, variance, key = dynamics(
            spot, state.variance, dt, mu, kappa, theta, sigma_v, rho,
            key=state.key,
        )
        return MarketState(spot=spot, variance=variance, key=key, t=state.t + 1)

    return market_step
