"""Differentiable market-impact functions (the coupling for multi-agent sims).

Impact is the channel through which agents' actions feed back into the price
every other agent sees — in multi-agent training it sits inside the
differentiable rollout graph and directly shapes equilibria. That is why it
lives *inside* the versioned kernel (same argument as the synthetic-data
layer): pin a mjollnir version and the coupling physics is frozen.

All functions are pure, jit/vmap/grad-safe, and share one signature shape::

    new_spot = impact_fn(spot, net_flow, adv, ...)

with ``net_flow`` the signed aggregate order flow over the step (shares,
buys positive) and ``adv`` the average daily volume in the same units.
``net_flow = 0`` is always the identity. Temporary-vs-permanent splits are
composed by the caller (see :func:`almgren_chriss_step`).

References: Kyle (1985); Almgren & Chriss (2000); the square-root law as
surveyed in Bouchaud et al., *Trades, Quotes and Prices* (2018).
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "almgren_chriss_step",
    "linear_impact",
    "sqrt_impact",
]


def linear_impact(spot, net_flow, adv, lam: float = 0.1):
    """Kyle-style linear impact: price moves proportionally to participation.

    ``spot * (1 + lam * net_flow / adv)`` — ``lam`` is the price move (as a
    fraction) caused by trading one full day's volume. The classic
    information-based benchmark; linear means round trips are free, so
    prefer :func:`sqrt_impact` when that matters.
    """
    participation = net_flow / adv
    return spot * (1.0 + lam * participation)


def sqrt_impact(spot, net_flow, adv, sigma, y: float = 1.0):
    """Square-root law: the empirical standard for metaorder impact.

    ``spot * (1 + sign(Q) * y * sigma * sqrt(|Q| / adv))`` with ``sigma`` the
    daily volatility. Concave: doubling the order size costs ~sqrt(2) more
    per share. ``y`` (the "Y-ratio") is empirically O(1) across markets.
    The gradient at ``net_flow = 0`` is unbounded (sqrt kink) — for
    optimization through the impact itself, smooth locally or use
    :func:`linear_impact` near zero.
    """
    participation = jnp.abs(net_flow) / adv
    move = jnp.sign(net_flow) * y * sigma * jnp.sqrt(participation)
    return spot * (1.0 + move)


def almgren_chriss_step(spot, net_flow, adv, sigma,
                        eta: float = 1.0, gamma: float = 0.1):
    """One step of the Almgren-Chriss temporary/permanent decomposition.

    Returns ``(exec_price, new_spot)``:

    * ``exec_price`` — what the trader pays: spot plus the **temporary**
      impact ``eta * sigma * (net_flow / adv)`` (vanishes next step),
    * ``new_spot`` — what the market keeps: the **permanent** component
      ``gamma * sigma * (net_flow / adv)`` impressed on the mid.

    Linear in flow on both legs (the original AC specification); use
    :func:`sqrt_impact` for the concave empirical variant of the
    permanent leg.
    """
    participation = net_flow / adv
    exec_price = spot * (1.0 + eta * sigma * participation)
    new_spot = spot * (1.0 + gamma * sigma * participation)
    return exec_price, new_spot
