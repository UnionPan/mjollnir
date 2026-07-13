"""Multi-agent market with impact: the composition pattern, end to end.

Five momentum traders with heterogeneous aggressiveness share one Heston
market and couple through square-root impact — each agent's trades move the
price every other agent sees. The whole episode (policies + impact +
dynamics) compiles into a single XLA program.

This is a *template*, not a framework: the market physics (``MarketState``,
``make_market_step``, ``sqrt_impact``, the QE kernel) is library-owned and
versioned; the agents (policy, observations, rewards) are ~15 lines of user
code you are meant to replace. Executed as a test by ``tests/test_examples.py``.
"""

import jax
import jax.numpy as jnp

from mjollnir.jax import MarketState, configure_runtime, make_market_step, sqrt_impact

configure_runtime()

# ---------------------------------------------------------------------------
# Market physics (library-owned, would normally come from a ParamSet)
# ---------------------------------------------------------------------------
HESTON = dict(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
DT = 1.0 / 252
ADV = 1e6            # shares/day
DAILY_SIGMA = 0.2 / jnp.sqrt(252.0)

market_step = make_market_step(
    dt=DT, mu=0.02, **HESTON,
    impact=lambda spot, q: sqrt_impact(spot, q, adv=ADV, sigma=DAILY_SIGMA),
)

# ---------------------------------------------------------------------------
# Agents (user research code — replace freely)
# ---------------------------------------------------------------------------
N_AGENTS = 5
AGGRESSION = jnp.linspace(0.5, 2.0, N_AGENTS)   # heterogeneous momentum gains
MAX_ORDER = 0.02 * ADV                          # per-agent per-step clip


def policies(prev_spot, spot, inventories):
    """Momentum with inventory mean-reversion, vmapped over agents."""
    signal = jnp.log(spot / prev_spot)
    orders = AGGRESSION * signal * ADV - 0.1 * inventories
    return jnp.clip(orders, -MAX_ORDER, MAX_ORDER)


def episode(key, n_steps=126):
    state = MarketState.create(spot=100.0, variance=HESTON["theta"], key=key)

    def body(carry, _):
        state, prev_spot, inv, cash = carry
        orders = policies(prev_spot, state.spot, inv)          # (N_AGENTS,)
        # everyone trades at the pre-impact mid, then flow moves the market
        cash = cash - orders * state.spot
        inv = inv + orders
        new_state = market_step(state, orders.sum())           # coupled here
        return (new_state, state.spot, inv, cash), new_state.spot

    init = (state, state.spot, jnp.zeros(N_AGENTS), jnp.zeros(N_AGENTS))
    (final, _, inv, cash), spots = jax.lax.scan(body, init, None, length=n_steps)
    wealth = cash + inv * final.spot
    return wealth, spots


wealth, spots = jax.jit(episode)(jax.random.PRNGKey(7))
print("terminal spot:", float(spots[-1]))
print("agent wealth :", [round(float(w), 1) for w in wealth])

# structural sanity: episode ran, prices stayed positive/finite,
# heterogeneous agents ended with different wealth
assert bool(jnp.isfinite(spots).all()) and bool((spots > 0).all())
assert len({round(float(w), 6) for w in wealth}) == N_AGENTS

# the coupling is real: without impact, identical seeds give a different path
no_impact_step = make_market_step(dt=DT, mu=0.02, **HESTON, impact=None)


def episode_no_impact(key, n_steps=126):
    state = MarketState.create(spot=100.0, variance=HESTON["theta"], key=key)

    def body(carry, _):
        state, prev_spot, inv, cash = carry
        orders = policies(prev_spot, state.spot, inv)
        cash = cash - orders * state.spot
        inv = inv + orders
        new_state = no_impact_step(state, orders.sum())
        return (new_state, state.spot, inv, cash), new_state.spot

    init = (state, state.spot, jnp.zeros(N_AGENTS), jnp.zeros(N_AGENTS))
    (_, _, _, _), spots = jax.lax.scan(body, init, None, length=n_steps)
    return spots


spots_free = jax.jit(episode_no_impact)(jax.random.PRNGKey(7))
assert not bool(jnp.allclose(spots, spots_free)), "impact must change the path"
print("multi-agent coupling OK")
