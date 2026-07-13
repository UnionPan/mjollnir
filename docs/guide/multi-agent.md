# Market impact & multi-agent simulation

## The contract

Multi-agent extension is **functional composition, not inheritance**: a
`MarketState` PyTree threaded through a pure step built by
`make_market_step`, which fuses two library-owned pieces —

1. an **impact function** (how aggregate order flow moves the mid), and
2. a **dynamics kernel** (QE family; Heston by default)

— into one jittable, vmappable, differentiable transition:

```python
from mjollnir.jax import MarketState, make_market_step, sqrt_impact

step = make_market_step(
    dt=1/252, mu=0.02, **params.as_kwargs(),
    impact=lambda s, q: sqrt_impact(s, q, adv=1e6, sigma=0.0126),
)

state = MarketState.create(spot=100.0, variance=0.04, key=key)
state = step(state, net_order_flow)     # inside your lax.scan body
```

Everything inside the market boundary (state, impact, dynamics) is
library-owned and versioned; everything inside an agent's head (policies,
observations, rewards, inventories) lives in *your* scan carry. See
`examples/multi_agent_impact.py` for five heterogeneous momentum traders
coupled through square-root impact in a single XLA program.

## The impact zoo

| Function | Law | Use |
|---|---|---|
| `linear_impact(spot, q, adv, lam)` | Kyle: move ∝ participation | information-based benchmark |
| `sqrt_impact(spot, q, adv, sigma, y)` | square-root law, concave | the empirical standard for metaorders |
| `almgren_chriss_step(...)` | linear temporary + permanent split | execution-cost modelling |

All are pure and differentiable; `net_flow = 0` is always the identity.
Impact lives *inside* the versioned kernel because it is the coupling
through which agents' actions reach each other — if it drifted between
versions, multi-agent equilibria would silently stop being comparable.

## Why this forces JAX policies

Once actions feed back into price, paths cannot be pre-simulated: the
policy must execute inside the step loop, inside XLA. Torch policies remain
fully supported at the gym boundary and for impact-free batch training (see
the integration-tiers table in the RL substrate guide); impact/multi-agent
work is where flax/equinox become the price of admission.
