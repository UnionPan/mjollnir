"""Minimal fused deep-hedging rollout: policy inside ``lax.scan``.

The tier-1 integration pattern: market step + differentiable pricer + policy
in one XLA program. The "policy" here is the model delta (no learnable
parameters) so the example stays dependency-light; swap in a flax module and
``jax.grad`` w.r.t. its params for actual training. Executed as a test by
``tests/test_examples.py``.
"""

import jax
import jax.numpy as jnp

from mjollnir.jax import configure_runtime, fourier_price, qe_heston_step

configure_runtime()

P = dict(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
S0, K, T, R = 100.0, 100.0, 0.25, 0.02
DT = 1.0 / 252
N_STEPS = int(T / DT)
N_PATHS = 256


def policy(spot, tau):
    """Model delta via the differentiable pricer (stand-in for a network)."""
    return jax.grad(lambda s: fourier_price(s, K, jnp.maximum(tau, 1e-3),
                                            R, 0.0, 0.04, **P))(spot)


def rollout(key):
    """One hedged episode; returns terminal hedging error."""
    def step(carry, i):
        spot, var, pos, cash, key = carry
        tau = T - i * DT
        d = policy(spot, tau)
        cash = cash - (d - pos) * spot
        pos = d
        spot, var, key = qe_heston_step(spot, var, DT, R, **P, key=key)
        return (spot, var, pos, cash, key), None

    premium = fourier_price(S0, K, T, R, 0.0, 0.04, **P)
    init = (jnp.asarray(S0), jnp.asarray(0.04), jnp.asarray(0.0),
            premium, key)
    (spot, _var, pos, cash, _key), _ = jax.lax.scan(
        step, init, jnp.arange(N_STEPS))
    payoff = jnp.maximum(spot - K, 0.0)
    return cash + pos * spot - payoff


keys = jax.random.split(jax.random.PRNGKey(0), N_PATHS)
errors = jax.jit(jax.vmap(rollout))(keys)
mean_err = float(errors.mean())
std_err = float(errors.std())
print(f"hedging error over {N_PATHS} paths: mean {mean_err:+.4f}, std {std_err:.4f}")

# daily delta-hedging under Heston is imperfect but tight:
assert abs(mean_err) < 0.5, "hedge is unbiased to within half a dollar"
assert std_err < 0.15 * fourier_price(S0, K, T, R, 0.0, 0.04, **P) + 1.0
print("rollout OK")
