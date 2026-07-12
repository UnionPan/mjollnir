# Mode 1 — the JAX simulation kernel (RL substrate)

The frozen public API that RL environments pin a version against: jittable,
batchable, differentiable, with explicit PRNG-key threading and no hidden
global RNG.

## Runtime configuration

Call once per process, before building jitted graphs:

```python
from mjollnir.jax import configure_runtime
configure_runtime()   # returns the active backend, e.g. "cpu"
```

Device and precision are controlled by environment variables
(`MJOLLNIR_JAX_BACKEND`, `MJOLLNIR_JAX_PRECISION`, `MJOLLNIR_JAX_STRICT_BACKEND`).

## Heston QE stepping

```python
import jax, jax.numpy as jnp
from mjollnir.jax import qe_heston_step

key = jax.random.PRNGKey(0)
spot = jnp.full((1024,), 100.0)
var = jnp.full((1024,), 0.04)

spot, var, key = qe_heston_step(
    spot, var, dt=1/252, mu=0.02,
    kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, key=key,
)
```

One call = one Andersen quadratic-exponential step of `(S, v)`; variance is
guaranteed non-negative. The returned key is the split successor — thread it
into the next step. For per-path independent noise, `jax.vmap` over a batch
of keys.

The full QE family is exported alongside it: `qe_bates_step` (Heston +
Merton jumps), `qe_three_half_step` (3/2 model), `qe_four_half_step`
(4/2, Grasselli 2017) — same signature shape, same key-threading contract,
all golden-tested in `tests/test_kernel_family_golden.py`. COS pricers for
Merton, Bates, Kou, VG and NIG are exported as `jax_cos_price_<model>` /
`jax_cos_price_<model>_multi`.

## Differentiable pricing

```python
from mjollnir.jax import fourier_price

price = fourier_price(100.0, 100.0, 0.5, 0.02, 0.0,   # S0, K, T, r, q
                      0.04, 2.0, 0.04, 0.3, -0.7)     # v0, kappa, theta, sigma_v, rho

delta = jax.grad(lambda s0: fourier_price(s0, 100.0, 0.5, 0.02, 0.0,
                                          0.04, 2.0, 0.04, 0.3, -0.7))(100.0)
```

`fourier_price` / `fourier_price_batch` are fully traced: `jax.grad`,
`jax.jit`, `jax.vmap` and `lax.scan` all work, with gradients with respect to
any market input. The COS truncation bounds are computed in-graph but wrapped
in `stop_gradient` — they only select the integration grid, and differentiating
through them adds noise, not signal.

For custom pricing graphs (e.g. padded option grids inside a `lax.scan`
environment step), compose the exported building blocks directly:
`heston_cf`, `HestonCFParams`, `cos_price_multi`, `cos_price_single`,
`chi_k`, `psi_k`, `cos_truncation_range`.
