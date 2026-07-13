# Reproducibility contract

`mjollnir` exists so RL experiments can pin a version and trust that market
dynamics do not drift. Three commitments back that:

## 1. Explicit randomness

Every stochastic primitive takes a JAX PRNG key and returns its successor.
There is no hidden global RNG and no mutable module state affecting dynamics.
Given the same key and inputs, outputs are deterministic.

## 2. Golden-value tests

`tests/test_parity_golden.py` pins the numerics of the public kernel:

- a fixed-key 64-step Heston QE rollout (spot & variance),
- the ATM COS price, its delta and its `v0` sensitivity,
- a jitted batch pricing result.

## 3. Versioning policy

- A change that breaks a golden test **is a breaking change of environment
  dynamics**, even if every API signature is unchanged. It must be released
  as a minor (pre-1.0) or major version bump with regenerated goldens and a
  changelog entry — never silently.
- Additive API changes (new processes, new estimators) are minor releases.
- Bug fixes that do not move golden values are patch releases.

Pin accordingly, e.g. `mjollnir==0.1.*` for an experiment series.

## Process-layer RNG contract (v0.2)

The `mjollnir.jax` kernel is fully key-threaded. The NumPy `processes` layer
uses **per-call `np.random.Generator`** instances (v0.2 migration; the old
global-`np.random.seed` pattern is gone):

- `SimulationConfig.random_seed` set → deterministic per call;
- `random_seed=None` → fresh OS entropy on **every** backend (the JAX fast
  paths no longer silently pin `None` to key 0);
- simulations never read or write the ambient global `np.random` stream, so
  interleaved processes cannot perturb each other;
- stochastic helper components (jump sizes, subordinators, regime paths)
  draw from the process's `sim_rng`, bound by `simulate()`.

`tests/test_rng_isolation.py` pins all four guarantees. Note: this migration
changed sampled trajectories relative to v0.1 (MT19937 → PCG64), per the
versioning policy above.

## Precision policy

- **`mjollnir.jax` kernel**: precision follows `configure_runtime()` —
  float64 on CPU/CUDA by default, controllable via `MJOLLNIR_JAX_PRECISION`
  (`high` | `metal_safe`). All golden values are generated under float64.
- **Batched calibrators** (`calibration.physical.batched`): accept an explicit
  `dtype` argument, **default `float32`** — chosen for GPU throughput on large
  cross-sections. The likelihood kernels are dtype-generic, so passing
  `dtype=jnp.float64` (and `pad_returns(..., dtype=np.float64)`) runs the whole
  fit in double precision. Expect ~1e-2-nat likelihood differences between the
  two on ~6k-observation fits; scipy reference implementations always run
  float64.
