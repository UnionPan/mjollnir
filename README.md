# mjollnir

A stable, reproducible, differentiable market-simulation and calibration substrate
for RL research (deep hedging / POMARL) and backtesting.

Extracted from the `option_pricer_DLC` monorepo (`src/options_desk`); this library is
the foundation the actual research sits on, not the end product. Research-grade and
usable at some scale — explicitly **not** exchange connectivity, tick data, or
latency-guaranteed infrastructure.

## Install

```bash
pip install -e .            # core (CPU jax)
pip install -e ".[cuda]"    # NVIDIA GPUs
pip install -e ".[metal]"   # Apple silicon
pip install -e ".[dev]"     # + pytest, ruff, mypy
```

Requires Python >= 3.10.

## Two consumption modes

### Mode 1 — the JAX simulation kernel (RL substrate)

The frozen public API that RL environments pin a version against. Jittable,
batchable, differentiable; explicit PRNG-key threading, no hidden global RNG.

```python
import jax, jax.numpy as jnp
from mjollnir.jax import configure_runtime, qe_heston_step, fourier_price

configure_runtime()                       # device / precision setup, once

# one Andersen-QE Heston step: (S, v, key) -> (S', v', key')
key = jax.random.PRNGKey(0)
spot, var, key = qe_heston_step(
    jnp.full((1024,), 100.0), jnp.full((1024,), 0.04),
    dt=1/252, mu=0.02, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, key=key,
)

# differentiable COS price: grad/jit/vmap-safe w.r.t. all market inputs
delta = jax.grad(lambda s0: fourier_price(
    s0, 100.0, 0.5, 0.02, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7))(100.0)
```

Also public: `fourier_price_batch` (strike vectors), `heston_cf` /
`HestonCFParams`, and the composable COS building blocks (`cos_price_multi`,
`cos_price_single`, `chi_k`, `psi_k`, `cos_truncation_range`) for consumers that
assemble their own pricing graphs (e.g. padded option grids inside `lax.scan`).

**Reproducibility contract.** `tests/test_parity_golden.py` pins golden-key rollout
and pricer outputs. A change that breaks those tests is a breaking change of env
dynamics and must be released as such — never silently.

### Mode 2 — calibration objects + CLIs (backtesting)

Source calibrated model parameters from market data, under both the physical and
risk-neutral measures.

```python
from mjollnir.calibration.physical.heston_particle_filter import HestonParticleFilter

pf = HestonParticleFilter(n_particles=2000)
```

Console entry points (installed with the package):

```bash
mjollnir-build-universes   # assemble ticker universes
mjollnir-calibrate         # calibrate per-universe model parameters
mjollnir-train-npe         # train the neural posterior estimator
```

## Package map

| Package               | Role |
|-----------------------|------|
| `mjollnir.jax`        | **Public frozen API**: QE Heston kernel + differentiable COS pricer + runtime config |
| `mjollnir.processes`  | SDE model layer: GBM, Heston, SABR, 3/2, 4/2, Merton, Kou, rough Bergomi, VG, NIG, SLV, regime-switching, Vasicek/CIR/HW |
| `mjollnir.pricer`     | COS/Fourier + MGF pricing kernels |
| `mjollnir.derivatives`| Vanilla contract definitions |
| `mjollnir.synthetic`  | Shared low-level layer: `OptionChain`/`OptionQuote` value types + synthetic option-chain generators (used by *both* simulations and calibration) |
| `mjollnir.calibration`| Physical (QMLE, particle filters, GARCH, OHLC vol, NPE) and risk-neutral (Heston-COS, SABR, Dupire, SVI/SSVI) calibration, data fetchers, CLIs |
| `mjollnir.simulations`| Gym-style Heston / Merton / rough-Bergomi hedging environments |

Dependency direction: `processes` → `pricer`/`derivatives` → `synthetic` →
`calibration`/`simulations`. `simulations` never imports `calibration`.

## Tests

```bash
python -m pytest tests/ -q
```
