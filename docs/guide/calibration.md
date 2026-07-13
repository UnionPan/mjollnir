# Mode 2 — calibration & backtesting

Source calibrated model parameters from market data under both measures:

- **Physical measure** (`mjollnir.calibration.physical`): QMLE, particle
  filters (Heston, rough Bergomi), GARCH, OHLC volatility estimators, and a
  neural posterior estimator (NPE).
- **Risk-neutral measure** (`mjollnir.calibration.risk_neutral`): Heston-COS,
  SABR, Dupire local vol, SVI/SSVI surface fits.

## Particle filtering

```python
from mjollnir.calibration.physical.heston_particle_filter import HestonParticleFilter

pf = HestonParticleFilter(n_particles=2000)
```

## Gradient-based surface calibration

`fit_heston_surface` differentiates the library's own COS pricer exactly
(`jax.grad` through `fourier_price_batch`) — no finite differences, and the
whole fit is jit-compiled. It returns a Q-measure `ParamSet`, so the result
plugs straight into the artifact store, scenarios, and the kernel:

```python
from mjollnir.calibration import fit_heston_surface

quotes = [(strike, T_years, is_call, mid), ...]
res = fit_heston_surface(quotes, S0=100.0, r=0.02, asset="SPY")
res.param_set.save("data/params/spy_heston_q.json")
```

On a noise-free 36-quote surface it recovers all five Heston parameters to
<0.01% (tested); the amortized-NPE and signature-generator successors are
specified in `docs/design/neural-calibration.md`.

## Command-line workflow

Installed with the package:

```bash
mjollnir-build-universes   # assemble ticker universes
mjollnir-calibrate         # calibrate per-universe model parameters
mjollnir-train-npe         # train the neural posterior estimator
```

## Synthetic option chains

`mjollnir.synthetic_data` generates realistic option chains (IV smile/skew from a
Heston or Merton pricer) for evaluation and for the hedging environments:

```python
from mjollnir.synthetic_data import SyntheticEquityOptionChainGenerator, HestonVolatilityProfile
```

Because these generators feed the RL environments' observations, they live
*inside* the versioned substrate — pinning a `mjollnir` version pins the
chains an agent sees.
