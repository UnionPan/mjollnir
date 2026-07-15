# Backtesting & the vol surface

## `mjollnir.backtest` — the driver

`run_backtest` formalizes the loop every research script hand-rolls:
market dynamics from a `ParamSet` (optionally shocked by a `Scenario`),
a jax-traceable strategy, optional impact/transaction costs, wealth
bookkeeping, and headline statistics — compiled into one XLA program and
vmapped over paths.

```python
from mjollnir.backtest import run_backtest
from mjollnir.scenarios import vol_spike

def momentum(spot, variance, position, step):
    ...  # target position in units of the underlying

res = run_backtest(params, momentum, n_paths=512, horizon_steps=252,
                   scenario=vol_spike(2.0), cost_bps=5.0, seed=7)
print(res.summary())   # terminal ± std | sharpe | max drawdown | scenario
```

Every result carries the **content hash of the ParamSet actually simulated**
(post-scenario), so a backtest is always attributable to a specific,
verifiable parameter artifact. Costs and impact are differentiable — a
strategy's parameters can be trained through the backtest itself.

## `mjollnir.surface` — the vol surface object

`ImpliedVolSurface` bridges market data and calibration:

```python
from mjollnir.surface import ImpliedVolSurface

surf = ImpliedVolSurface.from_chain(chain)      # inverts missing IVs
report = surf.arbitrage_report()                # vertical / butterfly / calendar
assert report.ok, report.summary()

res = fit_heston_surface(surf.to_quotes(), surf.spot, surf.rate)
```

The arbitrage checks are the same laws the property suite enforces on the
pricer (call monotonicity, butterfly convexity, calendar total variance) —
applied to *data* instead of models. They report rather than raise: real
snapshots routinely carry small in-spread violations, and the researcher
decides the tolerance.
