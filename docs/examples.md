# Examples

Both examples live in [`examples/`](https://github.com/unionpan/mjollnir/tree/main/examples)
and are **executed as part of the test suite** (`tests/test_examples.py`), so
they cannot drift out of date. They run offline, seeded, in seconds.

## End-to-end workflow: simulate → calibrate → synthesize → backtest

[`examples/heston_workflow.py`](https://github.com/unionpan/mjollnir/blob/main/examples/heston_workflow.py)

1. **Market data** — eight years of daily returns from a known Heston market
   (stand-in for a `marketdata` fetch, so the example is deterministic).
2. **Physical calibration** — batched Heston QMLE recovers the long-run
   variance well; the example documents what close-close returns *cannot*
   see (rho) and which estimators to reach for instead (OHLC proxy,
   particle filter, NPE).
3. **Synthetic chain** — a full equity option chain generated from the
   calibrated parameters.
4. **Backtest** — a 6-month ATM call is delta-hedged daily along a fresh
   simulated path, deltas from `jax.grad(fourier_price)`.

## Fused deep-hedging rollout (tier-1 pattern)

[`examples/deep_hedging_rollout.py`](https://github.com/unionpan/mjollnir/blob/main/examples/deep_hedging_rollout.py)

Market step, differentiable pricer, and policy compiled into **one XLA
program**: `qe_heston_step` + `fourier_price` inside `lax.scan`, `vmap`-ed
over 256 paths. The policy is the model delta (no parameters) to keep the
example dependency-light — swap in a flax module and differentiate the
episode loss w.r.t. its parameters for actual deep hedging.

## Multi-agent market with impact

[`examples/multi_agent_impact.py`](https://github.com/unionpan/mjollnir/blob/main/examples/multi_agent_impact.py)

Five heterogeneous momentum traders share one Heston market and couple
through square-root impact — `MarketState` + `make_market_step` +
`sqrt_impact`, with policies and bookkeeping as ~15 lines of user code in
the scan carry. Includes a falsification check: the same seed without
impact produces a different path, proving the coupling is real.
