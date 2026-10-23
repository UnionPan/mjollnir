"""Backtesting driver: (ParamSet, Scenario, strategy) -> P&L statistics.

The missing middle layer between the kernel and research scripts: everything
the examples hand-rolled (episode scan, wealth bookkeeping, metrics) done
once, correctly, with provenance. The whole backtest — market dynamics,
optional impact, and the strategy — compiles into a single XLA program and
is vmapped over paths.

Contract:

* market physics comes from a Q- or P-measure Heston ``ParamSet`` (optionally
  transformed by a ``Scenario`` first — the counterfactual path);
* the ``strategy`` is a jax-traceable callable
  ``(spot, variance, position, step) -> target_position`` in units of the
  underlying (deltas, not orders);
* trading costs enter either as proportional ``cost_bps`` or through a
  market-impact function (both differentiable, so a strategy's parameters
  can be trained through the backtest itself);
* an optional ``liability`` (a payoff of the terminal spot, e.g. a short
  option) turns the run into a hedging backtest: P&L is terminal wealth minus
  the payoff, and ``initial_cash`` typically carries the premium received.

Compilation. Market parameters always enter the compiled program as traced
inputs. A :class:`ParametricStrategy` (a stable function plus a pytree of
parameters) does too, so backtests that differ only in numbers (another
ParamSet, a scenario, a strike) reuse one compiled program. A plain closure
is compiled afresh for each new closure object.

Example::

    from mjollnir.backtest import run_backtest
    from mjollnir.scenarios import vol_spike

    result = run_backtest(param_set, momentum, n_paths=512, horizon_steps=252,
                          scenario=vol_spike(2.0), seed=7)
    print(result.summary())
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mjollnir.jax._market import MarketState, make_market_step
from mjollnir.params import ParamSet
from mjollnir.scenarios import Scenario

__all__ = ["BacktestResult", "ParametricStrategy", "run_backtest"]


class ParametricStrategy(NamedTuple):
    """A strategy split into a stable function and its numeric parameters.

    ``fn(params, spot, variance, position, step) -> target_position``. ``fn``
    should be a module-level function and ``params`` a pytree of numbers, so
    the compiled backtest is reused across parameter values. Instances are
    also plain strategies: calling one evaluates ``fn`` with its ``params``.
    """

    fn: Callable[..., Any]
    params: Any

    def __call__(self, spot, variance, position, step):
        return self.fn(self.params, spot, variance, position, step)


@dataclass(frozen=True)
class _Closure:
    """Adapts a plain ``(spot, variance, position, step)`` strategy; hashes by identity."""

    strategy: Callable[[Any, Any, Any, Any], Any]

    def __call__(self, params, spot, variance, position, step):
        return self.strategy(spot, variance, position, step)


@partial(jax.jit, static_argnames=("strategy_fn", "impact", "horizon_steps"))
def _simulate(keys, market, strategy_fn, strategy_params, *, impact, horizon_steps):
    """Vmapped episodes; every number arrives traced, so the program is reusable."""
    step = make_market_step(
        dt=market["dt"], mu=market["mu"], kappa=market["kappa"], theta=market["theta"],
        sigma_v=market["sigma_v"], rho=market["rho"], impact=impact,
    )

    def episode(key):
        state = MarketState.create(spot=market["spot0"], variance=market["v0"], key=key)

        def body(carry, i):
            state, pos, cash, paid = carry
            target = strategy_fn(strategy_params, state.spot, state.variance, pos, i)
            trade = target - pos
            cost = jnp.abs(trade) * state.spot * market["cost_rate"]
            cash = cash - trade * state.spot - cost
            new_state = step(state, trade)
            wealth = cash + target * new_state.spot
            return (new_state, target, cash, paid + cost), (new_state.spot, wealth, target)

        zero = jnp.zeros((), jnp.float64)
        init = (state, zero, jnp.asarray(market["initial_cash"], jnp.float64), zero)
        (_final, _pos, _cash, paid), (spots, wealth, positions) = jax.lax.scan(
            body, init, jnp.arange(horizon_steps))
        spots = jnp.concatenate([jnp.asarray(market["spot0"])[None], spots])
        wealth = jnp.concatenate([jnp.asarray(market["initial_cash"], jnp.float64)[None], wealth])
        return spots, wealth, positions, paid

    return jax.vmap(episode)(keys)

_HESTON_KEYS = {"v0", "kappa", "theta", "sigma_v", "rho"}


@dataclass(frozen=True)
class BacktestResult:
    """Per-path wealth trajectories plus headline statistics."""

    wealth: np.ndarray          # (n_paths, horizon+1) mark-to-market trading wealth
    spots: np.ndarray           # (n_paths, horizon+1)
    mean_terminal: float        # mean of ``pnl``
    std_terminal: float         # std of ``pnl``
    sharpe: float               # annualized, from per-step wealth increments
    max_drawdown: float         # worst peak-to-trough across paths (fraction of peak)
    params_hash: str            # provenance: the ParamSet actually simulated
    scenario: str | None        # scenario name if one was applied
    pnl: np.ndarray             # (n_paths,) terminal wealth minus the liability payoff, if any
    costs: np.ndarray           # (n_paths,) total transaction costs paid
    positions: np.ndarray       # (n_paths, horizon) position held after each step's trade

    def summary(self) -> str:
        return (f"terminal {self.mean_terminal:+.2f} ± {self.std_terminal:.2f} | "
                f"sharpe {self.sharpe:.2f} | max drawdown {self.max_drawdown:.1%}"
                + (f" | scenario {self.scenario}" if self.scenario else ""))


def run_backtest(
    params: ParamSet,
    strategy: Callable[[Any, Any, Any, Any], Any],
    *,
    n_paths: int = 256,
    horizon_steps: int = 252,
    dt: float = 1.0 / 252,
    mu: float = 0.0,
    spot0: float = 100.0,
    initial_cash: float = 0.0,
    cost_bps: float = 0.0,
    scenario: Scenario | None = None,
    impact: Callable[[Any, Any], Any] | None = None,
    seed: int = 0,
    steps_per_year: float = 252.0,
    liability: Callable[[Any], Any] | None = None,
) -> BacktestResult:
    """Run a vmapped, jit-compiled Heston backtest of ``strategy``.

    Args:
        params: a Heston :class:`ParamSet` (measure P or Q; ``mu`` is supplied
            separately since Q-sets carry no drift).
        strategy: jax-traceable ``(spot, variance, position, step) ->
            target_position`` in units of the underlying.
        n_paths: Monte Carlo paths (vmapped).
        horizon_steps: episode length in steps of ``dt``.
        mu: drift used for simulation (0 for risk-neutral experiments).
        cost_bps: proportional transaction cost in basis points of traded value.
        scenario: optional :class:`~mjollnir.scenarios.Scenario` applied to
            ``params`` first (provenance-chained).
        impact: optional impact function ``(spot, net_flow) -> spot`` — couples
            the strategy's own trading back into the price it receives.
        seed: PRNG seed for the path keys.
        liability: optional payoff of the terminal spot owed at the horizon
            (jax- or numpy-traceable), e.g. ``EuropeanOption.payoff`` for a
            short option; ``pnl`` and the headline statistics net it out.
    """
    if params.model != "heston":
        raise ValueError(f"run_backtest currently supports heston ParamSets, "
                         f"got model={params.model!r}")
    if scenario is not None:
        params = scenario.apply(params)
    p = params.as_kwargs()
    missing = _HESTON_KEYS - p.keys()
    if missing:
        raise ValueError(f"ParamSet missing Heston parameters: {sorted(missing)}")

    market = {
        "dt": dt, "mu": mu, "spot0": spot0, "initial_cash": initial_cash,
        "cost_rate": cost_bps * 1e-4,
        **{k: p[k] for k in ("v0", "kappa", "theta", "sigma_v", "rho")},
    }
    market = {k: jnp.asarray(v, jnp.float64) for k, v in market.items()}
    if isinstance(strategy, ParametricStrategy):
        strategy_fn, strategy_params = strategy.fn, strategy.params
    else:
        strategy_fn, strategy_params = _Closure(strategy), ()

    keys = jax.random.split(jax.random.PRNGKey(seed), n_paths)
    spots, wealth, positions, costs = _simulate(
        keys, market, strategy_fn, strategy_params,
        impact=impact, horizon_steps=horizon_steps,
    )
    spots, wealth = np.asarray(spots), np.asarray(wealth)

    terminal = wealth[:, -1]
    if liability is not None:
        terminal = terminal - np.asarray(liability(spots[:, -1]), dtype=terminal.dtype)
    increments = np.diff(wealth, axis=1)
    inc_mean, inc_std = increments.mean(), increments.std()
    sharpe = float(inc_mean / inc_std * np.sqrt(steps_per_year)) if inc_std > 0 else 0.0

    running_peak = np.maximum.accumulate(wealth, axis=1)
    denom = np.maximum(np.abs(running_peak), 1e-12)
    drawdown = (running_peak - wealth) / denom
    max_dd = float(drawdown.max())

    return BacktestResult(
        wealth=wealth,
        spots=spots,
        mean_terminal=float(terminal.mean()),
        std_terminal=float(terminal.std()),
        sharpe=sharpe,
        max_drawdown=max_dd,
        params_hash=params.content_hash(),
        scenario=getattr(scenario, "name", None),
        pnl=terminal,
        costs=np.asarray(costs),
        positions=np.asarray(positions),
    )
