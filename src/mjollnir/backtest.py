"""Backtesting driver: (ParamSet, Scenario, strategy) -> P&L statistics.

The missing middle layer between the kernel and research scripts: everything
the examples hand-rolled (episode scan, wealth bookkeeping, metrics) done
once, correctly, with provenance. The whole backtest — market dynamics,
optional impact, and the strategy — compiles into a single XLA program and
is vmapped over paths.

Contract:

* market physics comes from a ParamSet of any model with a simulated world in
  :mod:`mjollnir.worlds` (optionally transformed by a ``Scenario`` first, the
  counterfactual path);
* the ``strategy`` is a jax-traceable callable
  ``(spot, state, position, step) -> target_position`` in units of the
  underlying (deltas, not orders), where ``state`` is what the model's world
  exposes (the variance for Heston);
* cash earns ``rate`` and held shares earn the dividend yield ``dividend``;
  the spot itself drifts at ``mu``;
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

from mjollnir.params import ParamSet
from mjollnir.scenarios import Scenario
from mjollnir.worlds import World, market_params, world_for

__all__ = ["BacktestResult", "HedgeMarket", "ParametricStrategy", "hedge_market", "run_backtest", "run_backtests",
           "simulate_hedge"]


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


@partial(jax.jit, static_argnames=("world", "strategy_fn", "impact", "horizon_steps"))
def _simulate(keys, market, model, world: World, strategy_fn, strategy_params, *, impact, horizon_steps):
    """Vmapped episodes; every number arrives traced, so the program is reusable."""
    dt = market["dt"]
    growth = jnp.exp(market["rate"] * dt)          # cash over one step
    dividend = jnp.expm1(market["dividend"] * dt)  # per unit of value held over one step

    def episode(key):
        state, path, key = world.init(model, key, horizon_steps, dt)
        spot = jnp.asarray(market["spot0"], jnp.float64)

        def body(carry, i):
            spot, state, key, pos, cash, paid = carry
            target = strategy_fn(strategy_params, spot, state, pos, i)
            trade = target - pos
            cost = jnp.abs(trade) * spot * market["cost_rate"]
            cash = (cash - trade * spot - cost) * growth + target * spot * dividend
            moved = spot if impact is None else impact(spot, trade)
            new_spot, state, key = world.step(moved, state, i, model, path, dt, market["mu"], key)
            wealth = cash + target * new_spot
            return (new_spot, state, key, target, cash, paid + cost), (new_spot, wealth, target)

        zero = jnp.zeros((), jnp.float64)
        init = (spot, jnp.asarray(state, jnp.float64), key, zero,
                jnp.asarray(market["initial_cash"], jnp.float64), zero)
        (*_, paid), (spots, wealth, positions) = jax.lax.scan(body, init, jnp.arange(horizon_steps))
        spots = jnp.concatenate([spot[None], spots])
        wealth = jnp.concatenate([jnp.asarray(market["initial_cash"], jnp.float64)[None], wealth])
        return spots, wealth, positions, paid

    return jax.vmap(episode)(keys)


@partial(jax.jit, static_argnames=("world", "strategy_fns", "horizon_steps"))
def _simulate_many(keys, market, model, world: World, strategy_fns, strategy_params, *, horizon_steps):
    """Several strategies on the same paths in one scan: the world is simulated
    once and every strategy keeps its own position, cash and costs. Valid only
    without market impact (the strategies cannot move the price)."""
    dt = market["dt"]
    growth = jnp.exp(market["rate"] * dt)
    dividend = jnp.expm1(market["dividend"] * dt)
    n = len(strategy_fns)

    def episode(key):
        state, path, key = world.init(model, key, horizon_steps, dt)
        spot = jnp.asarray(market["spot0"], jnp.float64)

        def body(carry, i):
            spot, state, key, pos, cash, paid = carry
            target = jnp.stack([fn(params, spot, state, pos[j], i)
                                for j, (fn, params) in enumerate(zip(strategy_fns, strategy_params, strict=True))])
            trade = target - pos
            cost = jnp.abs(trade) * spot * market["cost_rate"]
            cash = (cash - trade * spot - cost) * growth + target * spot * dividend
            new_spot, state, key = world.step(spot, state, i, model, path, dt, market["mu"], key)
            wealth = cash + target * new_spot
            return (new_spot, state, key, target, cash, paid + cost), (new_spot, wealth, target)

        zeros = jnp.zeros((n,), jnp.float64)
        cash0 = jnp.full((n,), market["initial_cash"], jnp.float64)
        init = (spot, jnp.asarray(state, jnp.float64), key, zeros, cash0, zeros)
        (*_, paid), (spots, wealth, positions) = jax.lax.scan(body, init, jnp.arange(horizon_steps))
        spots = jnp.concatenate([spot[None], spots])
        wealth = jnp.concatenate([cash0[None, :], wealth])          # (horizon + 1, n)
        return spots, wealth, positions, paid

    return jax.vmap(episode)(keys)


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
    rate: float = 0.0,
    dividend: float = 0.0,
    spot0: float = 100.0,
    initial_cash: float = 0.0,
    cost_bps: float = 0.0,
    scenario: Scenario | None = None,
    impact: Callable[[Any, Any], Any] | None = None,
    seed: int = 0,
    steps_per_year: float = 252.0,
    liability: Callable[[Any], Any] | None = None,
) -> BacktestResult:
    """Run a vmapped, jit-compiled backtest of ``strategy`` in ``params``' world.

    Args:
        params: a :class:`ParamSet` of any model in :mod:`mjollnir.worlds`
            (measure P or Q; ``mu`` is supplied separately since Q-sets carry
            no drift).
        strategy: jax-traceable ``(spot, state, position, step) ->
            target_position`` in units of the underlying.
        n_paths: Monte Carlo paths (vmapped).
        horizon_steps: episode length in steps of ``dt``.
        mu: spot drift used for simulation (r - q for a risk-neutral drift).
        rate: interest earned on cash, continuously compounded.
        dividend: dividend yield earned on shares held.
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
    world, params, model, market = _prepare(params, scenario, dt, mu, rate, dividend, spot0, initial_cash,
                                            cost_bps)
    if isinstance(strategy, ParametricStrategy):
        strategy_fn, strategy_params = strategy.fn, strategy.params
    else:
        strategy_fn, strategy_params = _Closure(strategy), ()

    keys = jax.random.split(jax.random.PRNGKey(seed), n_paths)
    spots, wealth, positions, costs = _simulate(
        keys, market, model, world, strategy_fn, strategy_params,
        impact=impact, horizon_steps=horizon_steps,
    )
    return _result(np.asarray(spots), np.asarray(wealth), np.asarray(positions), np.asarray(costs),
                   params, scenario, liability, steps_per_year)


def _result(spots, wealth, positions, costs, params, scenario, liability, steps_per_year) -> BacktestResult:
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
        costs=costs,
        positions=positions,
    )


def _prepare(params: ParamSet, scenario: Scenario | None, dt, mu, rate, dividend, spot0, initial_cash, cost_bps):
    world = world_for(params.model)
    if scenario is not None:
        foreign = (set(scenario.scale) | set(scenario.set_)) - set(params.params)
        if foreign:
            # a set_ would silently add a parameter the world never reads: a
            # "counterfactual" identical to the factual world
            raise ValueError(f"scenario {scenario.name} changes {sorted(foreign)}, which a "
                             f"{params.model} model does not have")
        params = scenario.apply(params)
    model = market_params(params.model, params.as_kwargs())
    market = {
        "dt": dt, "mu": mu, "rate": rate, "dividend": dividend, "spot0": spot0,
        "initial_cash": initial_cash, "cost_rate": cost_bps * 1e-4,
    }
    return world, params, model, {k: jnp.asarray(v, jnp.float64) for k, v in market.items()}


class HedgeMarket(NamedTuple):
    """A world and its parameters as traced arrays: the inputs of :func:`simulate_hedge`.

    ``model`` holds the world's parameters and ``market`` the drift, rates,
    spot, cash and cost rate, all as JAX arrays, so both can be vmapped
    (several books or drifts in one program) or differentiated through.
    """

    world: World
    model: dict
    market: dict


def hedge_market(params: ParamSet, *, dt: float = 1.0 / 252, mu: float = 0.0, rate: float = 0.0,
                 dividend: float = 0.0, spot0: float = 100.0, initial_cash: float = 0.0,
                 cost_bps: float = 0.0, scenario: Scenario | None = None) -> HedgeMarket:
    """The traced inputs :func:`simulate_hedge` takes for ``params``' world."""
    world, _, model, market = _prepare(params, scenario, dt, mu, rate, dividend, spot0, initial_cash, cost_bps)
    return HedgeMarket(world, model, market)


def simulate_hedge(market: HedgeMarket, strategy_fn: Callable[..., Any], strategy_params: Any, keys, *,
                   horizon_steps: int, payoff: Callable[[Any], Any] | None = None):
    """Per-path hedging P&L and costs as JAX arrays: the differentiable core of a backtest.

    ``strategy_fn(strategy_params, spot, state, position, step)`` is run on one
    path per key of ``market``'s world; returns ``(pnl, costs)``, each
    ``(len(keys),)``, where pnl is terminal wealth minus ``payoff`` of the
    terminal spot. Everything is traceable, so ``jax.grad`` with respect to
    ``strategy_params`` (or the market's traced numbers) trains a strategy
    through the simulated world: proportional costs are differentiable almost
    everywhere. ``strategy_fn`` must be hashable and stable (a module-level
    function), as in :class:`ParametricStrategy`.
    """
    spots, wealth, _, costs = _simulate(keys, market.market, market.model, market.world, strategy_fn,
                                        strategy_params, impact=None, horizon_steps=horizon_steps)
    pnl = wealth[:, -1]
    if payoff is not None:
        pnl = pnl - payoff(spots[:, -1])
    return pnl, costs


def run_backtests(
    params: ParamSet,
    strategies: dict[str, ParametricStrategy],
    *,
    n_paths: int = 256,
    horizon_steps: int = 252,
    dt: float = 1.0 / 252,
    mu: float = 0.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    spot0: float = 100.0,
    initial_cash: float = 0.0,
    cost_bps: float = 0.0,
    scenario: Scenario | None = None,
    seed: int = 0,
    steps_per_year: float = 252.0,
    liability: Callable[[Any], Any] | None = None,
) -> dict[str, BacktestResult]:
    """:func:`run_backtest` for several parametric strategies on the same paths.

    One compiled program simulates the world once and runs every strategy
    alongside (no market impact, so none of them moves the price); results
    equal separate ``run_backtest`` calls with the same seed.
    """
    world, params, model, market = _prepare(params, scenario, dt, mu, rate, dividend, spot0, initial_cash,
                                            cost_bps)
    names = list(strategies)
    fns = tuple(strategies[n].fn for n in names)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_paths)
    spots, wealth, positions, costs = _simulate_many(
        keys, market, model, world, fns, tuple(strategies[n].params for n in names), horizon_steps=horizon_steps)
    spots, wealth, positions, costs = (np.asarray(x) for x in (spots, wealth, positions, costs))
    return {name: _result(spots, wealth[:, :, j], positions[:, :, j], costs[:, j], params, scenario, liability,
                          steps_per_year)
            for j, name in enumerate(names)}
