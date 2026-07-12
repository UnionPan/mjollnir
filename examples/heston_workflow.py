"""End-to-end workflow: simulate -> calibrate -> synthesize -> backtest.

Runs offline and deterministically (no market-data fetch); in a live setup
the simulated returns below are replaced by a `mjollnir.calibration.marketdata`
fetch. Executed as a test by ``tests/test_examples.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mjollnir.jax import configure_runtime, fourier_price, qe_heston_step

configure_runtime()

# ---------------------------------------------------------------------------
# 1. "Market data": a year of daily returns from a known Heston market.
#    (Stand-in for a marketdata fetch, so the example is offline + seeded.)
# ---------------------------------------------------------------------------
TRUE = dict(kappa=3.0, theta=0.05, sigma_v=0.4, rho=-0.6)
DT = 1.0 / 252

key = jax.random.PRNGKey(11)
spot = jnp.full((1,), 100.0)
var = jnp.full((1,), TRUE["theta"])
path = [float(spot[0])]
for _ in range(2016):  # eight years of dailies (rho needs a long sample)
    spot, var, key = qe_heston_step(spot, var, DT, 0.03, **TRUE, key=key)
    path.append(float(spot[0]))
prices = np.asarray(path)
returns = np.diff(np.log(prices))

# ---------------------------------------------------------------------------
# 2. Physical-measure calibration (batched Heston QMLE).
# ---------------------------------------------------------------------------
from mjollnir.calibration.physical.batched import heston_qmle
from mjollnir.calibration.physical.batched.common import pad_returns

R, M = pad_returns([returns])
fit = heston_qmle.fit_batch(R, M, DT)
cal = {k: float(fit[k][0]) for k in ("kappa", "theta", "sigma_v", "rho")}
print(f"calibrated: { {k: round(v, 4) for k, v in cal.items()} }")
# What close-close QMLE can and cannot see: theta is recovered well, kappa
# and sigma_v roughly, rho barely (its information lives in the return-
# variance covariation, which close-close returns almost erase). For a real
# rho estimate use fit_batch_ohlc (Garman-Klass proxy), HestonParticleFilter,
# or the NPE — that menu is the point of the physical-calibration package.
assert abs(cal["theta"] - TRUE["theta"]) < 0.02, "theta recovered"
assert cal["kappa"] > 0 and cal["sigma_v"] > 0
assert cal["rho"] <= 0.0, "rho sign not inverted"

# ---------------------------------------------------------------------------
# 3. Synthetic option chain from the calibrated parameters.
# ---------------------------------------------------------------------------
from datetime import date

from mjollnir.synthetic_data import (
    HestonVolatilityProfile,
    SyntheticEquityOptionChainGenerator,
)

profile = HestonVolatilityProfile(
    kappa=cal["kappa"], theta=cal["theta"], xi=cal["sigma_v"], rho=cal["rho"],
    v0=cal["theta"], atm_iv=float(np.sqrt(cal["theta"])),
)
chain = SyntheticEquityOptionChainGenerator(random_seed=0).generate_single_chain(
    reference_date=date(2026, 1, 2), spot_price=float(prices[-1]),
    vol_profile=profile,
)
print(f"synthetic chain: {len(chain.options)} quotes")
assert len(chain.options) > 20

# ---------------------------------------------------------------------------
# 4. Backtest: daily delta-hedge of a 6M ATM call over a fresh path,
#    deltas from the differentiable pricer (jax.grad).
# ---------------------------------------------------------------------------
S0 = float(prices[-1])
K, T, RATE = S0, 0.5, 0.03
delta_fn = jax.jit(jax.grad(
    lambda s, tau: fourier_price(s, K, tau, RATE, 0.0,
                                 cal["theta"], cal["kappa"], cal["theta"],
                                 cal["sigma_v"], cal["rho"]),
    argnums=0))

key = jax.random.PRNGKey(99)
s = jnp.full((1,), S0)
v = jnp.full((1,), cal["theta"])
steps = int(T / DT)
premium = float(fourier_price(S0, K, T, RATE, 0.0, cal["theta"], cal["kappa"],
                              cal["theta"], cal["sigma_v"], cal["rho"]))
cash = premium  # sell the call, hedge it
pos = 0.0
for i in range(steps):
    tau = T - i * DT
    d = float(delta_fn(float(s[0]), tau))
    cash -= (d - pos) * float(s[0])          # rebalance
    pos = d
    s, v, key = qe_heston_step(s, v, DT, RATE, cal["kappa"], cal["theta"],
                               cal["sigma_v"], cal["rho"], key=key)
payoff = max(float(s[0]) - K, 0.0)
pnl = cash + pos * float(s[0]) - payoff
print(f"hedged P&L: {pnl:+.3f} (premium collected {premium:.3f})")
assert abs(pnl) < 0.15 * S0, "delta hedge keeps P&L far below naked exposure"
print("workflow OK")
