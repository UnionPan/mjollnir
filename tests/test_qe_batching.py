"""Batched QE steps must draw independent noise per path.

The kernels accept array-valued state, and the documented usage passes a batch
with a single PRNG key. Every element of that batch is a separate path, so each
must receive its own noise; a scalar draw broadcast across the batch makes all
"paths" identical. Scalar and vmapped calls are covered by the golden parity
suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.jax import (
    configure_runtime,
    qe_bates_step,
    qe_four_half_step,
    qe_heston_step,
    qe_three_half_step,
)

configure_runtime()

N_PATHS = 100_000
S0, V0, DT, MU = 100.0, 0.04, 1 / 52, 0.02
HESTON = {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7}

KERNELS = {
    "heston": (qe_heston_step, HESTON),
    "bates": (qe_bates_step, {**HESTON, "lambda_j": 0.5, "mu_j": -0.05, "sigma_j": 0.1}),
    "three_half": (qe_three_half_step, {"kappa": 2.0, "theta": 0.04, "sigma_v": 2.0, "rho": -0.7}),
    "four_half": (qe_four_half_step, {**HESTON, "a_coef": 0.8, "b_coef": 0.2 * V0}),
}


@pytest.fixture(params=sorted(KERNELS), scope="module")
def one_step(request):
    step, params = KERNELS[request.param]
    spot, var, _ = step(
        jnp.full((N_PATHS,), S0), jnp.full((N_PATHS,), V0),
        dt=DT, mu=MU, **params, key=jax.random.PRNGKey(7),
    )
    return np.asarray(spot), np.asarray(var)


def test_batched_paths_are_distinct(one_step):
    spot, var = one_step
    assert np.unique(spot).size > 0.99 * N_PATHS
    assert np.unique(var).size > 0.5 * N_PATHS  # QE exponential branch can hit exactly 0
    assert spot.std() > 0.1 * S0 * np.sqrt(V0 * DT)


def test_batched_spot_is_a_discounted_martingale(one_step):
    spot, _ = one_step
    discounted = spot * np.exp(-MU * DT)
    standard_error = discounted.std() / np.sqrt(N_PATHS)
    assert abs(discounted.mean() - S0) < 6 * standard_error
