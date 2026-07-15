"""RNG contract for the numpy process layer (v0.2 Generator migration).

Three guarantees, none of which held under the old global-``np.random.seed``
pattern:

1. **Determinism**: same ``random_seed`` -> identical paths, per process.
2. **Isolation**: simulating process B between two identical runs of
   process A cannot change A's output (no shared global stream).
3. **No global stomping**: running a seeded simulation leaves the ambient
   ``np.random`` state untouched.
"""

import numpy as np
import pytest

from mjollnir.processes import (
    GBM,
    NIG,
    Heston,
    KouJD,
    MertonJD,
    SimulationConfig,
    VarianceGamma,
)


def _cfg(seed):
    return SimulationConfig(n_paths=4, n_steps=32, random_seed=seed)


def _make(cls):
    # minimal sane constructor args per process family
    if cls is GBM:
        return cls(mu=0.05, sigma=0.2)
    if cls is Heston:
        return cls(mu=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
    if cls is MertonJD:
        return cls(mu=0.05, sigma=0.2, lambda_jump=0.5, mu_J=-0.05, sigma_J=0.1)
    if cls is KouJD:
        return cls(mu=0.05, sigma=0.2, lambda_jump=0.5, p=0.4,
                   eta_up=10.0, eta_down=5.0)
    if cls is VarianceGamma:
        return cls(theta=-0.14, sigma=0.2, nu=0.2)
    if cls is NIG:
        return cls(alpha=15.0, beta=-5.0, delta=0.5, mu=0.05)
    raise AssertionError(cls)


def _x0(cls):
    return np.array([[100.0, 0.04]]) if cls is Heston else np.array([[100.0]])


PROCESSES = [GBM, Heston, MertonJD, KouJD, VarianceGamma, NIG]


@pytest.mark.parametrize("cls", PROCESSES, ids=lambda c: c.__name__)
def test_seeded_determinism(cls):
    proc = _make(cls)
    _, p1 = proc.simulate(X0=_x0(cls), T=0.25, config=_cfg(123))
    _, p2 = proc.simulate(X0=_x0(cls), T=0.25, config=_cfg(123))
    np.testing.assert_array_equal(p1, p2)
    _, p3 = proc.simulate(X0=_x0(cls), T=0.25, config=_cfg(124))
    assert not np.array_equal(p1, p3)


def test_cross_process_isolation():
    """Interleaving another simulation must not perturb a seeded run —
    the failure mode of the old global-seed pattern."""
    merton = _make(MertonJD)
    _, before = merton.simulate(X0=_x0(MertonJD), T=0.25, config=_cfg(7))

    # interleave: a different process, unseeded, chews randomness
    _make(Heston).simulate(X0=_x0(Heston), T=0.25, config=_cfg(None))

    _, after = merton.simulate(X0=_x0(MertonJD), T=0.25, config=_cfg(7))
    np.testing.assert_array_equal(before, after)


def test_no_global_stream_stomping():
    """A seeded simulation must not touch the ambient np.random stream."""
    np.random.seed(42)
    expected = np.random.get_state()[1][:8].copy()

    np.random.seed(42)
    _make(KouJD).simulate(X0=_x0(KouJD), T=0.25, config=_cfg(99))
    actual = np.random.get_state()[1][:8]

    np.testing.assert_array_equal(expected, actual)


def test_unseeded_runs_differ():
    proc = _make(GBM)
    _, p1 = proc.simulate(X0=_x0(GBM), T=0.25, config=_cfg(None))
    _, p2 = proc.simulate(X0=_x0(GBM), T=0.25, config=_cfg(None))
    assert not np.array_equal(p1, p2)
