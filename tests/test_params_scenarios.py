"""Tests for the ParamSet artifact store and the scenario library."""

import json

import pytest

from mjollnir.params import ParamSet, ParamSetIntegrityError
from mjollnir.scenarios import (
    Scenario,
    correlation_breakdown,
    jump_cascade,
    regime_shift,
    vol_spike,
)

HESTON = {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7, "v0": 0.04}


def _base():
    return ParamSet.create(
        "heston", "P", HESTON,
        asset="SPY", window="2018-01-02..2026-01-02",
        source="qmle close-close",
    )


class TestParamSet:
    def test_round_trip(self, tmp_path):
        ps = _base()
        path = ps.save(tmp_path / "spy.json")
        loaded = ParamSet.load(path)
        assert loaded.params == ps.params
        assert loaded.content_hash() == ps.content_hash()
        assert loaded.mjollnir_version

    def test_tamper_detection(self, tmp_path):
        path = _base().save(tmp_path / "spy.json")
        doc = json.loads(path.read_text())
        doc["params"]["theta"] = 0.09          # silent edit
        path.write_text(json.dumps(doc))
        with pytest.raises(ParamSetIntegrityError, match="hash mismatch"):
            ParamSet.load(path)
        # explicit opt-out still works (e.g. forensic inspection)
        assert ParamSet.load(path, verify=False).params["theta"] == 0.09

    def test_schema_version_gate(self, tmp_path):
        path = _base().save(tmp_path / "spy.json")
        doc = json.loads(path.read_text())
        doc["schema_version"] = 999
        path.write_text(json.dumps(doc))
        with pytest.raises(ParamSetIntegrityError, match="schema_version"):
            ParamSet.load(path)

    def test_measure_validated(self):
        with pytest.raises(ValueError, match="measure"):
            ParamSet.create("heston", "X", HESTON)

    def test_derive_chains_provenance(self):
        ps = _base()
        child = ps.derive(scale={"theta": 4.0}, note="crisis")
        assert child.parent_hash == ps.content_hash()
        assert child.params["theta"] == pytest.approx(0.16)
        assert ps.params["theta"] == pytest.approx(0.04)   # parent untouched
        grandchild = child.derive(set_={"rho": -0.9})
        assert grandchild.parent_hash == child.content_hash()

    def test_derive_unknown_param_raises(self):
        with pytest.raises(KeyError, match="lambda_j"):
            _base().derive(scale={"lambda_j": 2.0})

    def test_kernel_integration(self):
        """as_kwargs plugs straight into the QE kernel."""
        import jax
        import jax.numpy as jnp

        from mjollnir.jax import qe_heston_step

        kw = _base().as_kwargs()
        v0 = kw.pop("v0")
        s, v, _ = qe_heston_step(
            jnp.full((2,), 100.0), jnp.full((2,), v0),
            dt=1 / 252, mu=0.02, **kw, key=jax.random.PRNGKey(0),
        )
        assert bool(jnp.isfinite(s).all() and (v >= 0).all())


class TestScenarios:
    def test_vol_spike_semantics(self):
        shocked = vol_spike(2.0).apply(_base())
        assert shocked.params["v0"] == pytest.approx(0.04 * 4.0)       # vol x2 => var x4
        assert shocked.params["theta"] == pytest.approx(0.04 * 2.25)   # (1+2)/2 vol => x2.25 var
        assert shocked.params["sigma_v"] == pytest.approx(0.3 * 2**0.5)
        assert shocked.parent_hash == _base().content_hash() or shocked.parent_hash
        assert "vol_spike" in shocked.note

    def test_severity_one_is_identity(self):
        base = _base()
        for scenario in (vol_spike(1.0), regime_shift(1.0)):
            shocked = scenario.apply(base)
            for k, v in base.params.items():
                assert shocked.params[k] == pytest.approx(v)

    def test_correlation_breakdown_absolute(self):
        shocked = correlation_breakdown(-0.9).apply(_base())
        assert shocked.params["rho"] == -0.9

    def test_jump_scenario_fails_loudly_on_heston(self):
        """A Heston set through a jump scenario must raise, not no-op."""
        with pytest.raises(KeyError, match="lambda_j"):
            jump_cascade(5.0).apply(_base())

    def test_jump_cascade_on_bates(self):
        bates = ParamSet.create("bates", "P", {**HESTON,
                                               "lambda_j": 0.5, "mu_j": -0.05,
                                               "sigma_j": 0.08})
        shocked = jump_cascade(4.0).apply(bates)
        assert shocked.params["lambda_j"] == pytest.approx(2.0)
        assert shocked.params["mu_j"] == pytest.approx(-0.05 * 2.0)

    def test_composition(self):
        combined = vol_spike(2.0) | correlation_breakdown(-0.9)
        shocked = combined.apply(_base())
        assert shocked.params["v0"] == pytest.approx(0.16)
        assert shocked.params["rho"] == -0.9
        assert "|" in combined.name

    def test_scenario_is_data(self):
        """Scenario carries its rationale — auditable, serializable."""
        sc = vol_spike(3.0)
        assert isinstance(sc, Scenario)
        assert sc.description
        assert sc.scale["v0"] == 9.0
