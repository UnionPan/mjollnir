"""Tests for the mjollnir-params CLI (main() called directly)."""

import json

import pytest

from mjollnir.cli import main
from mjollnir.params import ParamSet

HESTON = {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7, "v0": 0.04}


@pytest.fixture
def artifact(tmp_path):
    ps = ParamSet.create("heston", "P", HESTON, asset="SPY")
    return str(ps.save(tmp_path / "spy.json"))


def test_show(artifact, capsys):
    assert main(["show", artifact]) == 0
    out = capsys.readouterr().out
    assert "heston" in out and "SPY" in out and "kappa" in out


def test_verify_ok_and_tampered(artifact, tmp_path, capsys):
    assert main(["verify", artifact]) == 0
    doc = json.loads(open(artifact).read())
    doc["params"]["theta"] = 9.9
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    assert main(["verify", str(bad)]) == 1
    assert main(["verify", artifact, str(bad)]) == 1   # any failure -> nonzero


def test_derive(artifact, tmp_path):
    out = tmp_path / "crisis.json"
    assert main(["derive", artifact, "--scale", "theta=4", "--set", "rho=-0.9",
                 "--note", "crisis", "-o", str(out)]) == 0
    child = ParamSet.load(out)
    assert child.params["theta"] == pytest.approx(0.16)
    assert child.params["rho"] == -0.9
    assert child.parent_hash == ParamSet.load(artifact).content_hash()


def test_scenario(artifact, tmp_path):
    out = tmp_path / "spike.json"
    assert main(["scenario", artifact, "vol_spike", "--severity", "2.0",
                 "-o", str(out)]) == 0
    child = ParamSet.load(out)
    assert child.params["v0"] == pytest.approx(0.16)


def test_unknown_scenario(artifact, tmp_path):
    with pytest.raises(SystemExit):
        main(["scenario", artifact, "nope", "-o", str(tmp_path / "x.json")])


def test_bad_kv(artifact, tmp_path):
    with pytest.raises(SystemExit):
        main(["derive", artifact, "--scale", "theta", "-o", str(tmp_path / "x.json")])
