"""Trained posterior networks as versioned, hash-verified artifacts.

An artifact is two files in a directory named after the model:
``params.msgpack`` (the network's weights, flax serialization) and
``meta.json`` (everything needed to use and to trust it: the model and its
parameter names, the training prior, standardization statistics, the network
shape, the offline validation, and the SHA-256 of the weights). Loading
refuses weights whose hash does not match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from mjollnir.inference.models import MODELS, InferenceModel
from mjollnir.inference.network import MixtureDensityNetwork

__all__ = ["NPEArtifact", "load_artifact", "save_artifact"]

SCHEMA_VERSION = 1


class ArtifactIntegrityError(RuntimeError):
    """The weights on disk are not the ones the metadata describes."""


@dataclass
class NPEArtifact:
    model: InferenceModel
    params: dict
    feature_mean: np.ndarray
    feature_std: np.ndarray
    theta_mean: np.ndarray
    theta_std: np.ndarray
    hidden: tuple[int, ...]
    n_components: int
    validation: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    created_at: str = ""
    mjollnir_version: str = ""
    weights_sha256: str = ""

    @property
    def network(self) -> MixtureDensityNetwork:
        return MixtureDensityNetwork(hidden=self.hidden, n_components=self.n_components,
                                     n_outputs=self.model.dim)


def save_artifact(artifact: NPEArtifact, directory: str | Path) -> Path:
    directory = Path(directory) / artifact.model.name
    directory.mkdir(parents=True, exist_ok=True)
    weights = serialization.to_bytes(artifact.params)
    (directory / "params.msgpack").write_bytes(weights)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "model": artifact.model.name,
        "params": list(artifact.model.params),
        "prior_low": list(artifact.model.low),
        "prior_high": list(artifact.model.high),
        "feature_mean": np.asarray(artifact.feature_mean).tolist(),
        "feature_std": np.asarray(artifact.feature_std).tolist(),
        "theta_mean": np.asarray(artifact.theta_mean).tolist(),
        "theta_std": np.asarray(artifact.theta_std).tolist(),
        "hidden": list(artifact.hidden),
        "n_components": artifact.n_components,
        "validation": artifact.validation,
        "training": artifact.training,
        "created_at": artifact.created_at,
        "mjollnir_version": artifact.mjollnir_version,
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory


def load_artifact(directory: str | Path) -> NPEArtifact:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    weights = (directory / "params.msgpack").read_bytes()
    if hashlib.sha256(weights).hexdigest() != meta["weights_sha256"]:
        raise ArtifactIntegrityError(f"{directory}: weights do not match their recorded hash")
    model = MODELS[meta["model"]]
    if list(model.params) != meta["params"] or list(model.low) != meta["prior_low"] \
            or list(model.high) != meta["prior_high"]:
        raise ArtifactIntegrityError(f"{directory}: trained for a different {model.name} specification")
    artifact = NPEArtifact(
        model=model, params={}, feature_mean=np.asarray(meta["feature_mean"]),
        feature_std=np.asarray(meta["feature_std"]), theta_mean=np.asarray(meta["theta_mean"]),
        theta_std=np.asarray(meta["theta_std"]), hidden=tuple(meta["hidden"]),
        n_components=meta["n_components"], validation=meta["validation"], training=meta["training"],
        created_at=meta["created_at"], mjollnir_version=meta["mjollnir_version"],
        weights_sha256=meta["weights_sha256"],
    )
    template = artifact.network.init(jax.random.PRNGKey(0), np.zeros((1, len(artifact.feature_mean))))
    artifact.params = serialization.from_bytes(template, weights)
    return artifact
