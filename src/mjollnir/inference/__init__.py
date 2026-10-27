"""Amortized Bayesian calibration: posterior networks trained offline, evaluated online.

A model's posterior given a price history is learned once, offline, from
millions of simulated histories (``train``); calibrating a real history is
then a forward pass of fixed cost (``posterior.fit``). Every artifact ships
with its offline validation (coverage, contraction, recovery error), so a
posterior can say how far to trust it.

See :mod:`mjollnir.inference.models` for what is inferred per model.
"""

from mjollnir.inference.artifact import NPEArtifact, load_artifact, save_artifact
from mjollnir.inference.models import MODELS, InferenceModel
from mjollnir.inference.posterior import GaussianPrior, Posterior, fit

__all__ = ["MODELS", "GaussianPrior", "InferenceModel", "NPEArtifact", "Posterior", "fit", "load_artifact",
           "save_artifact"]
