"""Versioned, integrity-checked calibrated-parameter artifacts.

The calibration pipeline stores bulk results as per-run parquet frames; this
module provides the complementary *artifact* format: a single, frozen,
hash-verified parameter set that an experiment or backtest can pin, load, and
trust — the parameter analogue of the golden-value contract on the kernel.

Design goals:

* **Self-describing** — model, measure (P/Q), asset, estimation window,
  free-text source, library version, schema version.
* **Tamper-evident** — a sha256 over the canonical JSON content; ``load``
  refuses silently edited files.
* **Provenance-chained** — deriving a new set (scenario shocks, manual
  tweaks) records the parent's hash, so any counterfactual traces back to
  the calibration that spawned it.

Example::

    ps = ParamSet.create("heston", "P", {"kappa": 2.0, "theta": 0.04, ...},
                         asset="SPY", source="qmle close-close 2018-2026")
    ps.save("data/params/spy_heston.json")
    ps2 = ParamSet.load("data/params/spy_heston.json")   # verifies hash
    shocked = ps2.derive(scale={"theta": 4.0}, note="vol crisis")
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

SCHEMA_VERSION = 1


class ParamSetIntegrityError(RuntimeError):
    """Raised when a stored ParamSet fails hash verification on load."""


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class ParamSet:
    """A frozen, versioned set of calibrated model parameters."""

    model: str                    # e.g. "heston", "merton", "bates"
    measure: str                  # "P" (physical) or "Q" (risk-neutral)
    params: Mapping[str, float]
    asset: str | None = None
    window: str | None = None     # e.g. "2018-01-02..2026-01-02"
    source: str | None = None     # estimator + data description
    note: str | None = None
    parent_hash: str | None = None
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION
    mjollnir_version: str = ""
    _extra: Mapping[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        model: str,
        measure: str,
        params: Mapping[str, float],
        **meta,
    ) -> ParamSet:
        """Build a ParamSet with library version + UTC timestamp stamped in."""
        from mjollnir import __version__

        if measure not in ("P", "Q"):
            raise ValueError(f"measure must be 'P' or 'Q', got {measure!r}")
        clean = {k: float(v) for k, v in dict(params).items()}
        return cls(
            model=model,
            measure=measure,
            params=MappingProxyType(clean),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            mjollnir_version=__version__,
            **meta,
        )

    @classmethod
    def from_frame_row(cls, model: str, measure: str, row, *,
                       param_names: list[str], asset_col: str = "ticker",
                       **meta) -> ParamSet:
        """Bridge from a calibration-pipeline results frame (one row)."""
        params = {name: float(row[name]) for name in param_names}
        asset = str(row[asset_col]) if asset_col in row else None
        return cls.create(model, measure, params, asset=asset, **meta)

    # ------------------------------------------------------------------
    # integrity
    # ------------------------------------------------------------------
    def _content(self) -> dict:
        return {
            "model": self.model,
            "measure": self.measure,
            "params": dict(self.params),
            "asset": self.asset,
            "window": self.window,
            "source": self.source,
            "note": self.note,
            "parent_hash": self.parent_hash,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "mjollnir_version": self.mjollnir_version,
        }

    def content_hash(self) -> str:
        return hashlib.sha256(_canonical(self._content())).hexdigest()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self._content()
        doc["content_hash"] = self.content_hash()
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> ParamSet:
        doc = json.loads(Path(path).read_text())
        stored_hash = doc.pop("content_hash", None)
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ParamSetIntegrityError(
                f"{path}: schema_version {doc.get('schema_version')} != {SCHEMA_VERSION}"
            )
        ps = cls(
            model=doc["model"],
            measure=doc["measure"],
            params=MappingProxyType({k: float(v) for k, v in doc["params"].items()}),
            asset=doc.get("asset"),
            window=doc.get("window"),
            source=doc.get("source"),
            note=doc.get("note"),
            parent_hash=doc.get("parent_hash"),
            created_at=doc.get("created_at", ""),
            schema_version=doc["schema_version"],
            mjollnir_version=doc.get("mjollnir_version", ""),
        )
        if verify and stored_hash != ps.content_hash():
            raise ParamSetIntegrityError(
                f"{path}: content hash mismatch — file was modified after save "
                f"(stored {stored_hash!r}, computed {ps.content_hash()!r})"
            )
        return ps

    # ------------------------------------------------------------------
    # derivation (provenance-chained)
    # ------------------------------------------------------------------
    def derive(
        self,
        *,
        scale: Mapping[str, float] | None = None,
        set_: Mapping[str, float] | None = None,
        note: str | None = None,
    ) -> ParamSet:
        """Return a new ParamSet with shocked parameters.

        ``scale`` multiplies parameters; ``set_`` overrides them absolutely.
        The child records this set's hash as ``parent_hash``, so counterfactual
        parameter sets always trace back to the calibration that spawned them.
        """
        new = dict(self.params)
        for k, m in (scale or {}).items():
            if k not in new:
                raise KeyError(f"cannot scale unknown parameter {k!r}")
            new[k] = new[k] * float(m)
        for k, v in (set_ or {}).items():
            new[k] = float(v)
        return replace(
            self,
            params=MappingProxyType(new),
            parent_hash=self.content_hash(),
            note=note if note is not None else self.note,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def as_kwargs(self) -> dict[str, float]:
        """Plain dict view, e.g. ``qe_heston_step(..., **ps.as_kwargs())``."""
        return dict(self.params)


__all__ = ["SCHEMA_VERSION", "ParamSet", "ParamSetIntegrityError"]
