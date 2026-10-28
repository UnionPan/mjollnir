"""Package exports resolved on first use (PEP 562).

A package lists, for each public name, the submodule that defines it; the
submodule (and whatever optional dependency it needs) loads only when the name
is first touched, so importing the package stays cheap and works without the
extras its heavier members need.

Usage, in a package's ``__init__``::

    from mjollnir._lazy import lazy_exports

    __getattr__, __dir__, __all__ = lazy_exports(__name__, globals(), {
        "YFinanceFetcher": ".yfinance_fetcher",
    })
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def lazy_exports(package: str, namespace: dict[str, Any], names: Mapping[str, str],
                 submodules: frozenset[str] | set[str] = frozenset()):
    """``__getattr__``, ``__dir__`` and ``__all__`` for ``package``.

    ``names`` maps each public name to the module (relative to ``package``)
    that defines it; ``submodules`` are exposed as attributes too. A resolved
    value is cached in ``namespace``, so each name loads once.
    """
    exported = [*names, *sorted(submodules)]

    def __getattr__(name: str) -> Any:
        if name in names:
            value = getattr(importlib.import_module(names[name], package), name)
        elif name in submodules:
            value = importlib.import_module(f".{name}", package)
        else:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*namespace, *exported})

    return __getattr__, __dir__, exported
