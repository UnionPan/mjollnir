"""The failures a fit can meet that are not bugs, for code that records them and moves on.

A universe run fits hundreds of assets: one asset whose fit fails numerically
is recorded as failed, and the run continues. Anything else (a TypeError, an
AttributeError) is a bug and propagates.
"""

from __future__ import annotations

# ValueError includes numpy's LinAlgError; RuntimeError includes XLA runtime
# errors and scipy's QhullError; ArithmeticError covers overflow and division.
NUMERICAL_ERRORS: tuple[type[BaseException], ...] = (ValueError, ArithmeticError, RuntimeError)
