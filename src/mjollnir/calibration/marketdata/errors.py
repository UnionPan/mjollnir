"""What a market-data request can fail with, besides bugs.

Network failures (requests' errors are OSErrors), yfinance's own errors, and
the parsing failures a malformed or empty response causes.
"""

from __future__ import annotations

from yfinance.exceptions import YFException

FETCH_ERRORS: tuple[type[BaseException], ...] = (OSError, YFException, ValueError, KeyError, IndexError)
# a quote row with a missing or non-numeric field
ROW_ERRORS: tuple[type[BaseException], ...] = (ValueError, TypeError, KeyError)
