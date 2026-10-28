"""What a market-data request can fail with, besides bugs.

Network failures (requests' errors are OSErrors), yfinance's own errors, and
the parsing failures a malformed or empty response causes.
"""

from __future__ import annotations

try:
    from yfinance.exceptions import YFException
except ModuleNotFoundError as error:
    # yfinance is the [data] extra; without it no request can raise its errors
    if error.name != "yfinance":
        raise
    _YFINANCE_ERRORS: tuple[type[BaseException], ...] = ()
else:
    _YFINANCE_ERRORS = (YFException,)

FETCH_ERRORS: tuple[type[BaseException], ...] = (OSError, *_YFINANCE_ERRORS, ValueError, KeyError, IndexError)
# a quote row with a missing or non-numeric field
ROW_ERRORS: tuple[type[BaseException], ...] = (ValueError, TypeError, KeyError)
