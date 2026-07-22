"""the-odds-api.com v4 client (optional enrichment source; degrade gracefully).

Requires ``FPLAI_ODDS_API_KEY`` (``settings.odds_api_key``). The free tier gives
500 credits/month; a request to an odds endpoint costs ``markets × regions``
credits (the ``/sports`` list is free). The planned budget (MODEL_DESIGN_INPUTS
§5.2) is one daily ``h2h+totals`` uk snapshot ≈ 2 credits/day.

Graceful degradation contract: when no API key is configured every public method
logs a warning and returns ``None`` — the pipeline treats this source as
optional. :class:`ConfigurationError` is only raised if the low-level request
path is reached without a key (a programming error, not a pipeline condition).

Quota accounting: the API reports ``x-requests-remaining``, ``x-requests-used``
and ``x-requests-last`` response headers; they are tracked on
:attr:`TheOddsApiClient.quota` after every call, when the shared fetch helper
exposes response headers (it may return raw bytes from its disk cache, in which
case the fields stay at their last known values).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fplai.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
EPL_SPORT_KEY = "soccer_epl"
DEFAULT_MARKETS = "h2h,totals"
DEFAULT_REGIONS = "uk"


class ConfigurationError(RuntimeError):
    """Raised when the-odds-api is used without an API key configured."""


@dataclass
class QuotaStatus:
    """Last-seen credit accounting from the-odds-api response headers."""

    remaining: float | None = None  # x-requests-remaining
    used: float | None = None  # x-requests-used
    last_cost: float | None = None  # x-requests-last
    spent_estimate: float = field(default=0.0)  # sum of estimated costs this process


def estimate_cost(markets: str = DEFAULT_MARKETS, regions: str = DEFAULT_REGIONS) -> int:
    """Estimated credit cost of one odds call: ``n_markets × n_regions``."""
    n_markets = len([m for m in markets.split(",") if m.strip()])
    n_regions = len([r for r in regions.split(",") if r.strip()])
    return max(1, n_markets) * max(1, n_regions)


def _split_payload(resp: object) -> tuple[bytes, Any]:
    """Normalise the fetch helper's return to ``(body_bytes, headers_or_None)``."""
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp), None
    if isinstance(resp, str):
        return resp.encode("utf-8"), None
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content), getattr(resp, "headers", None)
    raise TypeError(f"polite_get returned unsupported type {type(resp)!r}")


@retry(
    retry=retry_if_exception_type((httpx.TransportError, ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _fetch(url: str, *, min_interval_s: float = 1.0, cache_ttl_s: float = 0.0) -> tuple[bytes, Any]:
    """Fetch through the shared throttled helper; uncached by default (live odds)."""
    from fplai.data.fpl_api import polite_get  # deferred: module owned by another agent

    return _split_payload(polite_get(url, min_interval_s=min_interval_s, cache_ttl_s=cache_ttl_s))


class TheOddsApiClient:
    """Minimal the-odds-api v4 client with credit-cost accounting.

    All methods return ``None`` (with a logged warning) when no API key is set.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Args: api_key: overrides ``settings.odds_api_key`` (may be empty)."""
        self.api_key = api_key if api_key is not None else settings.odds_api_key
        self.quota = QuotaStatus()

    @property
    def enabled(self) -> bool:
        """Whether an API key is configured."""
        return bool(self.api_key)

    def _warn_disabled(self, method: str) -> None:
        logger.warning(
            "the-odds-api %s skipped: no FPLAI_ODDS_API_KEY configured (source is optional)",
            method,
        )

    def _update_quota(self, headers: Any, estimated: float) -> None:
        self.quota.spent_estimate += estimated
        if headers is None:
            return
        for attr, header in (
            ("remaining", "x-requests-remaining"),
            ("used", "x-requests-used"),
            ("last_cost", "x-requests-last"),
        ):
            raw = headers.get(header) if hasattr(headers, "get") else None
            if raw is not None:
                try:
                    setattr(self.quota, attr, float(raw))
                except (TypeError, ValueError):
                    logger.debug("unparseable %s header: %r", header, raw)

    def _get(self, path: str, params: dict[str, str], *, estimated_cost: float) -> Any:
        """Perform a GET against the v4 API and return the decoded JSON body."""
        if not self.enabled:
            raise ConfigurationError(
                "the-odds-api request attempted without an API key; set FPLAI_ODDS_API_KEY"
            )
        query = urlencode({"apiKey": self.api_key, **params})
        body, headers = _fetch(f"{BASE_URL}/{path}?{query}")
        self._update_quota(headers, estimated_cost)
        return json.loads(body)

    def sports(self) -> list[dict[str, Any]] | None:
        """List available sports (free — costs no credits). ``None`` if no key."""
        if not self.enabled:
            self._warn_disabled("sports()")
            return None
        return self._get("sports", {}, estimated_cost=0.0)

    def epl_odds(
        self,
        markets: str = DEFAULT_MARKETS,
        regions: str = DEFAULT_REGIONS,
    ) -> list[dict[str, Any]] | None:
        """Current EPL odds, one dict per upcoming event. ``None`` if no key.

        Costs ``markets × regions`` credits (default h2h,totals × uk = 2); the
        estimate is accumulated on :attr:`quota` and reconciled against the
        ``x-requests-*`` headers when available.
        """
        if not self.enabled:
            self._warn_disabled("epl_odds()")
            return None
        cost = estimate_cost(markets, regions)
        logger.info("the-odds-api epl_odds markets=%s regions=%s (~%d credits)",
                    markets, regions, cost)
        return self._get(
            f"sports/{EPL_SPORT_KEY}/odds",
            {
                "regions": regions,
                "markets": markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            estimated_cost=cost,
        )
