"""Gemini vision client for squad-screenshot recognition (optional; degrade gracefully).

Requires ``FPLAI_GEMINI_API_KEY`` (``settings.gemini_api_key``). One call per scan
sends the screenshot to ``gemini-3.7-flash`` and asks for the visible player cards
as structured JSON. The primary surface is the Interactions API (GA since June
2026); if the endpoint 404s (older API rollout) the call falls back once to the
legacy ``generateContent`` surface with the same schema.

Graceful degradation contract: no key configured raises :class:`ConfigurationError`
(the API layer maps it to 503); any transport/HTTP/parsing failure raises
:class:`RecognitionError` (mapped to 502). Nothing here touches disk or the
shared fetch throttle — screenshots are user-interactive, not pipeline batch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from fplai.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-3.7-flash"
TIMEOUT_S = 90.0

VALID_POSITIONS = {"GKP", "DEF", "MID", "FWD"}
#: Common on-screen position labels -> roster position codes.
_POSITION_ALIASES = {
    "GK": "GKP",
    "GOALKEEPER": "GKP",
    "DEFENDER": "DEF",
    "MIDFIELDER": "MID",
    "FORWARD": "FWD",
    "STRIKER": "FWD",
}


class ConfigurationError(RuntimeError):
    """Raised when the scanner is used without a Gemini API key configured."""


class RecognitionError(RuntimeError):
    """Raised when Gemini is unreachable or answers something unusable."""


@dataclass(frozen=True)
class SeenPlayer:
    """One player card as read off the screenshot (pre-matching)."""

    name: str
    club: str | None = None
    price: float | None = None  # £m as printed, e.g. 12.7
    position: str | None = None  # GKP/DEF/MID/FWD


PROMPT = """\
This is a screenshot of a Fantasy Premier League (FPL) squad — 15 player cards laid
out as goalkeeper/defender/midfielder/forward rows plus a bench. For EVERY player
card visible, extract:

- name: the player name exactly as printed on the card (e.g. "M.Salah", "Van Dijk").
- position: which row/section the card sits in — one of GKP, DEF, MID, FWD.
  The bench lists a position label per player; use it.
- price: the price in millions if printed on the card (e.g. 12.7 for "£12.7").
  Omit if the card shows points or a fixture instead of a price.
- club: the player's OWN club as a short name/abbreviation, only if you can tell
  from the shirt or badge. IMPORTANT: the small text under the name is usually the
  upcoming FIXTURE (the opponent, e.g. "MCI (H)") — never report that as the club.
  Omit the field when unsure.

Return every card you can see, in reading order. Do not invent players; if a card
is cut off or unreadable, skip it."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "players": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "position": {"type": "string"},
                    "price": {"type": "number"},
                    "club": {"type": "string"},
                },
                "required": ["name"],
            },
        }
    },
    "required": ["players"],
}


def recognize_squad(
    image_base64: str, mime_type: str, *, api_key: str | None = None
) -> list[SeenPlayer]:
    """Read the player cards off a squad screenshot via Gemini.

    ``image_base64`` is the raw base64 payload (no ``data:`` prefix).
    """
    key = api_key if api_key is not None else settings.gemini_api_key
    if not key:
        raise ConfigurationError(
            "Gemini API key not configured — set FPLAI_GEMINI_API_KEY in .env "
            "and restart the backend"
        )
    data = _call_gemini(key, image_base64, mime_type)
    text = _extract_text(data)
    return _parse_players(text)


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


def _call_gemini(key: str, image_base64: str, mime_type: str) -> dict[str, Any]:
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    interactions_body = {
        "model": MODEL,
        "store": False,
        "input": [
            {"type": "text", "text": PROMPT},
            {"type": "image", "data": image_base64, "mime_type": mime_type},
        ],
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": RESPONSE_SCHEMA,
        },
        "generation_config": {"thinking_level": "low"},
    }
    try:
        resp = httpx.post(
            f"{BASE_URL}/interactions",
            headers=headers,
            json=interactions_body,
            timeout=TIMEOUT_S,
        )
        if resp.status_code == 404:
            # Interactions surface not available — legacy generateContent fallback.
            logger.warning("interactions endpoint 404 — falling back to generateContent")
            resp = httpx.post(
                f"{BASE_URL}/models/{MODEL}:generateContent",
                headers=headers,
                json=_legacy_body(image_base64, mime_type),
                timeout=TIMEOUT_S,
            )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:300]
        raise RecognitionError(
            f"Gemini answered HTTP {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RecognitionError(f"Gemini unreachable: {exc}") from exc


def _legacy_body(image_base64: str, mime_type: str) -> dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }


# --------------------------------------------------------------------------------------
# Response parsing (tolerates both API surfaces)
# --------------------------------------------------------------------------------------


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the model's text out of an Interactions or generateContent response."""
    # Interactions: top-level convenience field.
    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    # Interactions: walk the steps timeline for the last text content.
    for step in reversed(data.get("steps") or []):
        for part in reversed(step.get("content") or []):
            part_text = part.get("text") if isinstance(part, dict) else None
            if isinstance(part_text, str) and part_text.strip():
                return part_text
    # Legacy generateContent: candidates[0].content.parts[*].text.
    for cand in data.get("candidates") or []:
        parts = ((cand.get("content") or {}).get("parts")) or []
        joined = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if joined.strip():
            return joined
    raise RecognitionError("Gemini response contained no text output")


def _parse_players(text: str) -> list[SeenPlayer]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a ```json ... ``` fence if the model added one despite the schema.
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RecognitionError(f"Gemini answered non-JSON: {cleaned[:200]}") from exc
    rows = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RecognitionError("Gemini JSON is missing the players array")
    out: list[SeenPlayer] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out.append(
            SeenPlayer(
                name=name,
                club=(str(row["club"]).strip() or None) if row.get("club") else None,
                price=_as_price(row.get("price")),
                position=_as_position(row.get("position")),
            )
        )
    if not out:
        raise RecognitionError("Gemini recognized no player cards in the image")
    return out


def _as_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    # Screenshots print prices in £m (3.8–15.5); anything else is misread noise.
    return price if 3.0 <= price <= 20.0 else None


def _as_position(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    pos = value.strip().upper()
    pos = _POSITION_ALIASES.get(pos, pos)
    return pos if pos in VALID_POSITIONS else None
