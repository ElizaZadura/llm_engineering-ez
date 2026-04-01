"""
Build curated JSONL records from Amazon Reviews 2023 raw meta rows.

Target line shape:
{"title": "...", "description": "...", "price": 79.99, "bucket": "50-100"}
"""

from __future__ import annotations

import json
import re
from typing import Any

# Align with parser.parse price gate
MIN_PRICE = 0.5
MAX_PRICE = 999.49

MIN_TITLE_LEN = 3
MIN_DESCRIPTION_LEN = 10
MAX_DESCRIPTION_LEN = 4000

# Match long alphanumeric product codes (same idea as parser.scrub)
_SKU_PATTERN = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")


def price_bucket(price: float) -> str:
    if price < 10:
        return "0-10"
    if price < 25:
        return "10-25"
    if price < 50:
        return "25-50"
    if price < 100:
        return "50-100"
    return "100+"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()


def _stringify_chunks(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_whitespace(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if item is None:
                continue
            s = _normalize_whitespace(str(item))
            if s:
                parts.append(s)
        return _normalize_whitespace(" ".join(parts))
    return _normalize_whitespace(str(value))


def build_description(row: dict[str, Any]) -> str:
    desc = _stringify_chunks(row.get("description"))
    if desc:
        return desc[:MAX_DESCRIPTION_LEN]
    features = row.get("features")
    return _stringify_chunks(features)[:MAX_DESCRIPTION_LEN]


def _scrub_skus(text: str) -> str:
    return _normalize_whitespace(_SKU_PATTERN.sub("", text))


def amazon_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Validate and map one HF meta row to a JSON-serializable dict, or None if unusable.
    """
    title = row.get("title")
    if not title or not isinstance(title, str):
        return None
    title = _scrub_skus(title)
    if len(title) < MIN_TITLE_LEN:
        return None

    try:
        price = float(row["price"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    description = _scrub_skus(build_description(row))
    if len(description) < MIN_DESCRIPTION_LEN:
        return None

    return {
        "title": title,
        "description": description,
        "price": round(price, 2),
        "bucket": price_bucket(price),
    }
