"""Harvest candidate queries from real user search behaviour.

Provenance matters more than volume here. Search-engine autocomplete returns
aggregated *real* queries — what people actually typed — which is the closest
legitimate substitute for production logs this project can obtain. It is
labelled as such in the dataset rather than passed off as production traffic.

The harvest is deliberately broad and noisy; `label.py` classifies and filters.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable

import httpx

log = logging.getLogger(__name__)

SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36"

# Seeds spanning the catalogue's product families and the ways customers frame
# a purchase: occasion, recipient, form factor, budget, logistics.
SEEDS: tuple[str, ...] = (
    "hoa tặng",
    "hoa tặng mẹ",
    "hoa tặng người yêu",
    "hoa tặng sinh nhật",
    "bó hoa sinh nhật",
    "giỏ hoa",
    "kệ hoa khai trương",
    "hoa khai trương",
    "hoa chia buồn",
    "hoa đám tang",
    "hoa cưới",
    "hoa lan hồ điệp",
    "hoa hồng",
    "hoa cẩm tú cầu",
    "hoa tulip",
    "chậu lan",
    "đặt hoa online",
    "mua hoa tươi",
    "shop hoa tươi",
    "giao hoa tận nơi",
    "hoa tươi giá rẻ",
    "hoa 500k",
    "hoa 1 triệu",
)

# The eight canned questions the original app shipped in its sidebar. Included
# because they are the shop author's own model of customer intent — and because
# several of them ("có freeship không") are exactly the out-of-scope questions
# the legacy system answered with five unrelated bouquets.
SIDEBAR_QUERIES: tuple[str, ...] = (
    "shop có giao nhanh không",
    "có freeship không",
    "hoa sinh nhật đẹp nhất",
    "hoa khai trương giá bao nhiêu",
    "tư vấn hoa tặng người yêu",
    "hoa chia buồn trang trọng",
    "đặt hoa theo yêu cầu được không",
    "hoa cưới cầm tay",
)


def _suggest(client: httpx.Client, query: str, lang: str = "vi") -> list[str]:
    try:
        resp = client.get(
            SUGGEST_URL,
            params={"client": "firefox", "hl": lang, "gl": "vn", "q": query},
        )
        resp.raise_for_status()
        data = json.loads(resp.text)
        return list(data[1]) if len(data) > 1 else []
    except Exception as exc:
        log.warning("autocomplete failed for %r: %s", query, exc)
        return []


def harvest_autocomplete(
    seeds: Iterable[str] = SEEDS,
    *,
    expand: bool = True,
    delay: float = 0.3,
) -> list[str]:
    """Collect autocomplete suggestions, optionally one level deep.

    Expansion (suggestions-of-suggestions) is what surfaces the long tail —
    budget phrasings like 'hoa khai trương 500k' and out-of-area requests like
    'hoa chia buồn đà nẵng' that make the out-of-scope slice realistic instead
    of invented.
    """
    seen: set[str] = set()
    out: list[str] = []

    with httpx.Client(headers={"User-Agent": UA}, timeout=20, follow_redirects=True) as client:
        first_round = []
        for seed in seeds:
            for s in _suggest(client, seed):
                key = s.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(s.strip())
                    first_round.append(s.strip())
            time.sleep(delay)

        log.info("Autocomplete round 1: %d unique suggestions", len(out))

        if expand:
            for s in first_round:
                for s2 in _suggest(client, s):
                    key = s2.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        out.append(s2.strip())
                time.sleep(delay)
            log.info("Autocomplete round 2: %d unique suggestions total", len(out))

    return out
