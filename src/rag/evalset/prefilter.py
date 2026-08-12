"""Deterministic pre-filtering of harvested queries.

897 raw autocomplete suggestions is far more than the golden set needs, and
sending them all to an LLM classifier is both wasteful and — at 20 free-tier
requests per day per model (docs/decisions.md D7) — impossible.

This module does everything that does not require a language model: dedupe,
drop noise, and assign a *provisional* intent from keyword evidence. The LLM
then only has to verify a small, already-balanced candidate set, which cuts the
classification budget from ~45 calls to ~5.

Provisional labels are never trusted as final. They exist to make sampling
balanced before the expensive step.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict

from rag.evalset.schema import Intent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# The shop serves Ho Chi Minh City only ("Freeship nội thành"). A query naming
# any other province is out of catalogue — these come from real autocomplete
# data ("hoa chia buồn đà nẵng", "shop hoa tươi cần thơ") and are the most
# realistic unanswerable queries available, far better than invented ones.
OTHER_PROVINCES = {
    "hà nội",
    "ha noi",
    "đà nẵng",
    "da nang",
    "cần thơ",
    "can tho",
    "hải phòng",
    "hai phong",
    "nha trang",
    "huế",
    "hue",
    "hội an",
    "hoi an",
    "đà lạt",
    "da lat",
    "vũng tàu",
    "vung tau",
    "biên hòa",
    "bien hoa",
    "bình dương",
    "binh duong",
    "đồng nai",
    "dong nai",
    "long an",
    "tiền giang",
    "sóc trăng",
    "soc trang",
    "an giang",
    "kiên giang",
    "cà mau",
    "ca mau",
    "vinh",
    "thanh hóa",
    "thanh hoa",
    "nghệ an",
    "nghe an",
    "quảng ninh",
    "quang ninh",
    "hạ long",
    "ha long",
    "bắc ninh",
    "bac ninh",
    "hải dương",
    "hai duong",
    "nam định",
    "nam dinh",
    "buôn ma thuột",
    "pleiku",
    "quy nhơn",
    "quy nhon",
    "phan thiết",
    "phan thiet",
    "rạch giá",
    "tây ninh",
    "tay ninh",
    "bến tre",
    "ben tre",
    "vĩnh long",
    # Hanoi districts show up in autocomplete without the city name.
    "cầu giấy",
    "cau giay",
    "đống đa",
    "dong da",
    "ba đình",
    "ba dinh",
    "hoàn kiếm",
    "hoan kiem",
    "long biên",
    "long bien",
    "hà đông",
    "ha dong",
    "mỹ đình",
    "my dinh",
    "thanh xuân",
    "thanh xuan",
}

# Districts and areas inside HCMC — these are in-area, not out of catalogue.
HCM_AREAS = {
    "quận 1",
    "quận 2",
    "quận 3",
    "quận 4",
    "quận 5",
    "quận 6",
    "quận 7",
    "quận 8",
    "quận 9",
    "quận 10",
    "quận 11",
    "quận 12",
    "gò vấp",
    "go vap",
    "tân bình",
    "tan binh",
    "tân phú",
    "tan phu",
    "bình thạnh",
    "binh thanh",
    "phú nhuận",
    "phu nhuan",
    "thủ đức",
    "thu duc",
    "bình tân",
    "binh tan",
    "hồ chí minh",
    "ho chi minh",
    "hcm",
    "sài gòn",
    "sai gon",
    "tphcm",
    "hồ thị kỷ",
    "ho thi ky",
}

# Autocomplete surfaces competitor brands and non-commercial intents. Neither
# belongs in a product-search slice, but the second group is excellent
# off-topic material.
COMPETITOR_MARKERS = re.compile(
    r"(shophoadep|dalatflower|hoayeuthuong|flowerbox|bảo hân|bao han|\.com|\.vn|shopee|lazada|tiki)",
    re.IGNORECASE,
)
OFF_TOPIC_MARKERS = re.compile(
    r"(lớp \d|lop \d|bài đọc|bai doc|văn bản|van ban|soạn bài|soan bai|tập đọc|tap doc|"
    r"ý nghĩa|y nghia|gọi là gì|goi la gi|tiếng anh là gì|cách vẽ|cach ve|cách làm|cach lam|"
    r"cách gấp|cach gap|hình ảnh|hinh anh|hình nền|hinh nen|download|"
    # 'hoa giấy' = paper flowers (a craft, not a product here). Bare 'giấy' was
    # too greedy: it swallowed 'Cầu Giấy', a Hanoi district, which is an
    # out-of-catalogue query rather than an off-topic one.
    r"hoa giấy|hoa giay|giay nhun|origami|đan len|móc len|"
    r"bằng tiền|bang tien|bằng vải|bang vai)",
    re.IGNORECASE,
)
POLICY_MARKERS = re.compile(
    r"(freeship|free ship|ship|giao hàng|giao hang|giao nhanh|mấy giờ|may gio|"
    r"mở cửa|mo cua|đặt hàng|dat hang|thanh toán|thanh toan|hoàn tiền|hoan tien|"
    r"theo yêu cầu|theo yeu cau|đổi trả|doi tra|liên hệ|lien he|hotline)",
    re.IGNORECASE,
)

OCCASIONS = {
    "sinh nhật": "sinh nhật",
    "sinh nhat": "sinh nhật",
    "khai trương": "khai trương",
    "khai truong": "khai trương",
    "chia buồn": "chia buồn",
    "chia buon": "chia buồn",
    "đám tang": "chia buồn",
    "dam tang": "chia buồn",
    "viếng": "chia buồn",
    "cưới": "cưới",
    "cuoi": "cưới",
    "đám cưới": "cưới",
    "tốt nghiệp": "tốt nghiệp",
    "tot nghiep": "tốt nghiệp",
    "20/10": "20/10",
    "20-10": "20/10",
    "8/3": "8/3",
    "8-3": "8/3",
    "20/11": "20/11",
    "valentine": "valentine",
    "tết": "tết",
    "tet": "tết",
    "kỷ niệm": "kỷ niệm",
    "ky niem": "kỷ niệm",
    "chúc mừng": "chúc mừng",
    "chuc mung": "chúc mừng",
}

RECIPIENTS = {
    "mẹ": "mẹ",
    "me": "mẹ",
    "người yêu": "người yêu",
    "nguoi yeu": "người yêu",
    "bạn gái": "bạn gái",
    "ban gai": "bạn gái",
    "vợ": "vợ",
    "vo": "vợ",
    "nam": "nam",
    "nữ": "nữ",
    "nu": "nữ",
    # Bare "cô"/"thầy" are unusable: after diacritic folding "cô", "có" and
    # "cỡ" all become "co", so 'có freeship không' matched a teacher recipient.
    # "cô giáo" is out too — it folds to ["co","giao"], identical to "có giao"
    # ("do you deliver"), which turned a delivery-policy question into a gift
    # for a teacher. Only genuinely unambiguous forms survive.
    "thầy cô": "thầy cô",
    "thầy giáo": "thầy cô",
    "sếp": "sếp",
    "bạn thân": "bạn thân",
}

FLOWER_TYPES = {
    "hồng": "hồng",
    "hong": "hồng",
    "lan hồ điệp": "lan hồ điệp",
    "lan ho diep": "lan hồ điệp",
    "cẩm tú cầu": "cẩm tú cầu",
    "cam tu cau": "cẩm tú cầu",
    "tulip": "tulip",
    "hướng dương": "hướng dương",
    "huong duong": "hướng dương",
    "ly": "ly",
    "cúc": "cúc",
    "cuc": "cúc",
    "baby": "baby",
    "mao lương": "mao lương",
    "cát tường": "cát tường",
    "sen đá": "sen đá",
    "lan": "lan",
}

FORM_FACTORS = {
    "bó": "bó",
    "bo hoa": "bó",
    "giỏ": "giỏ",
    "gio hoa": "giỏ",
    "kệ": "kệ",
    "ke hoa": "kệ",
    "chậu": "chậu",
    "chau": "chậu",
    "lẵng": "lẵng",
    "lang hoa": "lẵng",
    "hộp": "hộp",
    "hop hoa": "hộp",
    "bình": "bình",
}

# Must actually be about buying flowers to count as a product query at all.
FLOWER_MARKER = re.compile(r"(hoa|lan|hồng|tulip|bó|giỏ|kệ|chậu|lẵng)", re.IGNORECASE)


def _fold(text: str) -> str:
    """Lowercase + strip diacritics, for robust keyword matching."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", _fold(text)) if t]


def _contains_phrase(tokens: list[str], phrase: str) -> bool:
    """Whole-token phrase match.

    Substring matching on diacritic-folded Vietnamese is a trap: 'không' folds
    to 'khong', which *contains* 'hong' (hồng/rose), so 'có freeship không' was
    being tagged with flower_type='hồng'. Likewise 'lẵng'->'lang' contains
    'lan', 'phòng'->'phong' contains 'hong', and 'kết'->'ket' contains 'ke'.
    Matching whole tokens in sequence removes the entire class of error.
    """
    needle = _tokens(phrase)
    if not needle:
        return False
    n = len(needle)
    return any(tokens[i : i + n] == needle for i in range(len(tokens) - n + 1))


def _strip_place_names(tokens: list[str]) -> list[str]:
    """Remove province/district phrases before facet extraction.

    'hoa khai trương bình dương' names a province, but 'bình' is also a vase
    form factor, so the place name has to be consumed before facets are read or
    every Bình Dương query looks like a vase query.
    """
    places = sorted(OTHER_PROVINCES | HCM_AREAS, key=lambda p: -len(_tokens(p)))
    out = list(tokens)
    for place in places:
        needle = _tokens(place)
        n = len(needle)
        if not n:
            continue
        i = 0
        while i <= len(out) - n:
            if out[i : i + n] == needle:
                del out[i : i + n]
            else:
                i += 1
    return out


def _find(text: str, table: dict[str, str]) -> str | None:
    tokens = _strip_place_names(_tokens(text))
    # Longest key first so 'lan hồ điệp' wins over the bare 'lan'.
    for key in sorted(table, key=lambda k: -len(_tokens(k))):
        if _contains_phrase(tokens, key):
            return table[key]
    return None


def provisional_intent(query: str) -> Intent:
    """Best-effort intent from keywords alone. Verified by the LLM later."""
    from rag.retrieval.search import extract_budget_vnd

    tokens = _tokens(query)

    if OFF_TOPIC_MARKERS.search(query) or COMPETITOR_MARKERS.search(query):
        return Intent.OFF_TOPIC
    # Province check is token-based for the same reason facet lookup is: 'vinh'
    # would otherwise match inside unrelated words.
    if any(_contains_phrase(tokens, p) for p in OTHER_PROVINCES) and not any(
        _contains_phrase(tokens, a) for a in HCM_AREAS
    ):
        return Intent.OUT_OF_CATALOGUE
    if POLICY_MARKERS.search(query):
        return Intent.POLICY
    if not FLOWER_MARKER.search(query):
        return Intent.OFF_TOPIC
    if extract_budget_vnd(query) is not None:
        return Intent.BUDGET_SEARCH
    if _find(query, FLOWER_TYPES) or _find(query, FORM_FACTORS):
        return Intent.ATTRIBUTE
    return Intent.PRODUCT_SEARCH


def facets(query: str) -> dict[str, object]:
    from rag.retrieval.search import extract_budget_vnd

    return {
        "occasion": _find(query, OCCASIONS),
        "recipient": _find(query, RECIPIENTS),
        "flower_type": _find(query, FLOWER_TYPES),
        "form_factor": _find(query, FORM_FACTORS),
        "budget_max_vnd": extract_budget_vnd(query),
    }


def _token_set(query: str) -> frozenset[str]:
    return frozenset(t for t in _fold(query).split() if len(t) > 1)


def dedupe(queries: list[str], jaccard_threshold: float = 0.8) -> list[str]:
    """Remove exact and near-duplicate queries.

    Autocomplete expansion produces many variants that differ by a stopword
    ('bó hoa sinh nhật' vs 'bó hoa sinh nhật đẹp'). Keeping both inflates the
    dataset without adding evaluation signal, and makes one phrasing dominate a
    slice.
    """
    kept: list[str] = []
    kept_tokens: list[frozenset[str]] = []
    seen_exact: set[str] = set()

    for q in queries:
        folded = _fold(q).strip()
        if not folded or folded in seen_exact:
            continue
        seen_exact.add(folded)

        tokens = _token_set(q)
        if not tokens:
            continue
        duplicate = False
        for other in kept_tokens:
            union = tokens | other
            if union and len(tokens & other) / len(union) >= jaccard_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(q)
            kept_tokens.append(tokens)

    log.info("Dedupe: %d -> %d queries", len(queries), len(kept))
    return kept


def prefilter(
    queries: list[str],
    *,
    per_intent_cap: int = 30,
    max_words: int = 12,
) -> dict[str, list[str]]:
    """Dedupe, bucket by provisional intent, and cap each bucket.

    Returns intent -> queries. Capping before LLM classification is what keeps
    the verification step inside the free-tier budget while still producing a
    balanced dataset.
    """
    cleaned = [q for q in dedupe(queries) if 1 < len(q.split()) <= max_words]

    buckets: dict[str, list[str]] = defaultdict(list)
    for q in cleaned:
        buckets[provisional_intent(q).value].append(q)

    capped = {intent: qs[:per_intent_cap] for intent, qs in sorted(buckets.items())}
    for intent, qs in capped.items():
        log.info("  %-18s %3d (of %d available)", intent, len(qs), len(buckets[intent]))
    log.info("Prefilter: %d -> %d candidates", len(queries), sum(len(v) for v in capped.values()))
    return capped
