"""Detect and repair corpus damage, then score the detection.

The detectors here **never see the manifest**. They work only from the records,
exactly as they would against a real corpus of unknown quality. The manifest is
used afterwards, and only to score them.

That separation is the whole point: it turns "the data looks cleaner now" into
per-defect precision / recall / F1 against known ground truth.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from rag.ingest.clean import _tokens, detect_boilerplate_tokens
from rag.ingest.corrupt import Corruption, Manifest
from rag.ingest.schema import Product, is_product_url, parse_price_vnd

log = logging.getLogger(__name__)

JUNK_VALUES = {"nan", "none", "null", "<na>", "na"}

# Sequences that only appear when UTF-8 has been decoded as latin-1.
# Vietnamese text mangled this way is full of these runs.
#
# The suppression below is load-bearing: ruff flags these characters as
# "ambiguous" and suggests ASCII lookalikes, but matching those exact
# non-ASCII sequences is the entire purpose of the pattern. Replacing them
# would make the mojibake detector silently match nothing.
_MOJIBAKE_RE = re.compile(r"(Ã[-¿]|á»|áº|Æ°|Ä‘|â€|Ã¡|Ã´|Ã¬)")  # noqa: RUF001

MIN_DESCRIPTION_CHARS = 30


@dataclass
class Detection:
    """What the detectors found, keyed the same way the manifest is."""

    found: dict[str, list[str]] = field(default_factory=dict)

    def mark(self, url: str, label: str) -> None:
        self.found.setdefault(url, []).append(label)

    def urls_with(self, label: str) -> set[str]:
        return {u for u, labels in self.found.items() if label in labels}

    @property
    def all_urls(self) -> set[str]:
        return set(self.found)

    def summary(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for labels in self.found.values():
            counts.update(labels)
        return dict(counts)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def looks_junk(value: str) -> bool:
    return value.strip().lower() in JUNK_VALUES


def looks_mojibake(text: str) -> bool:
    return bool(_MOJIBAKE_RE.search(text))


def demojibake(text: str) -> str:
    """Reverse a latin-1 mis-decode, if that yields valid Vietnamese."""
    try:
        restored = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # Only accept if the result is actually cleaner.
    return restored if not looks_mojibake(restored) else text


def _canonical_url(url: str) -> str:
    """Strip tracking params and redundant slashes for duplicate detection."""
    base = url.split("?", 1)[0].split("#", 1)[0]
    return re.sub(r"/+$", "/", base).rstrip("/").lower()


def _is_canonical_form(url: str) -> bool:
    """True if the URL is already its own canonical form.

    Used to decide *which* member of a duplicate pair to drop. Without this the
    detector keeps whichever copy it happened to see first — after shuffling
    that is often the tracking-param variant, so it drops the good record and
    reports a duplicate at the wrong URL.
    """
    return url.rstrip("/").lower() == _canonical_url(url)


def _shingles(text: str, size: int = 8) -> set[str]:
    """Overlapping word n-grams, for repeated-passage detection."""
    tokens = _tokens(text)
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def detect_repeated_passages(
    texts: list[str], *, min_doc_share: float = 0.05, size: int = 8
) -> set[str]:
    """Shingles that recur across many documents.

    Replaces the token-document-frequency approach, which failed badly here:
    flooding 25% of the corpus with a shared paragraph never pushes any single
    token past a 60% document-frequency threshold, so recall was 0.017. Verbatim
    repeated *passages* are the actual signal, and they are detectable at any
    contamination rate above the threshold.
    """
    if not texts:
        return set()
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(_shingles(t, size))
    cutoff = max(2, int(min_doc_share * len(texts)))
    return {sh for sh, n in counts.items() if n >= cutoff}


def _normalized_body(product: Product) -> str:
    text = unicodedata.normalize("NFC", f"{product.title} {product.description}".lower())
    return re.sub(r"\s+", " ", text).strip()


def repair_products(
    products: list[Product],
    *,
    boilerplate_threshold: float = 0.6,
) -> tuple[list[Product], Detection]:
    """Detect defects, repair what is recoverable, drop what is not."""
    detection = Detection()

    # Corpus-level pass first: boilerplate is only visible across documents, not
    # within one. This is the detector that catches the defect the original
    # pipeline shipped and never noticed.
    descriptions = [p.description for p in products]
    shared_tokens = detect_boilerplate_tokens(descriptions, boilerplate_threshold)
    repeated = detect_repeated_passages(descriptions)

    seen_canonical: dict[str, str] = {}
    seen_body: dict[str, str] = {}
    kept: list[Product] = []

    # Canonical URLs first, so a duplicate pair drops the tracking-param variant
    # rather than whichever copy the shuffle happened to put first.
    ordered = sorted(products, key=lambda p: (not _is_canonical_form(p.url), len(p.url)))

    for original in ordered:
        p = original.model_copy(deep=True)
        drop = False

        # --- non-product pages ---
        # Judge the *canonical* URL: a tracking-param duplicate of a real
        # product is a duplicate, not an archive page. Checking the raw URL
        # flagged every '?utm_source=' clone as non-product (P=0.227).
        if not is_product_url(_canonical_url(p.url) + "/"):
            detection.mark(p.url, Corruption.NON_PRODUCT)
            drop = True

        # --- missing / junk required fields ---
        if any(looks_junk(v) or not v.strip() for v in (p.title, p.description, p.price_raw)):
            detection.mark(p.url, Corruption.MISSING_FIELD)
            if looks_junk(p.title) or not p.title.strip():
                drop = True  # unusable without a title
            else:
                if looks_junk(p.description):
                    p.description = ""
                if looks_junk(p.price_raw):
                    p.price_raw, p.price_vnd = "", None

        # --- mojibake (recoverable) ---
        if looks_mojibake(p.title) or looks_mojibake(p.description):
            detection.mark(p.url, Corruption.MOJIBAKE)
            p.title = demojibake(p.title)
            p.description = demojibake(p.description)

        # --- unparseable price (recoverable to None, not fatal) ---
        if p.price_raw and parse_price_vnd(p.price_raw) is None:
            detection.mark(p.url, Corruption.PRICE)
            p.price_vnd = None
        elif not p.price_raw:
            detection.mark(p.url, Corruption.PRICE)

        # --- truncated description ---
        if 0 < len(p.description) < MIN_DESCRIPTION_CHARS:
            detection.mark(p.url, Corruption.TRUNCATION)

        # --- boilerplate flooding ---
        doc_shingles = _shingles(p.description)
        if doc_shingles and repeated:
            repeated_share = len(doc_shingles & repeated) / len(doc_shingles)
            if repeated_share >= 0.30:
                detection.mark(p.url, Corruption.BOILERPLATE)
                # Strip the shared passage rather than dropping the document —
                # the product-specific tail is still worth indexing.
                p.description = _strip_boilerplate(p.description, shared_tokens, repeated)

        # --- duplicates ---
        canon = _canonical_url(p.url)
        body = _normalized_body(p)
        if canon in seen_canonical or body in seen_body:
            detection.mark(p.url, Corruption.DUPLICATE)
            drop = True
        else:
            seen_canonical[canon] = p.url
            seen_body[body] = p.url

        if not drop:
            # Recompute: fields changed, and a stale hash breaks incremental
            # re-indexing later.
            p.content_hash = p.compute_hash()
            kept.append(p)

    log.info(
        "Repair: %d -> %d records kept | detections: %s",
        len(products),
        len(kept),
        detection.summary(),
    )
    return kept, detection


def _strip_boilerplate(
    description: str, shared_tokens: set[str], repeated: set[str] | None = None
) -> str:
    """Drop sentences that recur verbatim across the corpus."""
    sentences = re.split(r"(?<=[.!?])\s+", description)
    kept = []
    for sent in sentences:
        toks = _tokens(sent)
        if not toks:
            continue
        sent_shingles = _shingles(sent)
        if repeated and sent_shingles and len(sent_shingles & repeated) / len(sent_shingles) >= 0.5:
            continue
        share = sum(1 for t in toks if t in shared_tokens) / len(toks)
        if share < 0.85:
            kept.append(sent)
    return " ".join(kept).strip() or description


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _prf(detected: set[str], truth: set[str]) -> dict[str, float]:
    tp = len(detected & truth)
    fp = len(detected - truth)
    fn = len(truth - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def score_detection(detection: Detection, manifest: Manifest) -> dict:
    """Per-defect and overall detection quality against the manifest.

    Note on `price_corruption` precision: the detector flags any record with an
    unparseable or absent price, which legitimately includes products that were
    already priceless in the clean corpus. Those count as false positives here
    even though flagging them is arguably correct — the score is reported as
    measured rather than adjusted to look better.
    """
    per_defect = {
        label: _prf(detection.urls_with(label), manifest.urls_with(label))
        for label in Corruption.ALL
    }
    overall = _prf(detection.all_urls, manifest.all_corrupted_urls)

    return {
        "seed": manifest.seed,
        "n_clean_input": manifest.n_clean_input,
        "n_corrupt_output": manifest.n_corrupt_output,
        "corruption_summary": manifest.summary(),
        "detection_summary": detection.summary(),
        "per_defect": per_defect,
        "overall_any_defect": overall,
        "macro_f1": round(sum(v["f1"] for v in per_defect.values()) / len(per_defect), 4),
    }
