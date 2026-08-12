"""Post-hoc citation validation.

Asking a model to cite is not the same as it citing correctly. This parses the
[n] markers back out of the answer and checks each against the context that was
actually supplied, which turns "we added citations" into a measurable
`citation_validity_rate` — deterministic, and free of LLM-judge cost or bias.

Three separate failure modes are distinguished, because they call for different
fixes:

* **invalid**  — cites [7] when only 5 products were shown (fabricated index)
* **uncited**  — makes product/price claims with no citation at all
* **unused**   — products supplied but never referenced (retrieval noise)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"\[(\d{1,2})\]")

# What counts as a claim needing a citation: a concrete, checkable assertion —
# a currency amount or a product code.
#
# An earlier version also matched the bare nouns "giá", "mẫu", "sản phẩm", "bó",
# and flagged both of these as ungrounded:
#   "Chào bạn, ... có một số mẫu kệ hoa khai trương đẹp với giá dưới 1 triệu ạ:"
#   "Bạn muốn tham khảo thêm thông tin chi tiết về mẫu nào không ạ?"
# Neither asserts anything checkable; the second is a question. Precision
# matters here because uncited_claim_rate is a reported metric — a validator
# that cries wolf makes the number meaningless.
_CLAIM_MARKERS = re.compile(
    r"(\d[\d.,]{2,}\s*(?:₫|đ\b|vnd|k\b|nghìn|triệu)"  # a real price
    r"|\bM\d{1,4}\b)",  # a product code, e.g. M440
    re.IGNORECASE,
)

# Second, weaker tier: assertions about product *attributes*. Measured against
# five realistic hallucinated sentences, the strict tier above caught only 2/5 —
# it is blind to exactly the claims a RAG system most often invents:
#   "Mẫu này dùng hoa hướng dương và lan hồ điệp trắng."
#   "Kệ hoa này cao khoảng hai mét và có thể giao trong ngày."
#   "Sản phẩm được thiết kế hai tầng với tông đỏ chủ đạo."
# A validator that cannot fail is not evidence, so these are detected too — but
# reported as a separate, lower-confidence rate, because the vocabulary
# heuristic is inherently less precise than a currency match.
_DESCRIPTIVE_MARKERS = re.compile(
    r"\b(thiết kế|tầng|tông|màu|gồm|bao gồm|kết hợp|sử dụng|dùng|làm từ|"
    r"cao|rộng|kích thước|chất liệu|phối|điểm xuyết|đi kèm|tặng kèm|"
    r"giao trong|giao hàng|bảo hành|size)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
# Questions and invitations assert nothing, so they cannot be ungrounded.
_QUESTION_RE = re.compile(r"\?\s*$")


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    unused: list[int] = field(default_factory=list)
    # High-confidence: price or product-code assertions with no citation.
    uncited_claims: list[str] = field(default_factory=list)
    # Lower-confidence: attribute assertions with no citation.
    uncited_descriptive: list[str] = field(default_factory=list)
    n_context: int = 0

    @property
    def is_valid(self) -> bool:
        """No fabricated indices and no ungrounded price/code claims.

        Deliberately excludes `uncited_descriptive`: that tier is a vocabulary
        heuristic and is reported separately rather than gating the headline
        validity rate.
        """
        return not self.invalid and not self.uncited_claims

    @property
    def has_citations(self) -> bool:
        return bool(self.cited)

    def to_dict(self) -> dict:
        return {
            "cited": self.cited,
            "invalid": self.invalid,
            "unused": self.unused,
            "n_uncited_claims": len(self.uncited_claims),
            "uncited_claims": self.uncited_claims[:3],
            "n_uncited_descriptive": len(self.uncited_descriptive),
            "uncited_descriptive": self.uncited_descriptive[:3],
            "n_context": self.n_context,
            "is_valid": self.is_valid,
        }


def validate_citations(answer: str, n_context: int) -> CitationReport:
    report = CitationReport(n_context=n_context)

    found = [int(m) for m in CITATION_RE.findall(answer)]
    report.cited = sorted({n for n in found if 1 <= n <= n_context})
    report.invalid = sorted({n for n in found if not 1 <= n <= n_context})
    report.unused = [i for i in range(1, n_context + 1) if i not in report.cited]

    # Abstention answers legitimately have no context and no citations.
    if n_context == 0:
        return report

    # Citations are scoped to the *block* (list item or paragraph), not the
    # sentence. Models place one marker at the end of a bullet that covers the
    # whole item:
    #
    #   - Hoa Chia Buồn M20: Vòng hoa ... lòng thành kính. Giá 1.100.000đ [1].
    #
    # Splitting that on the full stop leaves "…M20: Vòng hoa … lòng thành kính."
    # with no marker, and a sentence-scoped check flagged every such bullet.
    # Observed on a real answer: 5 bullets, all correctly cited, 5 false
    # positives.
    for block in answer.split("\n"):
        if not block.strip() or CITATION_RE.search(block):
            continue
        for sentence in _SENTENCE_SPLIT.split(block):
            s = sentence.strip()
            if not s or _QUESTION_RE.search(s):
                continue
            if _CLAIM_MARKERS.search(s):
                report.uncited_claims.append(s[:120])
            elif _DESCRIPTIVE_MARKERS.search(s):
                report.uncited_descriptive.append(s[:120])

    return report


def strip_invalid_citations(answer: str, n_context: int) -> str:
    """Remove citation markers pointing outside the supplied context.

    Rendering a fabricated [7] as if it were a real source is worse than showing
    no marker: it looks authoritative. The claim text is left intact so the
    reader still sees what was said — the UI flags it separately.
    """

    def _sub(match: re.Match[str]) -> str:
        n = int(match.group(1))
        return match.group(0) if 1 <= n <= n_context else ""

    return CITATION_RE.sub(_sub, answer)


def citation_validity_rate(reports: list[CitationReport]) -> dict[str, float]:
    """Aggregate over an evaluation run."""
    if not reports:
        return {}
    grounded = [r for r in reports if r.n_context > 0]
    if not grounded:
        return {"n": 0}
    n = len(grounded)
    return {
        "n": n,
        # Strongest signal: a citation pointing at a source that does not exist.
        # Purely structural, no heuristics — quotable without caveat.
        "invalid_index_rate": round(sum(1 for r in grounded if r.invalid) / n, 4),
        "citation_rate": round(sum(1 for r in grounded if r.has_citations) / n, 4),
        "validity_rate": round(sum(1 for r in grounded if r.is_valid) / n, 4),
        # Scoped to price/product-code assertions. Reliable but narrow.
        "uncited_price_or_code_claim_rate": round(
            sum(1 for r in grounded if r.uncited_claims) / n, 4
        ),
        # Vocabulary heuristic; lower precision, reported for completeness.
        # Attribute-level grounding is not deterministically checkable — that is
        # what RAGAS Faithfulness is for, and why this does not replace it.
        "uncited_descriptive_claim_rate": round(
            sum(1 for r in grounded if r.uncited_descriptive) / n, 4
        ),
        "mean_unused_context": round(sum(len(r.unused) for r in grounded) / n, 2),
    }
