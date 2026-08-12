"""Generate the markdown tables the README quotes, straight from eval/results/.

    uv run python -m eval.compare

Nothing in the README is typed by hand. Every number is read from a committed
JSON file, so a stale or invented figure is impossible by construction — which
is the whole acceptance gate of this project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rag.config import RESULTS_DIR

OUT = RESULTS_DIR / "summary_tables.md"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def quality_table() -> str:
    states = ["baseline", "clean", "corrupt", "repaired"]
    reports = {s: _load(RESULTS_DIR / f"quality_{s}.json") for s in states}
    present = {s: r for s, r in reports.items() if r}
    if not present:
        return "_No quality reports found._\n"

    rows = [
        ("Documents", lambda r: f"{r['documents']}"),
        (
            "Boilerplate tokens (doc-freq ≥60%)",
            lambda r: f"{r['boilerplate_ratio_embed_text']:.1%}",
        ),
        (
            "Docs built from repeated passages",
            lambda r: f"{r.get('repeated_passage_doc_rate', 0):.1%}",
        ),
        ("Documents embedding to junk", lambda r: f"{r['junk_embed_text']}"),
        ("Literal `nan` tokens", lambda r: f"{r['embed_text_containing_nan_token']}"),
        ("Non-product URLs indexed", lambda r: f"{r['non_product_urls']}"),
        ("Duplicate URLs", lambda r: f"{r['duplicate_urls']}"),
        ("Price parse rate", lambda r: f"{r['price_parse_rate']:.1%}"),
    ]

    header = "| Metric | " + " | ".join(present) + " |"
    sep = "|---|" + "---|" * len(present)
    lines = [header, sep]
    for label, fn in rows:
        cells = " | ".join(fn(present[s]) for s in present)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines) + "\n"


def repair_table() -> str:
    report = _load(RESULTS_DIR / "repair_detection.json")
    if not report:
        return "_No repair detection report found._\n"

    lines = [
        "| Defect | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in report["per_defect"].items():
        lines.append(
            f"| `{label}` | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"**{s['f1']:.3f}** | {s['tp']} | {s['fp']} | {s['fn']} |"
        )
    overall = report["overall_any_defect"]
    lines.append(
        f"| **any defect** | {overall['precision']:.3f} | {overall['recall']:.3f} | "
        f"**{overall['f1']:.3f}** | {overall['tp']} | {overall['fp']} | {overall['fn']} |"
    )
    lines.append("")
    lines.append(f"Macro F1 across defects: **{report['macro_f1']:.3f}**")
    return "\n".join(lines) + "\n"


def _cell(report: dict[str, Any], key: str) -> str:
    raw = report.get("retrieval_raw", {})
    if key in raw:
        return f"{raw[key]:.3f}"
    return "-"


def retrieval_table(state: str = "clean") -> str:
    """A/B ladder on one corpus state."""
    files = sorted(RESULTS_DIR.glob(f"retrieval_*_{state}.json"))
    reports = [(f, _load(f)) for f in files]
    reports = [(f, r) for f, r in reports if r]
    if not reports:
        return "_No retrieval results found._\n"

    # Present in the order the mechanisms stack up, not alphabetically.
    order = [
        "baseline",
        "dense",
        "dense_threshold",
        "dense_budget",
        "hybrid",
        "hybrid_budget",
        "dense_rerank",
        "full",
    ]
    by_config = {r["config"]: r for _, r in reports}
    ordered = [by_config[c] for c in order if c in by_config]
    ordered += [r for c, r in by_config.items() if c not in order]

    lines = [
        "| Config | Recall@5 | MRR@5 | nDCG@5 | Abstention F1 | p95 latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in ordered:
        lines.append(
            f"| `{r['config']}` | {_cell(r, 'recall@5')} | {_cell(r, 'mrr@5')} | "
            f"{_cell(r, 'ndcg@5')} | {r['abstention']['f1']:.3f} | "
            f"{r['latency_ms']['p95']:.0f} ms |"
        )

    n = ordered[0]["retrieval_raw"].get("n", "?")
    reviewed = ordered[0].get("golden_reviewed", False)
    lines.append("")
    lines.append(
        f"n = {n} answerable queries. "
        + (
            "Golden set human-reviewed."
            if reviewed
            else "**Golden set not yet human-reviewed — provisional.**"
        )
    )
    return "\n".join(lines) + "\n"


def state_table(config: str = "dense") -> str:
    """One config across all corpus states."""
    files = sorted(RESULTS_DIR.glob(f"retrieval_{config}_*.json"))
    reports = [r for r in (_load(f) for f in files) if r]
    if not reports:
        return "_No cross-state results found._\n"

    order = ["baseline", "clean", "corrupt", "repaired"]
    by_state = {r["state"]: r for r in reports}
    ordered = [by_state[s] for s in order if s in by_state]

    lines = [
        "| Corpus state | Recall@5 | MRR@5 | nDCG@5 |",
        "|---|---|---|---|",
    ]
    for r in ordered:
        lines.append(
            f"| `{r['state']}` | {_cell(r, 'recall@5')} | {_cell(r, 'mrr@5')} | {_cell(r, 'ndcg@5')} |"
        )

    # The recovery figure, computed rather than typed.
    need = {"clean", "corrupt", "repaired"}
    if need <= set(by_state):
        clean = by_state["clean"]["retrieval_raw"].get("recall@5")
        corrupt = by_state["corrupt"]["retrieval_raw"].get("recall@5")
        repaired = by_state["repaired"]["retrieval_raw"].get("recall@5")
        if None not in (clean, corrupt, repaired) and clean > corrupt:
            recovery = (repaired - corrupt) / (clean - corrupt)
            lines.append("")
            lines.append(
                f"Repair recovered **{recovery:.1%}** of the Recall@5 lost to corruption "
                f"({corrupt:.3f} → {repaired:.3f}, against a clean baseline of {clean:.3f})."
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    sections = [
        "<!-- GENERATED by `uv run python -m eval.compare` — do not edit by hand. -->\n",
        "## Corpus quality by state\n",
        quality_table(),
        "\n## Defect detection vs. the corruption manifest\n",
        repair_table(),
        "\n## Retrieval A/B ladder (clean corpus)\n",
        retrieval_table("clean"),
        "\n## Dense retrieval across corpus states\n",
        state_table("dense"),
    ]
    text = "\n".join(sections)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
