"""The demo must serve a product, not an experiment console.

Retrieval strategies and corpus states are evaluation apparatus. An end user
cannot meaningfully choose between `hybrid_budget` and `dense_rerank`, and
`corrupt` is a deliberately damaged index that must never serve a real answer.
These guards keep that separation from eroding the next time a control is added.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from rag.config import State
from rag.retrieval.search import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "chatbot.py"


def _source() -> str:
    return APP.read_text(encoding="utf-8")


def _const(name: str) -> str:
    match = re.search(rf'^{name} = (".*?")$', _source(), re.M)
    assert match, f"{name} not found in app/chatbot.py"
    return ast.literal_eval(match.group(1))


def test_production_defaults_are_valid() -> None:
    assert _const("PRODUCTION_CONFIG") in CONFIGS
    assert _const("PRODUCTION_STATE") in {s.value for s in State}


def test_production_serves_the_clean_corpus() -> None:
    """corrupt/repaired are experiment states; baseline is a weaker control."""
    assert _const("PRODUCTION_STATE") == "clean"


def test_production_config_is_the_measured_best() -> None:
    """Whatever the demo serves should be defensible from eval/results/.

    `dense_budget` leads Recall@5, MRR@5 and nDCG@5 while costing an order of
    magnitude less p95 latency than the reranking configs.
    """
    import json

    results = ROOT / "eval" / "results"
    scores = {}
    for path in results.glob("retrieval_*_clean.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        recall = report["retrieval_raw"].get("recall@5")
        if recall is not None:
            scores[report["config"]] = recall

    best = max(scores, key=scores.get)
    assert _const("PRODUCTION_CONFIG") == best, (
        f"demo serves {_const('PRODUCTION_CONFIG')} but {best} measured best "
        f"(recall@5 {scores[best]:.3f}); update the default or justify the gap"
    )


def test_experiment_controls_are_gated_behind_lab_mode() -> None:
    """The strategy and corpus selectors must sit inside the lab branch."""
    tree = ast.parse(_source())
    main = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    selector_lines = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
    ]
    assert selector_lines, "no selectbox found — did the controls move?"

    lab_branches = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "lab"
    ]
    assert lab_branches, "no `if lab:` branch in main()"

    gated = {
        line
        for branch in lab_branches
        for node in ast.walk(branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
        for line in [node.lineno]
    }
    ungated = set(selector_lines) - gated
    assert not ungated, f"selectbox outside lab mode at line(s) {sorted(ungated)}"


def test_lab_mode_is_opt_in() -> None:
    source = _source()
    assert "def lab_mode_enabled" in source
    assert "query_params" in source, "lab mode should be reachable via ?lab=1"


def test_non_production_state_warns() -> None:
    """Serving a damaged corpus must be visibly flagged, never silent."""
    assert "PRODUCTION_STATE" in _source()
    assert re.search(r"if state_name != PRODUCTION_STATE", _source())
