"""Export uv.lock to a pip-installable requirements.txt.

    uv run python -m scripts.export_requirements

Streamlit Community Cloud only reads requirements.txt, not uv.lock, so the
deployed app would otherwise run different versions than local — exactly the
drift this project is about.

`uv export` alone is not sufficient: the lock pins `torch==2.13.0+cpu`, which
lives on the PyTorch CPU index, and the export does not emit that index URL.
pip would fail to resolve it on deploy. The extra-index-url line is prepended
here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "requirements.txt"

HEADER = """\
# GENERATED FILE — DO NOT EDIT.
# Regenerate with:  uv run python -m scripts.export_requirements
#
# Exists only because Streamlit Community Cloud reads requirements.txt and not
# uv.lock. uv.lock remains the source of truth for local and CI installs.
#
# The extra index is required: torch is pinned to a +cpu build that is not on
# PyPI. Without this line pip cannot resolve it.
--extra-index-url https://download.pytorch.org/whl/cpu

"""


def main() -> int:
    result = subprocess.run(
        ["uv", "export", "--no-hashes", "--no-dev", "--no-emit-project"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    body = "\n".join(
        line for line in result.stdout.splitlines() if line.strip() and not line.startswith("#")
    )
    OUT.write_text(HEADER + body + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(body.splitlines())} requirement lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
