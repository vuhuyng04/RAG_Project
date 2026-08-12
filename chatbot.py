"""Entry point kept at the repo root for Streamlit Community Cloud.

The deployed app at https://vuhuy-rag.streamlit.app/ points at `chatbot.py`, so
this path has to keep working. The implementation moved to `app/chatbot.py`;
this is a shim, not a second copy.

DO NOT REMOVE THE sys.path INSERT BELOW. It looks redundant next to `uv sync`,
which installs the project locally — but the generated requirements.txt is
exported with `--no-emit-project`, so on Streamlit Community Cloud nothing ever
pip-installs `src/rag`. This line *is* the install mechanism there, and the
deployed app breaks without it.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "app" / "chatbot.py"), run_name="__main__")
