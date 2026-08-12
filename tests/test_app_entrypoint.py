"""Guards on the two ways the app can be started.

The deployed demo is launched through the root `chatbot.py` shim, not through
`app/chatbot.py` directly. That difference is invisible locally and broke the
app once already: Streamlit puts the script's own directory on `sys.path`, so
`import ui` worked when running the app file directly and raised
ModuleNotFoundError through the shim — which is the path Streamlit Cloud uses.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        timeout=180,
        check=False,
    )


def test_app_module_imports_without_streamlit_path_help() -> None:
    """Importing app/chatbot.py must not depend on the launcher's sys.path.

    Executed in a subprocess from the repo root with only the default path, i.e.
    the situation the runpy shim creates.
    """
    result = _run(
        "import runpy, sys;"
        "sys.argv=['chatbot.py'];"
        "runpy.run_path('app/chatbot.py', run_name='__not_main__')"
    )
    assert "ModuleNotFoundError" not in result.stderr, result.stderr[-1500:]
    assert result.returncode == 0, result.stderr[-1500:]


def test_ui_module_is_importable_standalone() -> None:
    result = _run("import sys; sys.path.insert(0, 'app'); import ui; assert ui.CSS")
    assert result.returncode == 0, result.stderr[-1000:]


def test_root_shim_points_at_the_app_file() -> None:
    """The shim must stay a shim.

    Pipeline logic in the root file is how the measured system and the served
    system drift apart. Checking for `st.` calls rather than the word
    "streamlit" — the docstring mentions Streamlit Cloud legitimately, and an
    earlier version of this test failed on exactly that.
    """
    shim = (ROOT / "chatbot.py").read_text(encoding="utf-8")
    assert "app" in shim and "chatbot.py" in shim
    assert "st." not in shim, "shim should delegate, not render UI itself"
    assert len(shim.splitlines()) < 40, "shim is growing — move logic into app/"
    # The sys.path insert is the install mechanism on Streamlit Cloud.
    assert "sys.path" in shim
