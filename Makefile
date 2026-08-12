.PHONY: install lock test lint states eval clean-cache app

install:
	uv sync

# Streamlit Community Cloud only reads requirements.txt, not uv.lock. Keep the
# exported file in sync or the live demo runs different versions than local.
lock:
	uv lock
	uv run python -m scripts.export_requirements

test:
	uv run pytest -q

lint:
	uv run ruff check src eval scripts tests
	uv run ruff format --check src eval scripts tests

# Build all four corpus states. Order matters: corrupt derives from clean,
# repaired derives from corrupt.
states:
	uv run python -m scripts.build_state --state clean
	uv run python -m scripts.build_state --state legacy
	uv run python -m scripts.build_state --state corrupt --seed 42
	uv run python -m scripts.build_state --state repaired

eval:
	uv run python -m eval.run_matrix
	uv run python -m eval.compare

app:
	uv run streamlit run app/chatbot.py

clean-cache:
	uv run python -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.cache'), pathlib.Path('eval/.ragas_cache')]]"
