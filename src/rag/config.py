"""Single source of truth for settings.

The legacy code duplicated the model name, the collection name and the vector
helper between craw_data.ipynb and chatbot.py, so changing one silently
desynced the other. Everything reads from here instead.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
STATES_DIR = DATA_DIR / "states"
EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
FIXTURES_DIR = EVAL_DIR / "fixtures"
DATASET_DIR = EVAL_DIR / "dataset"


class State(StrEnum):
    """The four corpus states the evaluation matrix runs against.

    LEGACY faithfully reproduces the original notebook's indexing logic: every
    field concatenated into one string (boilerplate promo text included) and
    the missing-fillna bug that wrote literal "nan" into the index. It is built
    from the *same crawl* as CLEAN, which makes the comparison a controlled
    experiment — raw data held constant, only the pipeline varies — and keeps
    the "before" column reproducible by anyone who clones the repo.

    CLEAN is the new pipeline. CORRUPT applies seeded, deliberate damage to the
    validated CLEAN records. REPAIRED is CORRUPT after the repair pass.
    """

    LEGACY = "legacy"
    CLEAN = "clean"
    CORRUPT = "corrupt"
    REPAIRED = "repaired"


def _read_streamlit_secret(key: str) -> str | None:
    """Read from st.secrets without requiring streamlit to be installed.

    Importing streamlit outside a Streamlit runtime is slow and noisy, and the
    eval harness runs headless, so this stays best-effort and quiet.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        return st.secrets[key]
    except Exception:
        # Raises when there is no secrets.toml at all, which is the normal case
        # for local and CI runs.
        return None


def _resolve(key: str) -> str | None:
    return os.getenv(key) or _read_streamlit_secret(key)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials ---
    gemini_api_key: str = ""
    qdrant_endpoint: str = ""
    qdrant_api_key: str = ""

    # --- Models ---
    embedding_model: str = "Alibaba-NLP/gte-multilingual-base"
    reranker_model: str = "Alibaba-NLP/gte-multilingual-reranker-base"
    embedding_dim: int = 768

    # Pinned deliberately, NOT "gemini-flash-latest": an alias silently changes
    # model underneath and would make committed eval numbers irreproducible.
    # (The original code used gemini-2.0-flash, which Google has since retired —
    # it now returns 404.)
    #
    # The *product's* generator — what the chatbot answers with.
    gemini_model: str = "gemini-2.5-flash"

    # The *evaluator* — gold-label judging and RAGAS. Deliberately a different
    # model from the generator: having a model grade its own output biases
    # faithfulness and relevancy upward. The free-tier quota being per-model
    # (docs/decisions.md D7) makes the separation free in practice, since the
    # two purposes then draw on separate daily budgets.
    judge_model: str = "gemini-3.6-flash"

    # --- Collections ---
    collection_prefix: str = "flowers"

    # --- Retrieval defaults ---
    top_k: int = 5
    fetch_k: int = 20  # candidates fetched before reranking
    score_threshold: float = 0.0  # calibrated in GĐ6; 0.0 reproduces legacy behaviour

    # Whether to append a coarse price phrase ("tầm giá 1 đến 2 triệu") to the
    # embedded text. Plausibly helps budget-phrased queries, but there are only
    # six bands across ~660 products, so ~110 products share the phrase verbatim
    # — a smaller version of the boilerplate problem this project exists to fix.
    # Left as an A/B variable rather than assumed; measured in the eval matrix.
    embed_price_band: bool = True

    # --- Ingest ---
    crawl_base_url: str = "https://hoatuoimymy.com"
    crawl_concurrency: int = 5

    # --- Gemini free-tier rate limiting ---
    gemini_rpm: int = 15
    gemini_daily_cap: int = 1500

    @model_validator(mode="after")
    def _fill_from_env_and_secrets(self) -> Settings:
        # pydantic-settings already handled .env and the environment. This adds
        # the Streamlit secrets fallback (for the deployed app) and the
        # misspelled legacy variable name.
        if not self.gemini_api_key:
            # GENMINI_KEY is a typo carried over from the original chatbot.py.
            # The deployed Streamlit Cloud secrets still use it, so dropping the
            # fallback would break the live demo on the next deploy.
            self.gemini_api_key = _resolve("GEMINI_API_KEY") or _resolve("GENMINI_KEY") or ""
        if not self.qdrant_endpoint:
            self.qdrant_endpoint = _resolve("QDRANT_ENDPOINT") or ""
        if not self.qdrant_api_key:
            self.qdrant_api_key = _resolve("QDRANT_API_KEY") or ""
        return self

    def collection_for(self, state: State | str) -> str:
        """Map a corpus state to its Qdrant collection name."""
        return f"{self.collection_prefix}_{State(state).value}"

    def require(self, *fields: str) -> None:
        """Fail loudly and early with an actionable message.

        Better than a None api_key surfacing 40 lines later as an opaque 401.
        """
        missing = [f for f in fields if not getattr(self, f, None)]
        if missing:
            names = ", ".join(f.upper() for f in missing)
            raise RuntimeError(
                f"Missing required setting(s): {names}. "
                f"Copy .env.example to .env and fill them in (see plan §0)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
