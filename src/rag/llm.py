"""Rate-limited, disk-cached Gemini calls.

The free tier allows ~15 requests/minute and 1500/day, and the evaluation
matrix would blow through that easily. Two mitigations live here:

* a token-bucket throttle so runs do not trip 429s, and
* a content-addressed disk cache, so re-running an evaluation costs nothing and
  a config change only re-computes what actually changed.

Both are what make `run_matrix` resumable across days rather than a single
fragile session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from rag.config import REPO_ROOT, get_settings

log = logging.getLogger(__name__)

CACHE_DIR = REPO_ROOT / ".cache" / "llm"


class RateLimiter:
    """Simple thread-safe spacing limiter."""

    def __init__(self, rpm: int) -> None:
        self.min_interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            sleep_for = self.min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


class GeminiClient:
    """Wraps generate_content with caching, throttling and retry."""

    def __init__(self, cache_dir: Path | None = None, *, role: str = "generator") -> None:
        """`role` selects the model: 'generator' answers, 'judge' evaluates.

        Keeping them distinct avoids a model grading its own output, and — since
        the free-tier quota is per-model — gives each purpose its own budget.
        """
        s = get_settings()
        self.settings = s
        self.role = role
        self.model_name = s.judge_model if role == "judge" else s.gemini_model
        self.limiter = RateLimiter(s.gemini_rpm)
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0
        self.cache_hits = 0

    def _key(self, prompt: str, **kw: Any) -> str:
        # The model name is part of the cache key: a cached answer from a
        # different model is not a valid substitute.
        blob = json.dumps(
            {"m": self.model_name, "p": prompt, **kw},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_retries: int = 5,
        use_cache: bool = True,
    ) -> str:
        key = self._key(prompt, t=temperature, j=json_mode)
        path = self.cache_dir / f"{key}.json"

        if use_cache and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        import google.generativeai as genai

        from rag.clients import get_gemini

        model = get_gemini(self.model_name)
        cfg: dict[str, Any] = {"temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            self.limiter.wait()
            try:
                resp = model.generate_content(
                    prompt, generation_config=genai.types.GenerationConfig(**cfg)
                )
                text = resp.text
                self.calls_made += 1
                if use_cache:
                    path.write_text(
                        json.dumps({"prompt": prompt, "response": text}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                return text
            except Exception as exc:
                last_exc = exc
                # 429 / transient: exponential backoff. Anything else will
                # exhaust retries and surface, which is what we want.
                wait = min(2**attempt * 5, 120)
                log.warning(
                    "Gemini call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    str(exc)[:160],
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini failed after {max_retries} attempts") from last_exc

    def generate_json(self, prompt: str, **kw: Any) -> Any:
        raw = self.generate(prompt, json_mode=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Models occasionally wrap JSON in a fence despite the mime type.
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            return json.loads(cleaned.strip())

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "role": self.role,
            "api_calls": self.calls_made,
            "cache_hits": self.cache_hits,
        }
