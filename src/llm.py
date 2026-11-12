"""Thin wrapper around the Gemini client: model resolution, pacing and retries.

Every Gemini call in the project goes through here, so rate limits, retries and
cost tracking live in one place.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

from . import config


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class UsageMeter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, response) -> None:
        self.calls += 1
        meta = getattr(response, "usage_metadata", None)
        i = getattr(meta, "prompt_token_count", 0) or 0
        o = getattr(meta, "candidates_token_count", 0) or 0
        self.input_tokens += i
        self.output_tokens += o
        m = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        m["calls"] += 1
        m["input_tokens"] += i
        m["output_tokens"] += o


USAGE = UsageMeter()
_client = None
_resolved: dict[str, str] = {}
_last_call = 0.0


def client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise LLMUnavailable("GEMINI_API_KEY is not set. Add it to the .env file.")
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def resolve_model(preferred: str, family: str) -> str:
    """Use the configured model; if the account does not have it, pick the newest
    available model of the same family ("flash" or "pro") instead of crashing."""
    if preferred in _resolved:
        return _resolved[preferred]
    try:
        names = [m.name.replace("models/", "") for m in client().models.list()]
    except Exception:  # listing is a convenience; fall back to the configured name
        _resolved[preferred] = preferred
        return preferred
    if preferred in names:
        choice = preferred
    else:
        candidates = sorted(
            (n for n in names if family in n and "gemini" in n and not any(x in n for x in ("tts", "image", "live", "audio", "embedding", "lite"))),
            reverse=True,
        )
        choice = candidates[0] if candidates else preferred
    _resolved[preferred] = choice
    return choice


def _pace() -> None:
    global _last_call
    gap = 60.0 / max(config.REQUESTS_PER_MINUTE, 1)
    wait = _last_call + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def generate_json(prompt: str, schema, *, model: str, system: str = "", temperature: float = 0.0):
    """Call Gemini and return the parsed JSON object, with retries and backoff."""
    from google.genai import types

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=system or None,
    )
    last_error = None
    for attempt in range(config.MAX_RETRIES):
        _pace()
        try:
            resp = client().models.generate_content(model=model, contents=prompt, config=cfg)
            USAGE.add(model, resp)
            if getattr(resp, "parsed", None) is not None:
                parsed = resp.parsed
                return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
            return json.loads(resp.text)
        except Exception as exc:  # network, quota, safety block, invalid JSON
            last_error = exc
            USAGE.failures += 1
            msg = str(exc).lower()
            if "api key" in msg or "permission" in msg or "not found" in msg and "model" in msg:
                raise LLMUnavailable(str(exc)) from exc
            time.sleep(min(60, (2 ** attempt) + random.random() * 2))
    raise LLMUnavailable(f"Gemini failed after {config.MAX_RETRIES} attempts: {last_error}")
