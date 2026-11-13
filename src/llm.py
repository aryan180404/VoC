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


class FatalLLMError(LLMUnavailable):
    """An error that retrying cannot fix (wrong model, bad key, no quota). The run stops at once."""


class QuotaExhausted(FatalLLMError):
    """Gemini refused because the key has no quota left."""


class ModelNotAvailable(FatalLLMError):
    """The chosen model does not exist or is closed to this account."""


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
_working_level: dict[str, int] = {}  # remembers which request settings each model accepts


def client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise LLMUnavailable("GEMINI_API_KEY is not set. Add it to the .env file.")
        from google import genai

        from google.genai import types as _t

        # Stop waiting on any single call after 90 seconds so a stalled connection is retried.
        _client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=_t.HttpOptions(timeout=90_000))
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


def generate_json(prompt: str, schema, *, model: str, system: str = "", temperature: float = 0.0, fast: bool = False):
    """Call Gemini and return the parsed JSON object, with retries and backoff."""
    from google.genai import types

    # Request settings, simplest last. If Gemini rejects a request as an invalid argument,
    # the next level drops the setting this model may not support.
    #   level 0: structured schema + thinking switched off (fastest)
    #   level 1: structured schema, default thinking
    #   level 2: plain JSON schema format, default thinking
    #   level 3: JSON output only; the schema is checked here in Python instead
    level = max(0 if fast else 1, _working_level.get(model, 0))

    def build(lvl):
        kw = dict(temperature=temperature, response_mime_type="application/json",
                  system_instruction=system or None)
        if lvl == 0:
            kw["response_schema"] = schema
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif lvl == 1:
            kw["response_schema"] = schema
        elif lvl == 2 and hasattr(schema, "model_json_schema"):
            kw["response_json_schema"] = schema.model_json_schema()
        return types.GenerateContentConfig(**kw)

    def from_text(text):
        text = (text or "").strip()
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data).model_dump()
        return data

    cfg = build(level)
    last_error = None
    for attempt in range(config.MAX_RETRIES):
        _pace()
        try:
            resp = client().models.generate_content(model=model, contents=prompt, config=cfg)
            USAGE.add(model, resp)
            if getattr(resp, "parsed", None) is not None:
                parsed = resp.parsed
                return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
            return from_text(resp.text)
        except Exception as exc:  # network, quota, safety block, invalid JSON
            last_error = exc
            USAGE.failures += 1
            print(f"    [gemini attempt {attempt + 1}/{config.MAX_RETRIES} failed: {str(exc)[:1500]}]", flush=True)
            low = str(exc).lower()
            if ("invalid_argument" in low or low.startswith("400")) and level < 3:
                level += 1
                _working_level[model] = level
                cfg = build(level)
                print(f"    [switching to simpler request settings, level {level} of 3]", flush=True)
                continue
            flat = low.replace(" ", "").replace("_", "")
            if "resourceexhausted" in flat or "429" in low:
                # A daily quota or a zero limit never clears during this run, so stop at once.
                # A per-minute limit does clear, so wait a full minute and try again a few times.
                if "perday" in flat or "limit: 0" in low or attempt >= 3:
                    raise QuotaExhausted(
                        "Gemini says this key has no quota left for this model. Check "
                        "https://aistudio.google.com/rate-limit , pick another model with VOC_TAGGING_MODEL, "
                        "turn on billing, or try again after the daily reset.") from exc
                time.sleep(35)
            msg = str(exc).lower()
            if "not_found" in msg or "no longer available" in msg or "404" in msg[:12]:
                raise ModelNotAvailable(
                    f"Gemini says the model '{model}' is not available to this account. "
                    "Set the repository variable VOC_TAGGING_MODEL (and VOC_REASONING_MODEL) to a model "
                    "listed on https://aistudio.google.com/rate-limit . Gemini's message: " + str(exc)[:300]) from exc
            if "api key" in msg or "permission_denied" in msg or "api_key_invalid" in msg:
                raise FatalLLMError("Gemini rejected the key: " + str(exc)[:300]) from exc
            time.sleep(min(60, (2 ** attempt) + random.random() * 2))
    raise LLMUnavailable(f"Gemini failed after {config.MAX_RETRIES} attempts: {last_error}")
