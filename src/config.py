"""Central settings for the Voice of Customer Copilot.

Everything a product manager might want to tune lives here, so nobody has to
hunt through the code. Values can be overridden with environment variables
(put them in the .env file next to this project).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
INBOX_DIR = DATA / "inbox"            # new files dropped here by the daily refresh
PROCESSED = DATA / "processed"
CACHE_DIR = DATA / "cache"
CONTEXT_DIR = DATA / "context"        # product release notes and similar context
DECISIONS_DIR = DATA / "decisions"    # product manager decisions (approve / modify / reject)
EVAL_DIR = DATA / "eval"              # answer key lives here; ONLY evaluate.py reads it
REPORTS = ROOT / "reports"

for _p in (INBOX_DIR, PROCESSED, CACHE_DIR, CONTEXT_DIR, DECISIONS_DIR, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---- Large language model settings (Gemini) -------------------------------
def _read_key() -> str:
    """Key lookup order: environment variable or .env (laptop, GitHub Actions),
    then Streamlit Secrets (Streamlit Community Cloud). The key is never stored in code."""
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key.strip()
    try:
        import streamlit as st

        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:  # no Streamlit, or no secrets file: fine, run without the key
        return ""


GEMINI_API_KEY = _read_key()
# Aliases always point at Google's current stable model of each tier.
# If your account does not have an alias, put an exact model name here instead.
TAGGING_MODEL = os.getenv("VOC_TAGGING_MODEL", "gemini-flash-latest")
REASONING_MODEL = os.getenv("VOC_REASONING_MODEL", "gemini-pro-latest")
BATCH_SIZE = int(os.getenv("VOC_BATCH_SIZE", "10"))
REQUESTS_PER_MINUTE = int(os.getenv("VOC_REQUESTS_PER_MINUTE", "8"))
MAX_RETRIES = int(os.getenv("VOC_MAX_RETRIES", "5"))
PROMPT_VERSION = "tagging-v4"          # bump when the prompt changes; invalidates the cache

# ---- Data assumptions ------------------------------------------------------
# Relative dates such as "3 weeks ago" are counted back from the export date.
EXPORT_DATE = date.fromisoformat(os.getenv("VOC_EXPORT_DATE", "2026-11-01"))
WINDOW_START = date.fromisoformat(os.getenv("VOC_WINDOW_START", "2025-12-01"))
WINDOW_END = date.fromisoformat(os.getenv("VOC_WINDOW_END", "2026-10-31"))
OWN_BRAND = "Zomato"


def llm_available() -> bool:
    return bool(GEMINI_API_KEY)
