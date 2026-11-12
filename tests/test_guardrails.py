"""Tests for the guardrails, using a fake Gemini so they run offline and free.

    python -m pytest -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config, enrich, insights, llm
from src.ingest import parse_date, parse_rating, redact_personal_details, rule_noise


# ---------------- cleaning ----------------
def test_ratings_in_every_format():
    for raw, want in [("1", 1), ("1.0", 1), ("1/5", 1), ("★☆☆☆☆", 1), ("one", 1), ("1 star", 1), ("4 stars", 4), ("", None)]:
        assert parse_rating(raw) == want


def test_dates_in_many_formats():
    assert str(parse_date("Tue Jun 02 04:36:04 +0000 2026")[0]) == "2026-06-02"
    assert str(parse_date("12/23/2025")[0]) == "2025-12-23"
    assert str(parse_date("28/04/2026")[0]) == "2026-04-28"
    assert parse_date("3 weeks ago")[1] == "relative_estimate"
    assert parse_date("Mar 14")[0].year == 2026


def test_personal_details_removed():
    text, changed = redact_personal_details("Call me on 9876543210 or mail ananya75@example.com")
    assert changed and "9876543210" not in text and "@example.com" not in text


def test_competitor_comparison_is_not_noise():
    assert rule_noise("Food was cold. Switching to Swiggy.", "Zomato") is None
    assert rule_noise("Swiggy One is better value than any other membership", "Swiggy") == "other_brand"


# ---------------- labelling guardrails ----------------
def test_invented_quote_is_dropped():
    lab = enrich.validate_label({"l2_issues": ["Speed of Delivery"], "evidence_quote": "the food took forever", "confidence": 0.9},
                                "Order stuck for 95 minutes due to rain.")
    assert lab["evidence_quote"] == "" and lab["evidence_verified"] is False


def test_real_quote_is_kept():
    lab = enrich.validate_label({"l2_issues": ["Speed of Delivery"], "evidence_quote": "stuck for 95 minutes", "confidence": 0.9},
                                "Order stuck for 95 minutes due to rain.")
    assert lab["evidence_verified"] is True


def test_unknown_issue_name_is_rejected():
    lab = enrich.validate_label({"l2_issues": ["Vibes"], "confidence": 0.9}, "meh")
    assert lab["l2_issues"] == ["No Specific Issue Stated"] and lab["confidence"] <= 0.3


def _fake_batch_answer(prompt, schema, **kw):
    """Fake Gemini: labels every record except the last one, and invents one extra record ID."""
    import json
    import re

    items = json.loads(prompt.split("\n\n", 1)[1])
    labels = []
    for it in items[:-1]:
        labels.append({"record_id": it["record_id"], "is_relevant": True, "noise_type": "none", "sentiment": "Negative",
                       "l2_issues": ["Speed of Delivery"], "is_feature_request": False, "feature_request_text": "",
                       "severity": 2, "journey_stage": "Delivery", "churn_risk": False, "is_sarcastic": False,
                       "evidence_quote": it["text"][:15], "confidence": 0.8})
    if labels:
        labels.append({**labels[0], "record_id": "R99999"})
    return {"labels": labels}


def test_missing_records_fall_back_and_invented_ids_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "resolve_model", lambda m, f: "fake-model")
    monkeypatch.setattr(llm, "generate_json", _fake_batch_answer)
    df = pd.DataFrame({"record_id": ["R00001", "R00002", "R00003"], "platform": "Reddit", "rating": [None, 1.0, 5.0],
                       "text": ["Order took 90 minutes", "Food was cold", "Great app, love it"]})
    out = enrich.label_with_gemini(df)
    assert set(out) == {"R00001", "R00002", "R00003"}           # nobody left unlabelled
    assert "R99999" not in out                                    # invented ID ignored
    assert out["R00001"]["label_source"].startswith("gemini")
    assert out["R00003"]["label_source"] == "rules_fallback"     # the skipped record was retried, then fell back


def test_gemini_outage_falls_back_to_rules(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "resolve_model", lambda m, f: "fake-model")

    def down(*a, **k):
        raise llm.LLMUnavailable("quota exceeded")
    monkeypatch.setattr(llm, "generate_json", down)
    df = pd.DataFrame({"record_id": ["R00001", "R00002"], "platform": "Reddit", "rating": [None, None],
                       "text": ["Platform fee hiked again to ₹17!", "App crashes on the coupon page"]})
    out = enrich.label_with_gemini(df)
    assert {v["label_source"] for v in out.values()} == {"rules_fallback"}
    assert out["R00001"]["l2_issues"] == ["Fees & Surcharges"]


# ---------------- insight guardrails ----------------
PACK = {"issue": "Speed of Delivery", "negative_mentions": 565, "negative_feedback_total": 3293, "share_of_negative_pct": 17.2,
        "quotes": [{"record_id": "R00010", "text": "a"}, {"record_id": "R00011", "text": "b"}], "contradicting_quotes": []}


def test_invented_number_in_headline_is_caught():
    draft = {"headline": "742 customers complained about delivery", "hypotheses": []}
    assert insights.verify(draft, PACK)["numbers_verified"] is False


def test_true_numbers_pass():
    draft = {"headline": "565 of 3293 negative reviews (17.2%) describe late delivery", "hypotheses": []}
    assert insights.verify(draft, PACK)["numbers_verified"] is True


def test_invented_citation_is_removed():
    draft = {"headline": "565 reviews", "hypotheses": [{"statement": "Riders are scarce at peak", "supporting_record_ids": ["R00010", "R77777"]}]}
    check = insights.verify(draft, PACK)
    assert draft["hypotheses"][0]["supporting_record_ids"] == ["R00010"]
    assert check["citations_verified"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
