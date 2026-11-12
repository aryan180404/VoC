"""Evaluation Centre: score the copilot against the hidden answer key.

The answer key is read ONLY here. The labeller never sees it.

Honest-scoring rules built in:
  * Holdout split. Records are split 30 / 70 by a fixed hash of record_id. The
    keyword rules were written while looking at the 30% development split only,
    so the headline numbers are reported on the untouched 70% holdout split.
  * Answer key audit. Copies of the same review text should carry the same
    label. Where the key disagrees with itself, those rows are reported, because
    an evaluation is only as good as its answer key.

Run:  python -m src.evaluate
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .ingest import dedupe_key
from .taxonomy import L2_TO_L1

INCIDENTS = {  # what the README says is planted, mapped to the issue we expect to spike
    "INC-A app update coupon crash": ("App Stability & Bugs", "2026-03"),
    "INC-B platform fee hike": ("Fees & Surcharges", "2026-06"),
    "INC-C monsoon delays and rain fee": ("Speed of Delivery", "2026-07"),
    "INC-D Gold renewal auto-debit": ("Gold Hidden Fees/Platform Charges", "2026-09"),
}


def split_of(record_id: str) -> str:
    return "development" if int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 10 < 3 else "holdout"


def load_answer_key() -> pd.DataFrame:
    ak = pd.read_csv(config.EVAL_DIR / "zomato_voice_of_customer_answer_key.csv", dtype=str, keep_default_na=False)
    ak["true_issues"] = ak.L2_Specific_Issue.map(lambda s: [x.strip() for x in s.split(";") if x.strip() and x.strip() != "Not Relevant"])
    return ak


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": tp + fn}


def audit_answer_key(df: pd.DataFrame, ak: pd.DataFrame, min_copies: int = 3, min_minority: float = 0.25) -> pd.DataFrame:
    """Find review texts that appear several times but get DIFFERENT sentiment in the answer key.

    Copies of the same sentence should carry the same label. When at least a quarter of the copies
    disagree, the key itself is unreliable for that text, so those rows are listed and the headline
    accuracy is also reported without them. We do not guess which copy is right.
    """
    core = df.text.map(dedupe_key)
    frame = pd.DataFrame({"core": core, "sentiment": ak.sentiment, "language": ak.language, "record_id": df.record_id,
                          "text": df.text, "relevant": ak.is_noise == "No"})
    frame = frame[frame.relevant & (frame.core.str.len() >= 8)]
    rows = []
    for _, g in frame.groupby("core"):
        counts = Counter(g.sentiment)
        if len(g) >= min_copies and len(counts) > 1:
            minority = 1 - counts.most_common(1)[0][1] / len(g)
            if minority >= min_minority:
                split = ", ".join(f"{k} {v}" for k, v in counts.most_common())
                for r in g.itertuples():
                    rows.append({"record_id": r.record_id, "text": r.text, "language": r.language,
                                 "answer_key_sentiment": r.sentiment, "labels_given_to_identical_copies": split,
                                 "copies": len(g)})
    return pd.DataFrame(rows)


def evaluate(enriched: pd.DataFrame, ak: pd.DataFrame, suspect_ids: set[str]) -> dict:
    from .analytics import fix_lists

    df = fix_lists(enriched.reset_index(drop=True))
    assert len(df) == len(ak) and (df.source_id.values == ak.ID.values).all(), "Rows are out of order with the answer key"
    df["split"] = df.record_id.map(split_of)
    truth_rel = ak.is_noise == "No"
    pred_rel = df.is_relevant

    out: dict = {"engine": str(df.engine.iloc[0]) if "engine" in df else "unknown"}
    # ---- cleaning (whole dataset; these rules were tuned openly) ----
    out["cleaning"] = {
        "noise_removal": prf(int((~pred_rel & ~truth_rel).sum()), int((~pred_rel & truth_rel).sum()), int((pred_rel & ~truth_rel).sum())),
        "platform_accuracy": round(float((df.platform == ak.true_platform).mean()), 3),
        "brand_accuracy_on_relevant": round(float((df.brand.replace("Unknown", config.OWN_BRAND)[truth_rel] == ak.true_brand[truth_rel]).mean()), 3),
        "language_accuracy": round(float((df.language == ak.language).mean()), 3),
    }
    d = pd.to_datetime(df.date).dt.strftime("%Y-%m-%d")
    tdate = pd.to_datetime(ak.true_date)
    gap = (pd.to_datetime(df.date) - tdate).dt.days.abs()
    out["cleaning"]["date_exact_match"] = round(float((d == ak.true_date).mean()), 3)
    out["cleaning"]["date_within_31_days"] = round(float((gap <= 31).mean()), 3)
    out["cleaning"]["date_missing"] = int(df.date.isna().sum())
    out["cleaning"]["date_month_match"] = round(float((d.str[:7] == ak.true_date.str[:7]).mean()), 3)
    pii = ak.contains_personal_details == "Yes"
    out["cleaning"]["personal_details_removed"] = prf(int((pii & df.had_personal_details).sum()), int((~pii & df.had_personal_details).sum()),
                                                      int((pii & ~df.had_personal_details).sum()))
    # duplicates: is each true duplicate grouped with its original?
    group = df.duplicate_of.where(df.duplicate_of != "", df.record_id)
    true_dup = ak.duplicate_of != ""
    hits, true_rows = 0, set()
    for i in ak.index[true_dup]:
        originals = [j for j in ak.index[ak.ID == ak.duplicate_of[i]] if j != i]
        true_rows |= {i, *originals}
        hits += any(group[i] == group[j] for j in originals)
    flagged = df.index[df.is_duplicate]
    out["cleaning"]["duplicates"] = {"recall": round(hits / max(true_dup.sum(), 1), 3),
                                     "precision": round(sum(i in true_rows for i in flagged) / max(len(flagged), 1), 3),
                                     "true_duplicates": int(true_dup.sum()), "flagged": int(len(flagged))}

    # ---- classification, on records that are truly relevant ----
    def score(mask: pd.Series) -> dict:
        m = mask & truth_rel & pred_rel
        sub, tru = df[m], ak[m]
        res = {"records": int(m.sum())}
        res["sentiment_accuracy"] = round(float((sub.sentiment == tru.sentiment).mean()), 3) if len(sub) else None
        per = {}
        for s in ("Positive", "Neutral", "Negative"):
            per[s] = prf(int(((sub.sentiment == s) & (tru.sentiment == s)).sum()), int(((sub.sentiment == s) & (tru.sentiment != s)).sum()),
                         int(((sub.sentiment != s) & (tru.sentiment == s)).sum()))
        res["sentiment_macro_f1"] = round(float(np.mean([v["f1"] for v in per.values()])), 3)
        res["sentiment_per_class"] = per
        tp = fp = fn = 0
        primary_hit = any_hit = exact_cat = 0
        for p_iss, t_iss in zip(sub.l2_issues, tru.true_issues):
            ps, ts = set(p_iss), set(t_iss)
            tp += len(ps & ts); fp += len(ps - ts); fn += len(ts - ps)
            primary_hit += bool(p_iss) and p_iss[0] in ts
            any_hit += bool(ps & ts)
            exact_cat += {L2_TO_L1[x] for x in ps} == {L2_TO_L1[x] for x in ts}
        n = max(len(sub), 1)
        res["issue_micro"] = prf(tp, fp, fn)
        res["primary_issue_correct"] = round(primary_hit / n, 3)
        res["any_issue_correct"] = round(any_hit / n, 3)
        res["category_set_exact"] = round(exact_cat / n, 3)
        fr_t, fr_p = tru.is_feature_request == "Yes", sub.is_feature_request.astype(bool)
        res["feature_request"] = prf(int((fr_t & fr_p).sum()), int((~fr_t & fr_p).sum()), int((fr_t & ~fr_p).sum()))
        return res

    everything = pd.Series(True, index=df.index)
    out["classification_holdout"] = score(df.split == "holdout")
    out["classification_all"] = score(everything)
    clean_key = ~df.record_id.isin(suspect_ids)
    out["classification_holdout_excluding_suspect_key_rows"] = score((df.split == "holdout") & clean_key)
    out["by_language"] = {lang: score(ak.language == lang)["sentiment_accuracy"] for lang in sorted(ak.language.unique())}
    out["by_language_issue"] = {lang: score(ak.language == lang)["primary_issue_correct"] for lang in sorted(ak.language.unique())}
    out["sarcasm_sentiment_accuracy"] = score(ak.messiness_applied.str.contains("sarcasm"))["sentiment_accuracy"]
    out["rating_text_mismatch_sentiment_accuracy"] = score(ak.messiness_applied.str.contains("rating_text_mismatch"))["sentiment_accuracy"]

    # per-issue table
    per_issue = []
    m = truth_rel & pred_rel
    for issue in L2_TO_L1:
        t = ak.true_issues[m].map(lambda v: issue in v)
        p = df.l2_issues[m].map(lambda v: issue in v)
        per_issue.append({"issue": issue, "category": L2_TO_L1[issue], **prf(int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()))})
    out["per_issue"] = per_issue
    return out


def error_sample(enriched: pd.DataFrame, ak: pd.DataFrame, n: int = 80) -> pd.DataFrame:
    from .analytics import fix_lists

    df = fix_lists(enriched.reset_index(drop=True))
    m = (ak.is_noise == "No") & df.is_relevant
    wrong = m & ((df.sentiment != ak.sentiment) | ~pd.Series([bool(set(p) & set(t)) for p, t in zip(df.l2_issues, ak.true_issues)]))
    s = df[wrong].assign(true_sentiment=ak.sentiment[wrong], true_issues=ak.L2_Specific_Issue[wrong], language=ak.language[wrong],
                         messiness=ak.messiness_applied[wrong])
    s["predicted_issues"] = s.l2_issues.map("; ".join)
    cols = ["record_id", "text", "sentiment", "true_sentiment", "predicted_issues", "true_issues", "language", "messiness", "label_source"]
    return s[cols].sample(min(n, len(s)), random_state=7).rename(columns={"sentiment": "predicted_sentiment"})


def truth_as_enriched(clean: pd.DataFrame, ak: pd.DataFrame) -> pd.DataFrame:
    """Build the table the dashboard WOULD show if every label were perfect."""
    t = clean.reset_index(drop=True).copy()
    t["is_relevant"] = (ak.is_noise == "No").values
    t["is_duplicate"] = (ak.duplicate_of != "").values
    t["date"] = pd.to_datetime(ak.true_date).dt.date.values
    t["sentiment"] = ak.sentiment.values
    t["l2_issues"] = ak.true_issues.values
    t["l1_categories"] = [sorted({L2_TO_L1[i] for i in v}) for v in ak.true_issues]
    t["severity"], t["churn_risk"], t["confidence"] = 2, False, 1.0
    return t


def compare_dashboard_numbers(active: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Product managers read aggregates, not single labels. How far off are the dashboard numbers?"""
    from . import analytics

    a, t = analytics.compute_all(active), analytics.compute_all(truth)
    m = a["monthly"].merge(t["monthly"], on="month", suffixes=("_copilot", "_truth"))
    share_a = a["tags"][a["tags"].is_negative].issue.value_counts(normalize=True)
    share_t = t["tags"][t["tags"].is_negative].issue.value_counts(normalize=True)
    issues = share_t.index.union(share_a.index)
    diff = (share_a.reindex(issues).fillna(0) - share_t.reindex(issues).fillna(0)).abs()
    top_a = [i for i in share_a.index if i != "No Specific Issue Stated"][:5]
    top_t = [i for i in share_t.index if i != "No Specific Issue Stated"][:5]
    return {
        "monthly_negative_share_mean_abs_error_points": round(float((m.negative_share_copilot - m.negative_share_truth).abs().mean() * 100), 2),
        "monthly": m[["month", "negative_share_copilot", "negative_share_truth"]].round(3).to_dict("records"),
        "issue_share_mean_abs_error_points": round(float(diff.mean() * 100), 2),
        "issue_share_max_abs_error_points": round(float(diff.max() * 100), 2),
        "issue_with_largest_error": str(diff.idxmax()),
        "top5_copilot": top_a, "top5_truth": top_t, "top5_overlap": len(set(top_a) & set(top_t)),
        "reference_findings_copilot": analytics.reference_findings(active),
        "reference_findings_truth": analytics.reference_findings(truth),
        "spikes_truth": [(x["issue"], x["month"]) for x in t["spike_alerts"]],
        "spikes_copilot": [(x["issue"], x["month"]) for x in a["spike_alerts"]],
    }


def incident_detection(trends: dict) -> list[dict]:
    """Did the trend engine flag each planted incident in the right month?"""
    alerts = trends.get("spike_alerts", [])
    rows = []
    for name, (issue, month) in INCIDENTS.items():
        hit = [a for a in alerts if a["issue"] == issue and a["month"] == month]
        rows.append({"incident": name, "expected_issue": issue, "expected_month": month, "detected": bool(hit),
                     "detected_lift": hit[0]["lift"] if hit else None})
    return rows


def run() -> dict:
    from . import analytics

    ak = load_answer_key()
    audit = audit_answer_key(pd.read_parquet(config.PROCESSED / "clean.parquet"), ak)
    suspect = set(audit.record_id) if len(audit) else set()
    results = {"answer_key_audit": {"suspect_rows": int(len(audit)),
                                    "distinct_texts": int(audit.text.map(dedupe_key).nunique()) if len(audit) else 0,
                                    "by_language": audit.language.value_counts().to_dict() if len(audit) else {}}}
    engines = {}
    for path in sorted(config.PROCESSED.glob("enriched_*.parquet")):
        if "trial" in path.name:
            continue
        enr = pd.read_parquet(path)
        engines[path.stem.replace("enriched_", "")] = evaluate(enr, ak, suspect)
    results["engines"] = engines
    active = pd.read_parquet(config.PROCESSED / "enriched.parquet")
    trends = analytics.compute_all(active)
    results["incident_detection"] = incident_detection(trends)
    results["reference_findings"] = analytics.reference_findings(active)
    clean = pd.read_parquet(config.PROCESSED / "clean.parquet")
    results["dashboard_numbers_vs_truth"] = compare_dashboard_numbers(analytics.fix_lists(active), truth_as_enriched(clean, ak))
    results["active_engine"] = str(active.engine.iloc[0])

    (config.REPORTS / "evaluation_results.json").write_text(json.dumps(results, indent=2, default=str))
    audit.to_csv(config.REPORTS / "answer_key_audit.csv", index=False)
    errs = error_sample(active, ak)
    errs.to_csv(config.REPORTS / "error_sample.csv", index=False)
    write_evaluation_sheet(results, audit, errs)
    return results


def write_evaluation_sheet(results: dict, audit: pd.DataFrame, errs: pd.DataFrame) -> None:
    """The Excel evaluation sheet: one of the four portfolio deliverables."""
    path = config.REPORTS / "evaluation_sheet.xlsx"
    rows = []
    for eng, r in results["engines"].items():
        c, h = r["cleaning"], r["classification_holdout"]
        rows += [
            (eng, "Cleaning", "Noise removal F1", c["noise_removal"]["f1"], "≥ 0.90"),
            (eng, "Cleaning", "Duplicate recall", c["duplicates"]["recall"], "≥ 0.85"),
            (eng, "Cleaning", "Duplicate precision", c["duplicates"]["precision"], "≥ 0.80"),
            (eng, "Cleaning", "Personal details removed (recall)", c["personal_details_removed"]["recall"], "1.00"),
            (eng, "Cleaning", "Platform accuracy", c["platform_accuracy"], "≥ 0.98"),
            (eng, "Cleaning", "Date in correct month", c["date_month_match"], "≥ 0.90"),
            (eng, "Classification (holdout)", "Sentiment accuracy", h["sentiment_accuracy"], "≥ 0.85"),
            (eng, "Classification (holdout)", "Sentiment macro F1", h["sentiment_macro_f1"], "≥ 0.80"),
            (eng, "Classification (holdout)", "Primary issue correct", h["primary_issue_correct"], "≥ 0.80"),
            (eng, "Classification (holdout)", "Issue micro F1 (multi-label)", h["issue_micro"]["f1"], "≥ 0.75"),
            (eng, "Classification (holdout)", "Feature request F1", h["feature_request"]["f1"], "≥ 0.80"),
            (eng, "Robustness", "Sarcasm sentiment accuracy", r["sarcasm_sentiment_accuracy"], "≥ 0.75"),
            (eng, "Robustness", "Rating-text mismatch sentiment accuracy", r["rating_text_mismatch_sentiment_accuracy"], "≥ 0.80"),
        ]
    summary = pd.DataFrame(rows, columns=["Engine", "Area", "Metric", "Score", "Target"])
    summary["Meets target"] = [
        (s is not None) and s >= float(re.sub(r"[^\d.]", "", t)) for s, t in zip(summary.Score, summary.Target)
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, "Summary", index=False)
        pd.DataFrame(results["incident_detection"]).to_excel(xw, "Incident detection", index=False)
        for eng, r in results["engines"].items():
            pd.DataFrame(r["per_issue"]).to_excel(xw, f"Per issue - {eng}"[:31], index=False)
            pd.DataFrame([{"language": k, "sentiment_accuracy": v, "primary_issue_correct": r["by_language_issue"][k]}
                          for k, v in r["by_language"].items()]).to_excel(xw, f"By language - {eng}"[:31], index=False)
        errs.assign(reviewer_verdict="", reviewer_note="").to_excel(xw, "Error review", index=False)
        audit.to_excel(xw, "Answer key audit", index=False)
        pd.DataFrame({
            "Rubric for insight quality (score each insight 1 to 5)": [
                "Accuracy: the numbers match the dashboard and the cited reviews",
                "Evidence: every claim has at least 3 supporting quotes that really say it",
                "Specificity: names a concrete behaviour, not 'users are unhappy'",
                "Actionability: a product manager could start work from it",
                "Honesty: hypotheses are labelled as hypotheses; contradicting reviews shown",
            ]}).to_excel(xw, "Insight rubric", index=False)
        pd.DataFrame(columns=["record_id", "text", "your_sentiment", "your_issue", "model_sentiment", "model_issue", "agree?"]).to_excel(
            xw, "Human labelling (150 rows)", index=False)
    try:  # basic formatting
        from openpyxl import load_workbook
        wb = load_workbook(path)
        for ws in wb.worksheets:
            for col in ws.columns:
                width = min(80, max(10, max(len(str(c.value or "")) for c in col[:200]) + 2))
                ws.column_dimensions[col[0].column_letter].width = width
            ws.freeze_panes = "A2"
        wb.save(path)
    except Exception:
        pass


if __name__ == "__main__":
    res = run()
    for eng, r in res["engines"].items():
        print(f"\n== {eng} ==")
        print(json.dumps({k: r[k] for k in ("cleaning",)}, indent=1))
        h = r["classification_holdout"]
        print({k: h[k] for k in ("records", "sentiment_accuracy", "sentiment_macro_f1", "primary_issue_correct", "any_issue_correct", "category_set_exact")},
              "issue F1", h["issue_micro"]["f1"], "feature request", h["feature_request"])
        print("holdout excl suspect:", r["classification_holdout_excluding_suspect_key_rows"]["sentiment_accuracy"])
        print("by language", r["by_language"], "sarcasm", r["sarcasm_sentiment_accuracy"])
    print("audit", res["answer_key_audit"])
    print("incidents", res["incident_detection"])
    print("reference", res["reference_findings"])
    dn = res["dashboard_numbers_vs_truth"]
    print({k: v for k, v in dn.items() if k not in ("monthly",)})
