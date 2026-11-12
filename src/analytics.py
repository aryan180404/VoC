"""All numbers on the dashboard are computed here, by plain code, from labels.

Key reliability choices:
  * Count each conversation once: duplicates and noise are excluded.
  * Trends use RATES (share of all feedback that month), not raw counts, so a
    month with more reviews overall does not look like a worse product.
  * Every alert carries its sample size and a significance test, so small
    numbers do not create false alarms.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .taxonomy import L2_TO_L1

CUSTOMER_SUPPORT = "Customer Support & Resolution"


def fix_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet returns list columns as numpy arrays; turn them back into plain lists."""
    for col in ("l2_issues", "l1_categories"):
        if col in df:
            df[col] = df[col].map(lambda v: [] if v is None or (isinstance(v, float) and math.isnan(v)) else list(v))
    return df


def load_enriched(path=None) -> pd.DataFrame:
    from . import config

    return fix_lists(pd.read_parquet(path or config.PROCESSED / "enriched.parquet"))


def base_table(enriched: pd.DataFrame) -> pd.DataFrame:
    enriched = fix_lists(enriched.copy())
    b = enriched[enriched.is_relevant & ~enriched.is_duplicate].copy()
    from . import config

    b["date"] = pd.to_datetime(b.date)
    outside = (b.date < pd.Timestamp(config.WINDOW_START)) | (b.date > pd.Timestamp(config.WINDOW_END))
    b.loc[outside, "date"] = pd.NaT  # impossible dates are kept in totals but left out of trends
    b["month"] = b.date.dt.strftime("%Y-%m")
    b["week"] = b.date.dt.to_period("W-SUN").dt.start_time
    b["is_negative"] = b.sentiment == "Negative"
    b["mentions_gold"] = b.text.str.contains(r"\bgold\b", case=False, regex=True)
    return b


def two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided p-value: is rate x2/n2 different from x1/n1?"""
    if min(n1, n2) == 0:
        return 1.0
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0
    if se == 0:
        return 1.0
    z = (x2 / n2 - x1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def issue_long(b: pd.DataFrame) -> pd.DataFrame:
    """One row per (record, issue). Mirrors the reference project's tag fact table."""
    t = b[["record_id", "l2_issues", "sentiment", "is_negative", "month", "week", "date", "platform", "severity",
           "churn_risk", "confidence", "text", "likes", "language"]].explode("l2_issues")
    t = t.rename(columns={"l2_issues": "issue"}).dropna(subset=["issue"])
    t["category"] = t.issue.map(L2_TO_L1)
    return t


def monthly_overview(b: pd.DataFrame) -> pd.DataFrame:
    dated = b.dropna(subset=["date"])
    g = dated.groupby("month")
    out = pd.DataFrame({
        "feedback": g.size(),
        "negative": g.is_negative.sum(),
        "positive": g.apply(lambda x: (x.sentiment == "Positive").sum(), include_groups=False),
    })
    out["negative_share"] = out.negative / out.feedback
    return out.reset_index()


def monthly_issue_rates(b: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    totals = b.dropna(subset=["date"]).groupby("month").size().rename("total")
    neg = t[t.is_negative].dropna(subset=["date"])
    c = neg.groupby(["issue", "month"]).size().rename("count").reset_index()
    grid = pd.MultiIndex.from_product([sorted(t.issue.unique()), totals.index], names=["issue", "month"]).to_frame(index=False)
    c = grid.merge(c, how="left").fillna({"count": 0}).merge(totals.reset_index(), on="month")
    c["rate"] = c["count"] / c.total
    return c


def spike_alerts(rates: pd.DataFrame, min_count: int = 12, min_lift: float = 1.6, max_p: float = 0.01) -> list[dict]:
    """Flag months where an issue's share of feedback jumps well above its usual level."""
    alerts = []
    for issue, g in rates[rates.issue != "No Specific Issue Stated"].groupby("issue"):
        g = g.sort_values("month").reset_index(drop=True)
        for i, row in g.iterrows():
            others = g.drop(index=i)
            prior = g.iloc[max(0, i - 3):i]
            ref = prior if len(prior) >= 2 else others
            base_rate = (ref["count"].sum() / ref.total.sum()) if ref.total.sum() else 0
            if row["count"] < min_count or base_rate == 0:
                continue
            lift = row.rate / base_rate
            p = two_proportion_p(int(ref["count"].sum()), int(ref.total.sum()), int(row["count"]), int(row.total))
            if lift >= min_lift and p <= max_p:
                alerts.append({"issue": issue, "category": L2_TO_L1[issue], "month": row.month, "count": int(row["count"]),
                               "rate": round(row.rate, 4), "baseline_rate": round(base_rate, 4), "lift": round(lift, 2),
                               "p_value": float(f"{p:.2g}")})
    return sorted(alerts, key=lambda a: (a["month"], -a["lift"]))


def emerging_and_declining(rates: pd.DataFrame, recent_months: int = 2, earlier_months: int = 3) -> pd.DataFrame:
    months = sorted(rates.month.unique())
    recent, earlier = months[-recent_months:], months[-(recent_months + earlier_months):-recent_months]
    rows = []
    for issue, g in rates[rates.issue != "No Specific Issue Stated"].groupby("issue"):
        r, e = g[g.month.isin(recent)], g[g.month.isin(earlier)]
        rr = r["count"].sum() / max(r.total.sum(), 1)
        er = e["count"].sum() / max(e.total.sum(), 1)
        p = two_proportion_p(int(e["count"].sum()), int(e.total.sum()), int(r["count"].sum()), int(r.total.sum()))
        change = (rr - er) / er if er else (np.inf if rr else 0)
        status = "Stable"
        if p < 0.05 and change > 0.25:
            status = "New" if er == 0 else "Rising"
        elif p < 0.05 and change < -0.25:
            status = "Declining"
        rows.append({"issue": issue, "category": L2_TO_L1[issue], "earlier_rate": er, "recent_rate": rr,
                     "earlier_count": int(e["count"].sum()), "recent_count": int(r["count"].sum()),
                     "change": change, "p_value": p, "status": status})
    out = pd.DataFrame(rows).sort_values("change", ascending=False)
    out.attrs["recent"], out.attrs["earlier"] = recent, earlier
    return out


def opportunity_table(b: pd.DataFrame, t: pd.DataFrame, trend: pd.DataFrame,
                      weights: dict | None = None) -> pd.DataFrame:
    """Rank customer problems. Every component is shown so the ranking can be explained and challenged."""
    w = weights or {"frequency": 0.40, "trend": 0.25, "severity": 0.20, "churn": 0.15}
    neg_total = int(b.is_negative.sum())
    neg = t[t.is_negative & ~t.issue.isin(["No Specific Issue Stated"])]
    g = neg.groupby("issue")
    df = pd.DataFrame({
        "negative_mentions": g.size(),
        "severity": g.severity.mean(),
        "churn_share": g.churn_risk.mean(),
        "label_confidence": g.confidence.mean(),
        "platforms": g.platform.nunique(),
    })
    all_mentions = t.groupby("issue").size()
    pos_mentions = t[t.sentiment == "Positive"].groupby("issue").size()
    df["total_mentions"] = all_mentions.reindex(df.index).fillna(0).astype(int)
    df["positive_mentions"] = pos_mentions.reindex(df.index).fillna(0).astype(int)
    df["share_of_negative"] = df.negative_mentions / max(neg_total, 1)
    tr = trend.set_index("issue")
    df["trend_change"] = tr.change.reindex(df.index).replace([np.inf], 3.0).fillna(0).clip(-0.9, 3.0)
    df["trend_status"] = tr.status.reindex(df.index).fillna("Stable")
    norm = lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0
    df["score"] = 100 * (
        w["frequency"] * norm(df.share_of_negative) + w["trend"] * norm(df.trend_change)
        + w["severity"] * norm(df.severity) + w["churn"] * norm(df.churn_share)
    )
    df["evidence_level"] = pd.cut(df.negative_mentions, [-1, 19, 59, 10 ** 9], labels=["Low", "Medium", "High"]).astype(str)
    df["category"] = [L2_TO_L1[i] for i in df.index]
    df = df.sort_values("score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["why"] = [explain_rank(r) for r in df.reset_index().itertuples()]
    return df.reset_index().rename(columns={"index": "issue"})


def explain_rank(r) -> str:
    parts = [f"{r.negative_mentions} negative mentions ({r.share_of_negative:.0%} of all negative feedback)"]
    if r.trend_status in ("Rising", "New"):
        parts.append("more than tripled in recent months" if r.trend_change >= 2.99 else f"rising {r.trend_change:+.0%} in recent months")
    elif r.trend_status == "Declining":
        parts.append(f"declining {abs(r.trend_change):.0%} in recent months")
    if r.severity >= 2.4:
        parts.append(f"high severity ({r.severity:.1f} of 3)")
    if r.churn_share >= 0.2:
        parts.append(f"{r.churn_share:.0%} of these customers say they will leave")
    if r.positive_mentions >= 0.3 * max(r.total_mentions, 1):
        parts.append(f"but {r.positive_mentions} reviews praise this area (contradicting evidence)")
    if r.evidence_level == "Low":
        parts.append("LOW evidence: treat as a signal, not a conclusion")
    return "; ".join(parts) + "."


def release_impact(b: pd.DataFrame, t: pd.DataFrame, release_date, window_days: int = 28) -> pd.DataFrame:
    """Compare each issue's share of feedback before and after a release date."""
    rd = pd.Timestamp(release_date)
    before = b[(b.date >= rd - pd.Timedelta(days=window_days)) & (b.date < rd)]
    after = b[(b.date >= rd) & (b.date < rd + pd.Timedelta(days=window_days))]
    tb, ta = t[t.record_id.isin(before.record_id) & t.is_negative], t[t.record_id.isin(after.record_id) & t.is_negative]
    rows = []
    for issue in sorted(set(tb.issue) | set(ta.issue)):
        xb, xa = int((tb.issue == issue).sum()), int((ta.issue == issue).sum())
        rb, ra = xb / max(len(before), 1), xa / max(len(after), 1)
        rows.append({"issue": issue, "before_count": xb, "after_count": xa, "before_rate": rb, "after_rate": ra,
                     "change_points": ra - rb, "p_value": two_proportion_p(xb, len(before), xa, len(after))})
    out = pd.DataFrame(rows)
    if len(out):
        out["signal"] = np.where((out.p_value < 0.01) & (out.after_count >= 10),
                                 np.where(out.change_points > 0, "Worse after release", "Better after release"), "No clear change")
        out = out.sort_values("change_points", ascending=False)
    out.attrs.update({"before_n": len(before), "after_n": len(after),
                      "before_negative_share": float(before.is_negative.mean()) if len(before) else None,
                      "after_negative_share": float(after.is_negative.mean()) if len(after) else None})
    return out


def sub_patterns(t: pd.DataFrame, issue: str, max_clusters: int = 4) -> list[dict]:
    """Group the negative reviews of one issue into recurring patterns (offline, no model).
    The Gemini insight step later names these more precisely."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    sub = t[(t.issue == issue) & t.is_negative].drop_duplicates("record_id")
    if len(sub) < 12:
        return [{"pattern": "Too few reviews to split into patterns", "count": int(len(sub)),
                 "record_ids": sub.record_id.tolist()[:5], "example": sub.text.iloc[0] if len(sub) else ""}]
    stop = ["zomato", "zomatocare", "zomatofail", "fooddelivery", "productmanagement", "order", "fix", "asap", "disappointed", "worst",
            "experience", "honestly", "fed", "unbelievable", "sorry", "happy", "time", "month", "pls", "poor", "really", "comment",
            "removed", "phone", "email", "everyone", "happening", "just", "rant", "reply", "ever", "updating", "review", "teh", "star"]
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS.union(stop)), ngram_range=(1, 2), min_df=3, max_df=0.6,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    try:
        X = vec.fit_transform(sub.text.str.lower().map(lambda s: re.sub(r"\d+", " ", s)))
    except ValueError:
        return []
    k = int(min(max_clusters, max(2, len(sub) // 25)))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    terms = np.array(vec.get_feature_names_out())
    out = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        top = terms[np.argsort(km.cluster_centers_[c])[::-1][:4]]
        dense = X[idx].toarray()
        dist = ((dense - km.cluster_centers_[c]) ** 2).sum(axis=1)
        rep = idx[np.argsort(dist)[:3]]
        out.append({"pattern": ", ".join(top), "count": int(len(idx)), "record_ids": sub.record_id.iloc[rep].tolist(),
                    "example": sub.text.iloc[rep[0]]})
    return sorted(out, key=lambda d: -d["count"])


def evidence_quotes(t: pd.DataFrame, issue: str, sentiment: str = "Negative", n: int = 8) -> pd.DataFrame:
    """Pick representative quotes: verified-confidence first, spread across platforms."""
    sub = t[(t.issue == issue) & (t.sentiment == sentiment)].drop_duplicates("record_id")
    sub = sub.sort_values(["confidence", "likes"], ascending=False)
    picked = sub.groupby("platform", group_keys=False).head(2).head(n)
    if len(picked) < n:
        picked = pd.concat([picked, sub[~sub.record_id.isin(picked.record_id)].head(n - len(picked))])
    return picked[["record_id", "date", "platform", "language", "text"]]


def reference_findings(enriched: pd.DataFrame) -> dict:
    """The headline findings of the reference project, recomputed on our labels."""
    b = base_table(enriched)
    m = monthly_overview(b)
    neg = b[b.is_negative]
    support_share = neg.l1_categories.map(lambda v: CUSTOMER_SUPPORT in v).mean()
    delivery_share = neg.l1_categories.map(lambda v: "Fulfillment & Logistics" in v).mean()
    gold = b[b.mentions_gold]
    return {
        "negative_share_first_month": round(float(m.negative_share.iloc[0]), 3),
        "negative_share_last_month": round(float(m.negative_share.iloc[-1]), 3),
        "first_month": m.month.iloc[0], "last_month": m.month.iloc[-1],
        "share_of_negative_about_customer_support": round(float(support_share), 3),
        "share_of_negative_about_delivery": round(float(delivery_share), 3),
        "gold_mentions": int(len(gold)),
        "gold_negative_share": round(float(gold.is_negative.mean()), 3) if len(gold) else None,
        "top_gold_issue": (issue_long(gold)[lambda x: x.is_negative].issue.value_counts().index[0] if len(gold) else None),
    }


def compute_all(enriched: pd.DataFrame) -> dict:
    b = base_table(enriched)
    t = issue_long(b)
    rates = monthly_issue_rates(b, t)
    trend = emerging_and_declining(rates)
    return {
        "base": b, "tags": t, "monthly": monthly_overview(b), "rates": rates,
        "spike_alerts": spike_alerts(rates), "trend": trend,
        "opportunities": opportunity_table(b, t, trend),
    }
