"""Voice of Customer Copilot: decision support for product managers.

    streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics, chat, config, insights
from src import ui
from src.ui import SENT_COLORS, SERIES

st.set_page_config(page_title="Voice of Customer Copilot", page_icon="📣", layout="wide", initial_sidebar_state="auto")
ui.inject()


# ------------------------------------------------------------------ data
@st.cache_data
def load_all():
    out = {"enriched": analytics.load_enriched()}
    for name, path in {"cleaning": config.PROCESSED / "cleaning_report.json",
                       "evaluation": config.REPORTS / "evaluation_results.json"}.items():
        out[name] = json.loads(path.read_text()) if path.exists() else None
    ip = config.PROCESSED / "insights.json"
    out["insights"] = json.loads(ip.read_text()) if ip.exists() else []
    rel = config.CONTEXT_DIR / "releases.csv"
    out["releases"] = pd.read_csv(rel, parse_dates=["release_date"]) if rel.exists() else pd.DataFrame()
    return out


@st.cache_data
def tables_for(start, end, platforms: tuple, languages: tuple, weights: tuple):
    df = analytics.load_enriched()
    d = pd.to_datetime(df.date)
    keep = df.platform.isin(platforms) & df.language.isin(languages) & (d.isna() | ((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))))
    t = analytics.compute_all(df[keep])
    t["opportunities"] = analytics.opportunity_table(t["base"], t["tags"], t["trend"], dict(zip(["frequency", "trend", "severity", "churn"], weights)))
    return t


if not (config.PROCESSED / "enriched.parquet").exists():
    st.error("No results yet. Run  python run_pipeline.py  first.")
    st.stop()
DATA = load_all()
ENR = DATA["enriched"]
ENGINE = str(ENR.engine.iloc[0]) if "engine" in ENR else "unknown"

# ------------------------------------------------------------------ sidebar
PAGES = {
    "Overview": "Overview",
    "Ask": "Ask the copilot",
    "Hub": "Feedback Hub",
    "Pain": "Pain points",
    "Investigate": "Investigation",
    "Trends": "Trends and releases",
    "Decide": "Decisions",
    "Eval": "Evaluation",
}
REP = DATA["cleaning"] or {}
N_RAW = int(REP.get("raw_records", len(ENR)))
N_KEPT = int((ENR.is_relevant & ~ENR.is_duplicate).sum())
N_SOURCES, N_LANGS = int(ENR.platform.nunique()), int(ENR.language.nunique())
PERIOD = f"{config.WINDOW_START:%B %Y} to {config.WINDOW_END:%B %Y}"
with st.sidebar:
    labels_txt = "Labelled by Gemini" if ENGINE == "gemini" else "Labelled by keyword rules (Gemini not run yet)"
    ask_txt = "Gemini questions on" if config.llm_available() else "Gemini questions off (no key)"
    st.markdown(
        "<div class='side-name'>Voice of Customer Copilot</div><div class='side-sub'>Zomato customer feedback</div>"
        f"<div class='side-status'><span class='dot' aria-hidden='true' style='background:{ui.GOOD if ENGINE == 'gemini' else ui.WARN}'></span>{labels_txt}<br>"
        f"<span class='dot' aria-hidden='true' style='background:{ui.GOOD if config.llm_available() else ui.NEU}'></span>{ask_txt}</div>",
        unsafe_allow_html=True)
    st.divider()
    # The open page lives in the address bar (?page=Trends), so a shared link opens the same page.
    _start = st.query_params.get("page", "Overview")
    PAGE = st.radio("Navigate", list(PAGES), index=list(PAGES).index(_start) if _start in PAGES else 0,
                    format_func=PAGES.get, label_visibility="collapsed")
    if st.query_params.get("page") != PAGE:
        st.query_params["page"] = PAGE
    st.divider()
    st.markdown("<div class='side-head'>Filters</div>", unsafe_allow_html=True)
    dr = st.date_input("Date range", (config.WINDOW_START, config.WINDOW_END), min_value=config.WINDOW_START, max_value=config.WINDOW_END)
    start, end = (dr if isinstance(dr, tuple) and len(dr) == 2 else (config.WINDOW_START, config.WINDOW_END))
    all_platforms = sorted(ENR.platform.unique())
    platforms = st.multiselect("Platforms", all_platforms, default=all_platforms)
    all_langs = sorted(ENR.language.unique())
    languages = st.multiselect("Languages", all_langs, default=all_langs)
    with st.expander("Priority weights"):
        st.caption("How the problem ranking is weighted.")
        wf = st.slider("Frequency", 0.0, 1.0, 0.40, 0.05)
        wt = st.slider("Trend", 0.0, 1.0, 0.25, 0.05)
        ws = st.slider("Severity", 0.0, 1.0, 0.20, 0.05)
        wc = st.slider("Churn risk", 0.0, 1.0, 0.15, 0.05)
    st.divider()
    st.markdown(f"<div class='side-head'>Dataset</div><div class='side-data'>{N_RAW:,} feedback records from {N_SOURCES} sources in "
                f"{N_LANGS} languages, {PERIOD}. {N_KEPT:,} remain after cleaning. Synthetic practice data, not real customers.</div>",
                unsafe_allow_html=True)

T = tables_for(start, end, tuple(platforms or all_platforms), tuple(languages or all_langs), (wf, wt, ws, wc))
B, TAGS = T["base"], T["tags"]


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def show_quotes(frame, sentiment_col: str | None = None):
    html = "".join(ui.quote(r.text, r.record_id, r.platform, r.date, getattr(r, "sentiment", "") if sentiment_col else "")
                   for r in frame.itertuples())
    st.markdown(html or "<div class='section-note'>None.</div>", unsafe_allow_html=True)


def records_by_id(ids):
    return B[B.record_id.isin(ids)].set_index("record_id").reindex(list(ids)).dropna(subset=["text"]).reset_index()


def chart(fig, height=360, title=None):
    st.plotly_chart(ui.style_fig(fig, height, title), width="stretch", theme=None, config={"displayModeBar": False})


# ================================================================== OVERVIEW
if PAGE == "Overview":
    ui.page_header("What should the product team look at this week?",
                   "A weekly briefing built from customer feedback. Every number is counted by code from individual "
                   "review labels, with duplicates and spam left out.")
    ui.panel(f"""<div class='about'>
<div><h4>What this tool does</h4>
<p>It reads raw customer feedback about Zomato and turns it into a ranked list of product problems, each backed by the reviews behind it.</p>
<ol>
<li>Cleans the export: removes spam, duplicates, posts about other brands, phone numbers and email addresses.</li>
<li>Labels every review with its sentiment, the specific issue, severity and whether the customer says they are leaving.</li>
<li>Counts the labels in code to find the biggest problems, what is rising, and sudden spikes.</li>
<li>Suggests causes to investigate and lets you record a decision. The product manager decides, not the tool.</li>
</ol></div>
<div><h4>Dataset in use</h4>
<table class='facts'>
<tr><td>Records</td><td>{N_RAW:,} raw, {N_KEPT:,} after cleaning</td></tr>
<tr><td>Sources</td><td>{N_SOURCES}: app stores, X, Reddit, YouTube, Quora, complaint forums, LinkedIn, news comments</td></tr>
<tr><td>Languages</td><td>{N_LANGS}, including Hinglish, Hindi, Marathi, Tamil, Telugu, Kannada and Bengali</td></tr>
<tr><td>Period</td><td>{PERIOD}</td></tr>
<tr><td>Nature</td><td>Synthetic practice data. No real customers or real Zomato events.</td></tr>
<tr><td>Labels</td><td>{'Gemini' if ENGINE == 'gemini' else 'Keyword rules for now; Gemini labels replace them after the first run'}</td></tr>
</table></div></div>""", pad=True)
    m = T["monthly"]
    top = T["opportunities"].iloc[0]
    delta = (m.negative_share.iloc[-1] - m.negative_share.iloc[0]) * 100 if len(m) >= 2 else 0
    recent = [a for a in T["spike_alerts"] if len(m) >= 3 and a["month"] >= m.month.iloc[-3]]
    ui.kpi_row([
        ui.kpi("Feedback in view", f"{len(B):,}", f"of {N_KEPT:,} cleaned records, after filters"),
        ui.kpi("Negative share, latest month", f"{m.negative_share.iloc[-1]:.0%}" if len(m) else "–", f" since {m.month.iloc[0]}" if len(m) else "",
               ui.badge(f"{delta:+.1f} pts", "red" if delta > 0 else "green")),
        ui.kpi("Top-priority problem", top.issue, f"{top.negative_mentions} negative mentions", small=True),
        ui.kpi("Spike alerts, last 3 months", len(recent), "significance-tested"),
    ])
    left, right = st.columns([3, 2], gap="medium")
    with left:
        fig = go.Figure(go.Scatter(x=m.month, y=m.negative_share, mode="lines+markers", line=dict(color=ui.NEG, width=2),
                                   marker=dict(size=7), fill="tozeroy", fillcolor="rgba(196,96,79,0.06)",
                                   hovertemplate="%{x}<br>%{y:.1%} negative<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, max(0.9, float(m.negative_share.max()) + 0.05)])
        chart(fig, 330, "Share of feedback that is negative")
    with right:
        ui.section("Spike alerts", "Months where a problem jumped well above its usual share of feedback")
        alerts = sorted(T["spike_alerts"], key=lambda x: x["month"], reverse=True)[:5]
        ui.panel("".join(ui.alert(a["issue"], f"{pd.Timestamp(a['month']):%B %Y}: {a['count']} complaints, "
                                               f"{a['lift']:.1f} times its usual share of feedback") for a in alerts)
                 or "<div class='alert'><div class='alert-body'>No spikes in this range.</div></div>")
    ui.section("Top problems, ranked", "Each rank comes with its reason. Change the weights in the sidebar to challenge it.")
    o = T["opportunities"].head(6)
    mx = float(T["opportunities"].score.max())
    ui.panel("".join(ui.rank_card(r.rank, r.issue, r.why, r.score, mx, r.trend_status, r.evidence_level) for r in o.itertuples()))
    with st.expander("Does it reproduce the reference project's findings?"):
        ref = analytics.reference_findings(ENR)
        st.markdown(
            f"- Negative share moved from **{ref['negative_share_first_month']:.0%}** ({ref['first_month']}) to "
            f"**{ref['negative_share_last_month']:.0%}** ({ref['last_month']}). The reference project reported 59% to 67%.\n"
            f"- Customer support appears in **{ref['share_of_negative_about_customer_support']:.0%}** of negative reviews "
            f"(the reference reported 43.7% on real data; this synthetic dataset was not built to match it).\n"
            f"- Reviews mentioning Gold are **{ref['gold_negative_share']:.0%}** negative (reference: 63%). Top Gold issue: **{ref['top_gold_issue']}**.")


# ================================================================== ASK THE COPILOT (chat)
elif PAGE == "Ask":
    ui.page_header("Ask the copilot",
                   "Chat with Gemini about this feedback: rankings, trends, releases, platforms, languages, quotes and accuracy. "
                   "It answers from the numbers on this dashboard and the matching reviews, and uses the filters on the left.")
    if not config.llm_available():
        st.info("Gemini is off, so answers are search results only. Add GEMINI_API_KEY in the app's Secrets to switch it on.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    filters = {"dates": f"{start} to {end}",
               "platforms": "all" if set(platforms or all_platforms) == set(all_platforms) else list(platforms),
               "languages": "all" if set(languages or all_langs) == set(all_langs) else list(languages)}

    def render_answer(msg, key):
        st.markdown(msg["content"])
        res = msg.get("result") or {}
        stats, checks = res.get("stats", {}), res.get("checks", {})
        bits = []
        if str(res.get("source", "")).startswith("gemini"):
            bits.append(ui.badge("Written by Gemini", "ai"))
        if res.get("confidence"):
            bits.append(ui.badge("Confidence " + str(res["confidence"]).lower(), "blue"))
        if checks:
            ok = checks.get("verified")
            bits.append(ui.badge("Citations and numbers checked" if ok else "Check flagged something", "green" if ok else "red"))
        if stats:
            bits.append(f"{stats.get('matching_reviews', 0):,} matching reviews ({'; '.join(stats.get('how_matched', []))})")
        if bits:
            st.markdown(f"<div class='kpi-note'>{' '.join(bits)}</div>", unsafe_allow_html=True)
        if checks.get("numbers_not_in_data"):
            st.caption("These numbers were not found in the data and should not be trusted: " + ", ".join(checks["numbers_not_in_data"]))
        if checks.get("invalid_citations_removed"):
            st.caption("Removed citations to reviews that do not exist: " + ", ".join(checks["invalid_citations_removed"]))
        ids = res.get("cited_record_ids", [])[:8]
        if ids:
            with st.expander(f"Reviews cited ({len(ids)})"):
                show_quotes(records_by_id(ids), "sentiment")
        follow = res.get("follow_up_questions", [])[:3]
        if follow:
            cols = st.columns(len(follow))
            for i, (c, f) in enumerate(zip(cols, follow)):
                if c.button(f, key=f"fu-{key}-{i}", width="stretch"):
                    st.session_state.pending = f
                    st.rerun()

    typed = st.chat_input("Ask about the feedback, for example: what do Gold members complain about after renewal…")
    question = typed or st.session_state.pop("pending", None)
    AVATAR = {"user": ":material/person:", "assistant": ":material/forum:"}
    if not st.session_state.chat and not question:
        st.markdown("<div class='section-note'>Try one of these, or type your own question below.</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, q in enumerate(chat.STARTERS):
            if cols[i % 3].button(q, key=f"start-{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()
    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"], avatar=AVATAR[m["role"]]):
            render_answer(m, i) if m["role"] == "assistant" else st.markdown(m["content"])

    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATAR["user"]):
            st.markdown(question)
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            with st.spinner("Reading the numbers and the matching reviews…"):
                try:
                    res = chat.answer(question, history, T, DATA, filters)
                    msg = {"role": "assistant", "content": res["answer"], "result": res}
                except Exception as exc:  # never crash the page; say what went wrong
                    msg = {"role": "assistant", "result": {},
                           "content": (f"Gemini did not answer: {str(exc)[:240]}\n\nTry again in a minute. If it keeps failing, "
                                       "check that GEMINI_API_KEY in the app's Secrets is correct and that the model has quota left.")}
            render_answer(msg, len(st.session_state.chat))
        st.session_state.chat.append(msg)
    if st.session_state.chat:
        if st.session_state.get("confirm_clear"):
            st.markdown("<div class='section-note'>Clear all questions and answers on this page? This cannot be undone.</div>",
                        unsafe_allow_html=True)
            k1, k2, _ = st.columns([2, 2, 3])
            if k1.button("Yes, clear conversation", type="primary"):
                st.session_state.chat, st.session_state.confirm_clear = [], False
                st.rerun()
            if k2.button("Keep conversation"):
                st.session_state.confirm_clear = False
                st.rerun()
        elif st.button("Clear conversation"):
            st.session_state.confirm_clear = True
            st.rerun()

# ================================================================== FEEDBACK HUB
elif PAGE == "Hub":
    ui.page_header("What came in, and what cleaning changed",
                   "Every change is recorded on the row, so you can always see what the tool did to a review.")
    rep = DATA["cleaning"] or {}
    n_raw = rep.get("raw_records", len(ENR))
    n_noise = int((~ENR.is_relevant).sum())
    n_dup = int((ENR.is_relevant & ENR.is_duplicate).sum())
    skipped = rep.get("skipped_lines", {})
    ui.kpi_row([
        ui.kpi("Raw records", f"{n_raw:,}", f"{skipped.get('header', 0)} repeated headers and {skipped.get('blank', 0)} blank lines skipped"),
        ui.kpi("Noise removed", n_noise, "spam, brand replies, jobs, stocks, other brands"),
        ui.kpi("Duplicates merged", n_dup, "same user re-posting; same text from different users kept"),
        ui.kpi("Personal details removed", rep.get("personal_details_redacted", 0), "phone numbers and emails, before any model sees the text"),
    ])
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        stages = ["Raw records", "After removing noise", "After merging duplicates"]
        vals = [n_raw, n_raw - n_noise, n_raw - n_noise - n_dup]
        fig = go.Figure(go.Funnel(y=stages, x=vals, textinfo="value+percent initial", marker_color=["#9ec5f4", "#3987e5", "#1c5cab"],
                                  connector=dict(fillcolor="#eeebe3")))
        chart(fig, 300, "Cleaning funnel")
    with c2:
        dq = pd.Series(rep.get("date_quality", {})).rename_axis("How the date was read").reset_index(name="Records")
        dq["How the date was read"] = dq["How the date was read"].str.replace("_", " ").str.capitalize()
        ui.section("Date quality", "12 date formats in the raw file. Estimates are labelled as estimates.")
        st.dataframe(dq, hide_index=True, width="stretch")
    c1, c2 = st.columns(2, gap="medium")
    for col, field, title in ((c1, "platform", "Feedback by platform"), (c2, "language", "Feedback by language")):
        with col:
            v = B[field].value_counts().reset_index()
            fig = px.bar(v, x="count", y=field, orientation="h")
            fig.update_traces(marker_color=SERIES[0], hovertemplate="%{y}: %{x}<extra></extra>")
            fig.update_yaxes(categoryorder="total ascending", title=None).update_xaxes(title=None)
            chart(fig, 340, title)
    ui.section("Record explorer", "Search any word, in any language, and compare the raw row with the cleaned one.")
    q = st.text_input("Search reviews", placeholder="Search reviews, for example rain fee, refund, Gold or खाना…", label_visibility="collapsed")
    view = ENR if not q else ENR[ENR.text.str.contains(q, case=False, regex=False)]
    cols = ["record_id", "raw_platform", "platform", "raw_date", "date", "raw_rating", "rating", "raw_text", "text",
            "is_relevant", "noise_type", "is_duplicate", "duplicate_of", "language", "sentiment", "primary_issue", "label_source"]
    st.dataframe(view[cols].head(300), hide_index=True, width="stretch", height=380)

# ================================================================== PAIN POINTS
elif PAGE == "Pain":
    ui.page_header("What customers complain about",
                   "Counts are of negative reviews. Praise in the same area is shown alongside, so you see contradicting evidence too.")
    neg = TAGS[TAGS.is_negative]
    total_neg = int(B.is_negative.sum())
    order = [i for i in neg.issue.value_counts().index.tolist()][:12]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        cat = neg.drop_duplicates(["record_id", "category"]).category.value_counts().reset_index()
        cat["share"] = cat["count"] / max(total_neg, 1)
        fig = px.bar(cat, x="share", y="category", orientation="h", custom_data=["count"])
        fig.update_traces(marker_color=ui.NEG, hovertemplate="%{y}<br>%{x:.1%} of negative reviews (%{customdata[0]})<extra></extra>")
        fig.update_xaxes(tickformat=".0%", title=None).update_yaxes(categoryorder="total ascending", title=None)
        chart(fig, 400, "Share of negative reviews by category")
    with c2:
        iss = TAGS.groupby(["issue", "sentiment"]).size().reset_index(name="count")
        fig = px.bar(iss[iss.issue.isin(order)], x="count", y="issue", color="sentiment", orientation="h",
                     color_discrete_map=SENT_COLORS, category_orders={"issue": order[::-1], "sentiment": ["Negative", "Neutral", "Positive"]})
        fig.update_yaxes(title=None).update_xaxes(title=None)
        chart(fig, 400, "Top issues: complaints next to praise")
    heat = neg.groupby(["issue", "platform"]).size().unstack(fill_value=0)
    heat = heat.div(B.groupby("platform").size(), axis=1).reindex(order[:10]).fillna(0)
    fig = px.imshow(heat, color_continuous_scale=["#f5f8fe", "#b7d3f6", "#5598e7", "#184f95"], aspect="auto")
    fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1%} of this platform's feedback<extra></extra>")
    fig.update_coloraxes(colorbar_tickformat=".0%", colorbar_title="")
    fig.update_xaxes(title=None).update_yaxes(title=None)
    chart(fig, 430, "Where each problem shows up (share of each platform's feedback)")
    ui.section("Read the evidence")
    pick = st.selectbox("Issue", order, label_visibility="collapsed")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"**Complaints** {ui.badge(str(int((neg.issue == pick).sum())), 'red')}", unsafe_allow_html=True)
        show_quotes(analytics.evidence_quotes(TAGS, pick, "Negative", 6).assign(sentiment="Negative"), "sentiment")
    with c2:
        npos = int(((TAGS.issue == pick) & (TAGS.sentiment == "Positive")).sum())
        st.markdown(f"**Praise in the same area** {ui.badge(str(npos), 'blue')}", unsafe_allow_html=True)
        show_quotes(analytics.evidence_quotes(TAGS, pick, "Positive", 4).assign(sentiment="Positive"), "sentiment")

# ================================================================== INVESTIGATION
elif PAGE == "Investigate":
    ui.page_header("Investigate a problem",
                   "Causes are always labelled as hypotheses: reviews show experiences, not root causes. "
                   "Every claim links to the reviews behind it.")
    ins_list = DATA["insights"]
    if not ins_list:
        st.warning("No insights yet. Run  python run_pipeline.py")
        st.stop()
    names = [i["issue"] for i in ins_list]
    choice = st.selectbox("Pain point", names, format_func=lambda n: f"{names.index(n) + 1}. {n}")
    ins = ins_list[names.index(choice)]
    pack, v = ins["evidence_pack"], ins["verification"]
    st.markdown(f"<div class='headline'><div class='headline-text'>{ui.e(ins['headline'])}</div>"
                f"<div class='headline-meta'>"
                f"{ui.badge('Written by Gemini', 'ai') if str(ins['source']).startswith('gemini') else ui.badge('Written by template', 'grey')}"
                f"Numbers counted by code.</div></div>", unsafe_allow_html=True)
    if not (v["numbers_verified"] and v["citations_verified"]):
        st.warning("The checks flagged this insight: " + "; ".join(v["notes"]) + ". The numbers shown come from code and are unaffected.")
    contra = pack["positive_mentions"] / max(pack["positive_mentions"] + pack["negative_mentions"], 1)
    conf = pack["label_confidence_avg"]
    st.markdown("<div class='rel-grid'>" + "".join([
        ui.rel_tile("Reviews behind it", pack["negative_mentions"]),
        ui.rel_tile("Citation check", "Passed" if v["citations_verified"] else "Flagged", "ok" if v["citations_verified"] else "bad"),
        ui.rel_tile("Label confidence", f"{conf:.2f}" if conf is not None else "n/a", "ok" if (conf or 0) >= 0.75 else "mid"),
        ui.rel_tile("Platforms", len(pack["platforms"])),
        ui.rel_tile("Contradicting", f"{contra:.0%}", "mid" if contra >= 0.25 else ""),
        ui.rel_tile("Data sufficiency", pack["evidence_level"], {"High": "ok", "Medium": "mid", "Low": "bad"}[pack["evidence_level"]]),
    ]) + "</div>", unsafe_allow_html=True)
    if ENGINE != "gemini":
        st.caption("Label confidence is fixed at 0.50 for the rules baseline; Gemini reports its own.")
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        ui.section("Recurring patterns", "Groups of similar complaints. Open one to read the reviews.")
        for p in ins["patterns"]:
            with st.expander(f"{p['name']} ({p['count']} reviews)"):
                show_quotes(records_by_id(p["record_ids"]))
        ui.section("Hypotheses to investigate")
        for h in ins["hypotheses"]:
            with st.expander(h["statement"]):
                st.markdown(ui.badge(h.get("label", "Hypothesis"), "ai" if "Gemini" in h.get("label", "") else "amber"), unsafe_allow_html=True)
                if h.get("what_would_confirm_it"):
                    st.markdown(f"**What would confirm it:** {h['what_would_confirm_it']}")
                show_quotes(records_by_id(h["supporting_record_ids"]))
    with c2:
        monthly = pd.Series(pack["monthly_negative_mentions"]).rename_axis("month").reset_index(name="complaints")
        fig = px.bar(monthly, x="month", y="complaints")
        fig.update_traces(marker_color=ui.NEG, hovertemplate="%{x}: %{y}<extra></extra>")
        fig.update_xaxes(title=None).update_yaxes(title=None)
        chart(fig, 250, "Complaints per month")
        st.markdown(f"<div class='opportunity'><div class='opportunity-title'>Product opportunity</div>"
                    f"<div>{ui.e(ins['opportunity']['title'])}</div>"
                    f"<div class='kpi-note'>Success metric: {ui.e(ins['opportunity']['success_metric'])}</div></div>", unsafe_allow_html=True)
        ui.section("Data to check")
        st.markdown("\n".join(f"- {d}" for d in ins["data_to_check"]))
        ui.section("Customer interview questions")
        st.markdown("\n".join(f"- {q}" for q in ins["interview_questions"]))
        ui.section("Contradicting evidence")
        st.write(ins["contradicting_evidence"])
        st.markdown("".join(ui.quote(q["text"], q["record_id"], q["platform"], "", "Positive") for q in pack["contradicting_quotes"][:2]),
                    unsafe_allow_html=True)
    st.divider()
    ui.section("Ask the data", "Answers come only from matching reviews; counts come from code; every cited record is checked.")
    if not config.llm_available():
        st.info("Gemini questions are off, so this box only searches. Add GEMINI_API_KEY to Streamlit Secrets to turn them on.")
    with st.form("ask"):
        qtext = st.text_input("Question", placeholder="For example: what do Gold members complain about after renewal…", label_visibility="collapsed")
        asked = st.form_submit_button("Ask", type="primary")
    if asked and qtext:
        with st.spinner("Reading the feedback…"):
            try:
                res = insights.ask(qtext, B)
            except Exception as exc:
                res = {"answer": f"Could not answer: {exc}", "cited_record_ids": [], "verified": False, "confidence": "Low", "stats": {}}
        st.markdown(f"<div class='panel pad'>{ui.e(res['answer'])}<div class='kpi-note'>"
                    f"{ui.badge('Confidence ' + res['confidence'].lower(), 'blue')}"
                    f"{ui.badge('Citations verified' if res.get('verified') else 'Unverified citations removed', 'green' if res.get('verified') else 'red')}"
                    f"{res.get('stats', {}).get('matching_records', 0)} matching reviews</div></div>", unsafe_allow_html=True)
        show_quotes(records_by_id([c for c in res.get("cited_record_ids", []) if c not in res.get("invalid_citations", [])][:5]))

# ================================================================== TRENDS & RELEASES
elif PAGE == "Trends":
    ui.page_header("What is changing",
                   "Trends use each issue's share of all feedback that month, so a busy month never looks like a worse product.")
    rates, tr = T["rates"], T["trend"]
    default = tr[tr.status.isin(["Rising", "New"])].issue.tolist()[:3] or T["opportunities"].issue.tolist()[:3]
    pick = st.multiselect("Issues to plot", sorted(rates.issue.unique()), default=default[:4])
    if pick:
        fig = px.line(rates[rates.issue.isin(pick)], x="month", y="rate", color="issue", markers=True, color_discrete_sequence=SERIES)
        fig.update_traces(line_width=2.2, marker_size=7, hovertemplate="%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title=None).update_xaxes(title=None)
        for r in DATA["releases"].itertuples():
            x = r.release_date.strftime("%Y-%m")
            fig.add_vline(x=x, line_dash="dot", line_color="#b9b5aa")
            fig.add_annotation(x=x, y=1.02, yref="paper", text="▼", showarrow=False, font=dict(size=10, color="#8a877d"),
                               hovertext=f"{r.release_date:%d %b %Y}: {r.name}")
        chart(fig, 420, "Complaints as a share of all feedback   (▼ marks a product release; hover for its name)")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.section("Rising and declining", f"{', '.join(tr.attrs.get('recent', []))} compared with {', '.join(tr.attrs.get('earlier', []))}")
        show = tr[tr.status != "Stable"][["issue", "status", "earlier_rate", "recent_rate", "recent_count"]].rename(columns={
            "issue": "Issue", "status": "Status", "earlier_rate": "Earlier share", "recent_rate": "Recent share", "recent_count": "Recent complaints"})
        st.dataframe(show.style.format({"Earlier share": "{:.1%}", "Recent share": "{:.1%}"}), hide_index=True, width="stretch")
    with c2:
        ui.section("Spike alerts", "Lift of at least 1.6×, at least 12 complaints, significant at p ≤ 0.01")
        sp = pd.DataFrame(T["spike_alerts"])
        if len(sp):
            sp = sp[["month", "issue", "count", "rate", "baseline_rate", "lift"]].rename(columns={
                "month": "Month", "issue": "Issue", "count": "Complaints", "rate": "Share", "baseline_rate": "Usual share", "lift": "Lift"})
            st.dataframe(sp.style.format({"Share": "{:.1%}", "Usual share": "{:.1%}", "Lift": "{:.1f}×"}), hide_index=True, width="stretch")
    ui.section("Before and after a release", "Pick a release to compare each issue's share of feedback in the weeks before and after it.")
    rel = DATA["releases"]
    if rel.empty:
        st.info("Add product releases to data/context/releases.csv")
    else:
        labels = [f"{r.release_date:%d %b %Y}: {r.name}" for r in rel.itertuples()]
        c1, c2 = st.columns([3, 1])
        idx = c1.selectbox("Release", range(len(labels)), format_func=lambda i: labels[i])
        window = c2.slider("Days each side", 14, 60, 28, 7)
        imp = analytics.release_impact(B, TAGS, rel.release_date.iloc[idx], window)
        a = imp.attrs
        bn, an = a["before_negative_share"] or 0, a["after_negative_share"] or 0
        worse = imp[imp.signal == "Worse after release"] if len(imp) else imp
        ui.kpi_row([
            ui.kpi("Reviews before / after", f"{a['before_n']} / {a['after_n']}"),
            ui.kpi("Negative share", f"{bn:.0%} → {an:.0%}", badge_html=ui.badge(f"{(an - bn) * 100:+.1f} pts", "red" if an > bn else "green")),
            ui.kpi("Issues clearly worse after", len(worse), ", ".join(worse.issue.head(2)) if len(worse) else "no clear change", small=False),
        ])
        if len(imp):
            st.dataframe(imp.head(10).rename(columns={"issue": "Issue", "before_count": "Before", "after_count": "After", "before_rate": "Share before",
                                                      "after_rate": "Share after", "change_points": "Change", "p_value": "p-value", "signal": "Signal"})
                         .style.format({"Share before": "{:.1%}", "Share after": "{:.1%}", "Change": "{:+.1%}", "p-value": "{:.3f}"}),
                         hide_index=True, width="stretch")
        st.caption(f"Release note: {rel.notes.iloc[idx]}")
        st.warning("A change after a release is a reason to investigate, not proof the release caused it. Seasonality, traffic mix "
                   "and other changes at the same time move feedback too. When one problem surges, other problems' shares fall "
                   "even if nothing improved.")

# ================================================================== DECISION ENGINE
elif PAGE == "Decide":
    ui.page_header("Decide what to work on",
                   "Observed data is computed and locked. Impact, confidence and effort are your assumptions, and stay editable.")
    o = T["opportunities"].copy()
    log_path = config.DECISIONS_DIR / "decision_log.csv"
    saved = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["issue"])
    months = max(B.month.nunique(), 1)
    o["monthly_reach"] = (o.negative_mentions / months).round(1)
    ins_by_issue = {i["issue"]: i for i in DATA["insights"]}
    o["suggested_opportunity"] = o.issue.map(lambda i: ins_by_issue.get(i, {}).get("opportunity", {}).get("title", ""))
    defaults = {"impact_1_to_3": 2, "confidence_pct": None, "effort_person_weeks": 4.0, "decision": "Pending", "note": "", "shipped_date": ""}
    for col, val in defaults.items():
        o[col] = o.issue.map(saved.set_index("issue")[col]) if col in saved else val
        if val is not None:
            o[col] = o[col].fillna(val)
    o["confidence_pct"] = o.confidence_pct.fillna(o.evidence_level.map({"High": 80, "Medium": 60, "Low": 40}))
    o["shipped_date"] = o.shipped_date.astype(str).replace("nan", "")
    o["note"] = o.note.astype(str).replace("nan", "")
    edit_cols = ["rank", "issue", "score", "negative_mentions", "trend_status", "evidence_level", "suggested_opportunity",
                 "impact_1_to_3", "confidence_pct", "effort_person_weeks", "decision", "note", "shipped_date"]
    edited = st.data_editor(
        o[edit_cols], hide_index=True, width="stretch", height=430,
        disabled=["rank", "issue", "score", "negative_mentions", "trend_status", "evidence_level"],
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "issue": st.column_config.TextColumn("Problem"),
            "score": st.column_config.ProgressColumn("Data score", min_value=0, max_value=100, format="%.0f"),
            "negative_mentions": st.column_config.NumberColumn("Complaints"),
            "trend_status": st.column_config.TextColumn("Trend"),
            "evidence_level": st.column_config.TextColumn("Evidence"),
            "suggested_opportunity": st.column_config.TextColumn("Opportunity (editable)", width="large"),
            "impact_1_to_3": st.column_config.NumberColumn("Impact 1-3", min_value=1, max_value=3, step=1),
            "confidence_pct": st.column_config.NumberColumn("Confidence %", min_value=0, max_value=100, step=5),
            "effort_person_weeks": st.column_config.NumberColumn("Effort (weeks)", min_value=0.5, step=0.5),
            "decision": st.column_config.SelectboxColumn("Decision", options=["Pending", "Approve", "Modify", "Reject"]),
            "note": st.column_config.TextColumn("Your note"),
            "shipped_date": st.column_config.TextColumn("Shipped (YYYY-MM-DD)"),
        })
    edited = edited.merge(o[["issue", "monthly_reach", "why"]], on="issue")
    edited["priority"] = (edited.monthly_reach * edited.impact_1_to_3 * edited.confidence_pct / 100 / edited.effort_person_weeks).round(1)
    c1, c2, c3 = st.columns([1.1, 1.4, 3])
    if c1.button("Save decisions", type="primary"):
        edited.drop(columns=["monthly_reach", "why"]).to_csv(log_path, index=False)
        st.toast("Decisions saved to the decision log")
    approved = edited[edited.decision.isin(["Approve", "Modify"])]
    export = pd.DataFrame({
        "Summary": approved.suggested_opportunity.where(approved.suggested_opportunity != "", approved.issue),
        "Description": [f"Customer problem: {r.issue}. {r.negative_mentions} negative reviews; trend {r.trend_status}. Evidence: {r.why} "
                        f"Product manager note: {r.note}" for r in approved.itertuples()],
        "Labels": "voice-of-customer", "Priority score": approved.priority})
    c2.download_button("Export approved items", export.to_csv(index=False), "roadmap_items.csv", disabled=approved.empty,
                       help="CSV that Jira and Notion can import")
    c3.caption("Priority = monthly reach × impact × confidence ÷ effort. Reach is observed; the other three are yours. "
               "On Streamlit Cloud, saved decisions reset when the app restarts; export them to keep them.")
    top = edited.sort_values("priority", ascending=False).head(12)
    fig = px.bar(top, x="priority", y="issue", color="decision", orientation="h", custom_data=["monthly_reach", "effort_person_weeks"],
                 color_discrete_map={"Pending": ui.NEU, "Approve": ui.GOOD, "Modify": ui.WARN, "Reject": ui.NEG},
                 category_orders={"issue": top.issue.tolist()})
    fig.update_traces(hovertemplate="%{y}<br>Priority %{x}<br>%{customdata[0]} complaints a month, %{customdata[1]} weeks of effort<extra></extra>")
    fig.update_xaxes(title=None).update_yaxes(title=None)
    chart(fig, 420, "Priority with your assumptions applied")
    ui.section("Closed loop: did the problem fall after we shipped?",
               "Enter a shipped date for an approved item. The tool compares that problem's share of feedback before and after.")
    shipped = edited[edited.shipped_date.astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
    if shipped.empty:
        st.markdown("<div class='section-note'>Nothing shipped yet.</div>", unsafe_allow_html=True)
    for r in shipped.itertuples():
        imp = analytics.release_impact(B, TAGS, r.shipped_date, 28)
        row = imp[imp.issue == r.issue] if len(imp) else imp
        if len(row):
            x = row.iloc[0]
            ui.panel(ui.alert(f"{r.issue}, shipped {r.shipped_date}",
                                 f"{x.before_rate:.1%} of feedback before → {x.after_rate:.1%} after ({x.signal}). "
                                 "Association only: check review volume and user mix before claiming the fix worked."))

# ================================================================== EVALUATION
elif PAGE == "Eval":
    ui.page_header("How accurate the tool is",
                   "Scored against a hidden answer key the model never sees. Headline numbers use a 70% holdout split "
                   "that was not used to write the baseline rules.")
    ev = DATA["evaluation"]
    if not ev:
        st.warning("Run  python -m src.evaluate  (needs the answer key in data/eval/).")
        st.stop()
    engines = ev["engines"]
    active = ev.get("active_engine", ENGINE)
    r = engines.get(active, next(iter(engines.values())))
    h = r["classification_holdout"]
    inc = sum(x["detected"] for x in ev["incident_detection"])
    ui.kpi_row([
        ui.kpi("Sentiment accuracy", f"{h['sentiment_accuracy']:.1%}", f"on held-back data, {active}", ui.badge("target 85%", "green" if h["sentiment_accuracy"] >= .85 else "amber")),
        ui.kpi("Main issue correct", f"{h['primary_issue_correct']:.1%}", f"on held-back data, {active}", ui.badge("target 80%", "green" if h["primary_issue_correct"] >= .8 else "amber")),
        ui.kpi("Planted incidents caught", f"{inc} of 4", "in the right month"),
        ui.kpi("Dashboard error", f"{ev['dashboard_numbers_vs_truth']['monthly_negative_share_mean_abs_error_points']} pts", "monthly negative share vs truth"),
    ])
    rows = []
    for name, rr in engines.items():
        hh, c = rr["classification_holdout"], rr["cleaning"]
        rows.append({"Engine": name, "Sentiment accuracy": hh["sentiment_accuracy"], "Sentiment macro F1": hh["sentiment_macro_f1"],
                     "Main issue correct": hh["primary_issue_correct"], "Issue F1": hh["issue_micro"]["f1"],
                     "Feature request F1": hh["feature_request"]["f1"], "Sarcasm": rr["sarcasm_sentiment_accuracy"],
                     "Noise F1": c["noise_removal"]["f1"], "Duplicate recall": c["duplicates"]["recall"]})
    ui.section("Engines compared")
    st.dataframe(pd.DataFrame(rows).style.format({k: "{:.1%}" for k in rows[0] if k != "Engine"}), hide_index=True, width="stretch")
    if "gemini" not in engines:
        st.info("Only the rule-based baseline has been scored so far. Run the pipeline with your Gemini key "
                "(GitHub → Actions → Run full pipeline with Gemini) and a Gemini row appears here.")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        pi = pd.DataFrame(r["per_issue"]).sort_values("f1")
        fig = px.bar(pi, x="f1", y="issue", orientation="h", custom_data=["precision", "recall", "support"])
        fig.update_traces(marker_color=[ui.NEG if f < 0.6 else ui.WARN if f < 0.8 else SERIES[0] for f in pi.f1],
                          hovertemplate="%{y}<br>F1 %{x:.2f}, precision %{customdata[0]:.2f}, recall %{customdata[1]:.2f}"
                                        "<br>%{customdata[2]} true mentions<extra></extra>")
        fig.update_yaxes(title=None).update_xaxes(title=None, range=[0, 1])
        chart(fig, 640, f"F1 by issue ({active})")
    with c2:
        dn = ev["dashboard_numbers_vs_truth"]
        mm = pd.DataFrame(dn["monthly"]).melt("month", var_name="series", value_name="share")
        mm["series"] = mm.series.map({"negative_share_copilot": "Copilot", "negative_share_truth": "Answer key"})
        fig = px.line(mm, x="month", y="share", color="series", markers=True, color_discrete_sequence=[SERIES[0], SERIES[1]])
        fig.update_yaxes(tickformat=".0%", title=None).update_xaxes(title=None)
        chart(fig, 290, "Do the dashboard numbers match the truth?")
        st.markdown(f"- Issue shares are off by **{dn['issue_share_mean_abs_error_points']} points** on average "
                    f"(worst: {dn['issue_with_largest_error']}, {dn['issue_share_max_abs_error_points']}).\n"
                    f"- Top-5 problems: **{dn['top5_overlap']} of 5** match the answer key's top 5.")
        lang = pd.DataFrame({"Language": list(r["by_language"]), "Sentiment": list(r["by_language"].values()),
                             "Main issue": [r["by_language_issue"][k] for k in r["by_language"]]})
        st.dataframe(lang.style.format({"Sentiment": "{:.0%}", "Main issue": "{:.0%}"}), hide_index=True, width="stretch")
    ui.section("Planted incident detection")
    st.dataframe(pd.DataFrame(ev["incident_detection"]).rename(columns=lambda c: c.replace("_", " ").capitalize()), hide_index=True, width="stretch")
    ui.section("Answer key audit", "An evaluation is only as good as its answer key.")
    au = ev["answer_key_audit"]
    st.markdown(f"<div class='panel pad'><b>{au['suspect_rows']} rows</b> across <b>{au['distinct_texts']} repeated sentences</b> have "
                f"contradictory sentiment in the answer key itself ({', '.join(f'{k}: {v}' for k, v in au['by_language'].items())}). "
                f"Identical sentences are labelled Positive in some copies and Negative in others. Holdout sentiment accuracy "
                f"without these rows: <b>{r['classification_holdout_excluding_suspect_key_rows']['sentiment_accuracy']:.1%}</b>.</div>",
                unsafe_allow_html=True)
    sheet = config.REPORTS / "evaluation_sheet.xlsx"
    if sheet.exists():
        st.download_button("Download the evaluation sheet (Excel)", sheet.read_bytes(), "evaluation_sheet.xlsx")
    with st.expander("Known failure modes and guardrails"):
        st.markdown("""
| Failure mode | Guardrail |
|---|---|
| Model invents counts or percentages | Model never counts; code computes every number; headline numbers checked against the evidence |
| Model invents a quote | Evidence quotes must appear word for word in the review, otherwise removed |
| Model cites a review that does not exist | Every cited record checked; invalid ones removed and the insight flagged |
| Gemini down, rate-limited, or broken output | Retries, smaller batches, then the rules fallback, marked on each row |
| Instructions hidden inside a review | Prompts treat review text as untrusted data |
| Sarcasm read as praise | Prompt rule plus a sarcasm score in this centre |
| Star rating contradicts the text | Sentiment comes from text; conflicts flagged on the row |
| A busy month looks like a worse product | Trends use share of feedback; spikes need a significance test |
| Correlation read as causation after a release | Release analysis labelled as association, with a caveat |
| Personal data sent to the model | Phone numbers and emails removed before labelling |
""")
