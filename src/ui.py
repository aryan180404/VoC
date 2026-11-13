"""Visual building blocks for the dashboard: one stylesheet and small HTML helpers.

Design notes (kept here so the next change stays consistent):
  * It should look like a working tool a product team uses every day, not a landing page.
  * One typeface (IBM Plex Sans), modest sizes, sentence case everywhere.
  * Soft contrast: slate ink on a cool light grey, never black on white.
  * Colour is reserved for data and status. The interface itself is grey plus one teal.
  * Related numbers share one panel with dividers instead of a row of separate cards.
"""
from __future__ import annotations

import html

import streamlit as st

INK = "#2b3440"       # slate, not black
INK_2 = "#56616e"
INK_3 = "#6b7683"
LINE = "#e2e6eb"
SURFACE = "#ffffff"
CANVAS = "#f3f5f7"
ACCENT = "#2f6f73"    # muted teal: links, selection, score bars
ACCENT_SOFT = "#e6f0f0"
NEG = "#c4604f"       # soft brick
POS = "#3f78b0"       # soft blue
NEU = "#b9c0c8"
GOOD = "#3f8a68"
WARN = "#b5852a"

SENT_COLORS = {"Negative": NEG, "Neutral": NEU, "Positive": POS}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, .stMarkdown, p, label, li, td, th, h1, h2, h3, h4, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {CANVAS}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stDeployButton"], .stDeployButton, [data-testid="stToolbar"] {{ display: none !important; }}
.block-container {{ padding-top: 2rem; padding-bottom: 4rem; max-width: 1240px; }}
.stMarkdown p, .stMarkdown li {{ color: {INK}; }}

/* Sidebar: light, quiet */
section[data-testid="stSidebar"] {{ background: #fafbfc; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{ padding: 7px 10px; border-radius: 6px; width: 100%; cursor: pointer; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label > div > div:first-child:not([data-testid]) {{ display: none; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label p {{ font-size: 0.92rem; color: {INK_2}; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{ background: #eef1f4; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {{ background: {ACCENT_SOFT}; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p {{ color: {ACCENT}; font-weight: 600; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:focus-visible) {{ box-shadow: inset 0 0 0 1px #9fc0c2; }}
[data-tag] {{ background: {ACCENT_SOFT} !important; color: {INK} !important; border-radius: 4px !important; }}
[data-tag] span, [data-tag] button {{ color: {INK} !important; }}
.side-name {{ font-weight: 600; font-size: 1rem; color: {INK}; line-height: 1.25; }}
.side-sub {{ font-size: 0.8rem; color: {INK_3}; margin-top: 2px; }}
.side-status {{ font-size: 0.8rem; color: {INK_2}; margin-top: 10px; line-height: 1.6; }}
.dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px; vertical-align:middle; }}
.side-head {{ font-size: 0.8rem; font-weight: 600; color: {INK_2}; margin: 4px 0 6px 0; }}
.side-data {{ font-size: 0.78rem; color: {INK_3}; line-height: 1.55; }}

/* Page header */
.page-title {{ font-size: 1.55rem; font-weight: 600; color: {INK}; line-height: 1.25; margin: 0 0 6px 0; letter-spacing: -0.005em; }}
.page-sub {{ color: {INK_2}; font-size: 0.95rem; line-height: 1.55; max-width: 68ch; margin-bottom: 20px; }}
.section-title {{ font-size: 1rem; font-weight: 600; color: {INK}; margin: 26px 0 6px 0; }}
.section-note {{ font-size: 0.85rem; color: {INK_3}; margin: 0 0 10px 0; max-width: 72ch; }}

/* Panels */
.panel {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; margin-bottom: 16px; }}
.panel.pad {{ padding: 16px 18px; }}
.stats {{ display: grid; background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; margin-bottom: 16px; }}
.kpi {{ padding: 14px 18px; border-left: 1px solid {LINE}; min-width: 0; }}
.kpi:first-child {{ border-left: none; }}
.kpi-label {{ font-size: 0.8rem; color: {INK_3}; margin-bottom: 4px; }}
.kpi-value {{ font-size: 1.45rem; font-weight: 600; color: {INK}; line-height: 1.2; font-variant-numeric: tabular-nums; }}
.kpi-value.small {{ font-size: 1.05rem; line-height: 1.35; }}
.kpi-note {{ font-size: 0.78rem; color: {INK_3}; margin-top: 4px; }}

/* About block */
.about {{ display: grid; grid-template-columns: 3fr 2fr; gap: 28px; }}
.about h4 {{ font-size: 0.9rem; font-weight: 600; color: {INK}; margin: 0 0 6px 0; padding: 0; }}
.about p, .about li {{ font-size: 0.9rem; color: {INK_2}; line-height: 1.6; margin: 0 0 6px 0; }}
.about ol {{ margin: 0; padding-left: 1.15rem; }}
.facts {{ width: 100%; border-collapse: collapse; font-size: 0.87rem; border: none !important; }}
.facts tr {{ border: none !important; }}
.facts td {{ padding: 6px 0 !important; border: none !important; border-bottom: 1px solid #eef0f3 !important; color: {INK}; vertical-align: top; line-height: 1.5; }}
.facts td:first-child {{ color: {INK_3}; width: 42%; }}
.facts tr:last-child td {{ border-bottom: none !important; }}

/* Badges */
.badge {{ display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 0.74rem; font-weight: 500; margin-right: 6px; white-space: nowrap; }}
.b-red {{ background: #f7e9e6; color: #8f3f31; }}
.b-amber {{ background: #f6eeda; color: #6f5215; }}
.b-green {{ background: #e5f1eb; color: #2c6249; }}
.b-blue {{ background: #e6eef6; color: #2c5682; }}
.b-grey {{ background: #eef0f3; color: {INK_2}; }}

/* Ranked problems: rows in one panel */
.rank-row {{ display: grid; grid-template-columns: 28px 1fr 130px; gap: 12px; align-items: start; padding: 13px 18px; border-bottom: 1px solid #eef0f3; }}
.rank-row:last-child {{ border-bottom: none; }}
.rank-num {{ color: {INK_3}; font-weight: 600; font-size: 0.95rem; padding-top: 1px; font-variant-numeric: tabular-nums; }}
.rank-title {{ font-weight: 600; font-size: 0.96rem; color: {INK}; margin-bottom: 3px; }}
.rank-why {{ font-size: 0.86rem; color: {INK_2}; line-height: 1.5; max-width: 78ch; }}
.score-wrap {{ text-align: right; }}
.score-num {{ font-size: 0.86rem; color: {INK_2}; font-variant-numeric: tabular-nums; }}
.score-bar {{ height: 4px; background: #eef0f3; border-radius: 2px; overflow: hidden; margin-top: 6px; }}
.score-fill {{ height: 4px; background: {ACCENT}; opacity: 0.75; }}

/* Alerts and quotes */
.alert {{ padding: 10px 16px; border-bottom: 1px solid #eef0f3; }}
.alert:last-child {{ border-bottom: none; }}
.alert-title {{ font-weight: 600; color: {INK}; font-size: 0.9rem; }}
.alert-body {{ font-size: 0.83rem; color: {INK_2}; margin-top: 1px; }}
.quote {{ background: {SURFACE}; border: 1px solid {LINE}; border-left: 3px solid {NEU}; border-radius: 0 6px 6px 0; padding: 9px 13px; margin-bottom: 8px; }}
.quote.neg {{ border-left-color: {NEG}; }} .quote.pos {{ border-left-color: {POS}; }}
.quote-text {{ color: {INK}; font-size: 0.9rem; line-height: 1.55; }}
.quote-meta {{ color: {INK_3}; font-size: 0.76rem; margin-top: 5px; }}

/* Insight headline: a finding, stated plainly */
.headline {{ background: {SURFACE}; border: 1px solid {LINE}; border-left: 3px solid {ACCENT}; border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 6px 0 14px 0; }}
.headline-text {{ font-size: 1.15rem; font-weight: 500; line-height: 1.5; color: {INK}; max-width: 75ch; }}
.headline-meta {{ font-size: 0.78rem; color: {INK_3}; margin-top: 8px; }}
.rel-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; margin-bottom: 6px; }}
.rel {{ padding: 10px 14px; border-left: 1px solid {LINE}; }}
.rel:first-child {{ border-left: none; }}
.rel-label {{ font-size: 0.76rem; color: {INK_3}; }}
.rel-value {{ font-size: 1rem; font-weight: 600; color: {INK}; margin-top: 2px; }}
.rel-value.ok {{ color: {GOOD}; }} .rel-value.bad {{ color: {NEG}; }} .rel-value.mid {{ color: {WARN}; }}
.opportunity {{ background: {ACCENT_SOFT}; border-radius: 8px; padding: 14px 16px; margin-top: 4px; }}
.opportunity-title {{ font-weight: 600; font-size: 0.85rem; color: {ACCENT}; margin-bottom: 4px; }}
.opportunity div {{ color: {INK}; font-size: 0.92rem; line-height: 1.5; }}

/* Streamlit widgets */
div[data-testid="stExpander"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; }}
div[data-testid="stExpander"] summary p {{ font-size: 0.9rem; color: {INK}; }}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{ border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; }}
div[data-testid="stPlotlyChart"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; padding: 4px 6px; }}
/* Theme-proof widgets: the dashboard is designed for a light theme, so every control is coloured here
   explicitly. If a viewer's browser or Streamlit setting asks for dark mode, nothing turns dark-on-dark. */
:root, .stApp {{ color-scheme: light; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp label p,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], .stApp small {{ color: {INK_2} !important; }}
.stButton button, .stDownloadButton button, .stFormSubmitButton button {{
  border-radius: 6px; font-weight: 500; background: {SURFACE} !important; color: {INK} !important;
  border: 1px solid #d3d9df !important; box-shadow: none !important; }}
.stButton button p, .stDownloadButton button p, .stFormSubmitButton button p {{ color: inherit !important; }}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {{
  border-color: {ACCENT} !important; color: {ACCENT} !important; background: #fbfcfc !important; }}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"], .stDownloadButton button[kind="primary"],
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {{
  background: {ACCENT} !important; color: #ffffff !important; border-color: {ACCENT} !important; }}
.stButton button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{ background: #285f63 !important; color: #fff !important; }}
button:focus-visible {{ outline: 2px solid #9fc0c2 !important; outline-offset: 1px; }}
div[data-testid="stAlert"] {{ border-radius: 8px; background: #eaf2f6 !important; border: 1px solid #d4e3ec !important; }}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] div {{ color: {INK} !important; }}
/* text, number, date, select and multiselect inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div,
.stDateInput > div > div, .stTextInput > div > div, .stNumberInput > div > div {{
  background: {SURFACE} !important; border-color: #d3d9df !important; }}
[data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] input, [data-baseweb="select"] div, .stDateInput input {{ color: {INK} !important; background: transparent !important; }}
input::placeholder, textarea::placeholder {{ color: {INK_3} !important; opacity: 1; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [data-baseweb="calendar"],
[data-baseweb="popover"] > div {{ background: {SURFACE} !important; color: {INK} !important; }}
[data-baseweb="popover"] li, [data-baseweb="calendar"] div, [data-baseweb="calendar"] button {{ color: {INK} !important; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {{ background: {ACCENT_SOFT} !important; }}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSliderThumbValue"] {{ color: {INK_2} !important; }}
div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary {{ background: {SURFACE} !important; color: {INK} !important; }}
div[data-testid="stExpander"] summary:hover p {{ color: {ACCENT} !important; }}
.stTabs [data-baseweb="tab"] p {{ color: {INK_2}; }} .stTabs [aria-selected="true"] p {{ color: {ACCENT}; font-weight: 600; }}
/* chat */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: {CANVAS} !important; }}
[data-testid="stChatInput"] {{ background: {SURFACE} !important; border: 1px solid #d3d9df !important; border-radius: 10px !important; }}
[data-testid="stChatInput"] > div {{ background: {SURFACE} !important; }}
[data-testid="stChatInput"] textarea {{ color: {INK} !important; background: {SURFACE} !important; }}
[data-testid="stChatInput"] button {{ background: {ACCENT} !important; color: #fff !important; border-radius: 8px !important; }}
[data-testid="stChatInput"] button:disabled {{ background: #dfe4e8 !important; color: {INK_3} !important; }}
[data-testid="stChatMessage"] {{ background: transparent !important; padding: 10px 4px; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {{ border-radius: 10px; }}
[data-testid="stChatMessageAvatarCustom"], [data-testid^="stChatMessageAvatar"] {{
  background: {ACCENT_SOFT} !important; color: {ACCENT} !important; border-radius: 8px; }}
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {{ color: {INK} !important; line-height: 1.6; }}
[data-testid="stSpinner"] p, [data-testid="stSpinner"] div {{ color: {INK_2} !important; }}
.stButton button {{ height: auto; min-height: 42px; padding-top: 8px; padding-bottom: 8px; }}
.stButton button p {{ white-space: normal !important; line-height: 1.35; }}
@media (max-width: 900px) {{
  .stats, .rel-grid {{ grid-template-columns: repeat(2, 1fr) !important; }}
  .kpi, .rel {{ border-left: none; border-top: 1px solid {LINE}; }}
  .about {{ grid-template-columns: 1fr; }}
  .rank-row {{ grid-template-columns: 24px 1fr; }} .score-wrap {{ display: none; }}
}}
</style>
"""


def e(text) -> str:
    return html.escape(str(text))


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='page-title'>{e(title)}</div>"
                + (f"<div class='page-sub'>{e(subtitle)}</div>" if subtitle else ""), unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f"<div class='section-title'>{e(title)}</div>" + (f"<div class='section-note'>{e(note)}</div>" if note else ""),
                unsafe_allow_html=True)


def kpi(label: str, value, note: str = "", badge_html: str = "", small: bool = False) -> str:
    return (f"<div class='kpi'><div class='kpi-label'>{e(label)}</div>"
            f"<div class='kpi-value{' small' if small else ''}'>{e(value)}</div>"
            f"<div class='kpi-note'>{badge_html}{e(note)}</div></div>")


def kpi_row(items: list[str]) -> None:
    """Related numbers share one panel, separated by hairlines."""
    st.markdown(f"<div class='stats' style='grid-template-columns: repeat({len(items)}, 1fr)'>{''.join(items)}</div>",
                unsafe_allow_html=True)


def badge(text: str, kind: str = "grey") -> str:
    return f"<span class='badge b-{kind}'>{e(text)}</span>"


def panel(inner_html: str, pad: bool = False) -> None:
    st.markdown(f"<div class='panel{' pad' if pad else ''}'>{inner_html}</div>", unsafe_allow_html=True)


TREND_BADGE = {"Rising": "red", "New": "red", "Declining": "green", "Stable": "grey"}
EVIDENCE_BADGE = {"High": "grey", "Medium": "amber", "Low": "red"}


def rank_card(rank: int, title: str, why: str, score: float, max_score: float, trend: str, evidence: str) -> str:
    pct = 0 if not max_score else max(4, min(100, 100 * score / max_score))
    badges = (badge(trend, TREND_BADGE.get(trend, "grey")) if trend != "Stable" else "") + \
             (badge(f"{evidence} evidence", EVIDENCE_BADGE.get(evidence, "grey")) if evidence != "High" else "")
    return (f"<div class='rank-row'><div class='rank-num'>{rank}</div>"
            f"<div><div class='rank-title'>{e(title)} &nbsp;{badges}</div><div class='rank-why'>{e(why)}</div></div>"
            f"<div class='score-wrap'><div class='score-num'>Priority score {score:.0f}</div>"
            f"<div class='score-bar'><div class='score-fill' style='width:{pct:.0f}%'></div></div></div></div>")


def alert(title: str, body: str) -> str:
    return f"<div class='alert'><div class='alert-title'>{e(title)}</div><div class='alert-body'>{e(body)}</div></div>"


def quote(text: str, record_id: str, platform: str = "", date: str = "", sentiment: str = "") -> str:
    cls = {"Negative": "neg", "Positive": "pos"}.get(sentiment, "")
    when = str(date)[:10] if date is not None and str(date) not in ("", "NaT", "None") else ""
    meta = ", ".join(x for x in [platform, when, f"record {record_id}"] if x)
    return f"<div class='quote {cls}'><div class='quote-text'>{e(text)}</div><div class='quote-meta'>{e(meta)}</div></div>"


def rel_tile(label: str, value, tone: str = "") -> str:
    return f"<div class='rel'><div class='rel-label'>{e(label)}</div><div class='rel-value {tone}'>{e(value)}</div></div>"


def style_fig(fig, height: int = 360, title: str | None = None):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=44 if title or fig.layout.title.text else 14, b=8),
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=12.5, color=INK_2),
        title=dict(text=title if title is not None else fig.layout.title.text, font=dict(size=14, color=INK), x=0.01),
        legend=dict(orientation="h", y=-0.16, title_text=""), hoverlabel=dict(bgcolor="white", font_size=12, font_color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=INK_3), title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#eef0f3", zeroline=False, tickfont=dict(color=INK_3), title_font=dict(size=12))
    return fig
