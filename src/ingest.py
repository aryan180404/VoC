"""Feedback Hub: turn messy multi-platform exports into one clean table.

Everything here is ordinary deterministic code, not the language model:
cleaning must be cheap, repeatable and auditable. Each record keeps its raw
values next to the cleaned ones, plus flags saying what was changed, so a
product manager can always ask "what did the tool do to this row?".
"""
from __future__ import annotations

import csv
import hashlib
import html
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import ftfy
except ImportError:  # optional; a small manual fixer is used instead
    ftfy = None

from . import config

RAW_COLUMNS = ["source_id", "raw_brand", "raw_platform", "raw_date", "username", "title",
               "raw_text", "raw_rating", "raw_likes", "extra_info", "raw_source"]

# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_raw_file(path: Path) -> pd.DataFrame:
    """Read a messy export: skip comment lines, blank lines and repeated headers."""
    rows, skipped = [], {"comment": 0, "blank": 0, "header": 0, "malformed": 0}
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.reader(fh):
            if not row or all(not c.strip() for c in row):
                skipped["blank"] += 1
                continue
            if len(row) == 1 and row[0].lstrip().startswith("#"):
                skipped["comment"] += 1
                continue
            if row[0].strip().lower() == "id":
                skipped["header"] += 1
                continue
            if len(row) != len(RAW_COLUMNS):
                skipped["malformed"] += 1
                continue
            rows.append(row)
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df.attrs["skipped"] = skipped
    df["file"] = path.name
    return df


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
_MOJIBAKE = {"â‚¹": "₹", "â€™": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€”": "-", "Ã©": "é"}


def clean_text(text: str) -> str:
    t = text or ""
    if ftfy is not None:
        t = ftfy.fix_text(t)
    else:
        for bad, good in _MOJIBAKE.items():
            t = t.replace(bad, good)
    t = html.unescape(t)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


_PHONE = re.compile(r"(?:\+?91[\s-]?)?(?<!\d)[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def redact_personal_details(text: str) -> tuple[str, bool]:
    new = _EMAIL.sub("[email removed]", text)
    new = _PHONE.sub("[phone removed]", new)
    return new, new != text


# --------------------------------------------------------------------------
# Brand and platform
# --------------------------------------------------------------------------
OTHER_BRANDS = ("swiggy", "blinkit", "district")


def normalise_brand(raw: str) -> str:
    b = (raw or "").strip().lower()
    if not b:
        return "Unknown"
    for other in OTHER_BRANDS:
        if other in b:
            return other.title()
    if "zom" in b or "eternal" in b:
        return config.OWN_BRAND
    return "Unknown"


PLATFORM_RULES = [  # checked against the Source column first, then the Platform column
    (r"play\.google|google ?play|play ?store|gplay|playstore", "Google Play"),
    (r"apps\.apple|app ?store|appstore|\bios\b|apple", "Apple App Store"),
    (r"reddit|\br/", "Reddit"),
    (r"twitter|x\.com|x search|^x$|^x \(twitter\)$|@zomato", "X (Twitter)"),
    (r"youtube|^yt$", "YouTube"),
    (r"quora", "Quora"),
    (r"consumercomplaints|consumer forum|complaint board|^forum$", "Consumer Complaints Forum"),
    (r"mouthshut|review site", "MouthShut"),
    (r"linkedin", "LinkedIn"),
    (r"times of india|moneycontrol|economic times|news|^comments$", "News Site Comments"),
]


def _match_platform(value: str) -> str | None:
    v = (value or "").strip().lower()
    if not v:
        return None
    for pattern, name in PLATFORM_RULES:
        if re.search(pattern, v):
            return name
    return None


_GENERIC_PLATFORM = {"forum", "review site", "comments", "platform", ""}


def normalise_platform(raw_platform: str, raw_source: str, source_id: str) -> str:
    # The Platform column is what the scraper recorded; the Source column is sometimes a
    # merged-file label. Trust Platform unless it is a generic word like "Forum".
    by_platform = _match_platform(raw_platform)
    by_source = _match_platform(raw_source)
    if by_platform and raw_platform.strip().lower() not in _GENERIC_PLATFORM:
        return by_platform
    if by_source:
        return by_source
    if by_platform:
        return by_platform
    if re.fullmatch(r"t3_\w+|rd-\d+", source_id or ""):
        return "Reddit"
    if re.fullmatch(r"18\d{17}", source_id or ""):
        return "X (Twitter)"
    return "Unknown"


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
_DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%a %b %d %H:%M:%S %z %Y",
    "%d-%m-%Y %H:%M", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%B %d, %Y", "%b %d, %Y",
    "%d.%m.%Y",
]
_REL = re.compile(r"(\d+|a|an)\s+(day|week|month|year)s?\s+ago", re.I)


def _in_window(d: date) -> bool:
    return config.WINDOW_START - timedelta(days=31) <= d <= config.WINDOW_END + timedelta(days=31)


def parse_date(raw: str) -> tuple[date | None, str]:
    """Return (date, how_it_was_parsed). The second value feeds data-quality reporting."""
    s = (raw or "").strip()
    if not s:
        return None, "missing"
    if re.fullmatch(r"\d{9,11}", s):
        return datetime.fromtimestamp(int(s), tz=timezone.utc).date(), "unix_timestamp"
    m = _REL.search(s)
    if m:
        n = 1 if m.group(1).lower() in ("a", "an") else int(m.group(1))
        days = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}[m.group(2).lower()] * n
        est = config.EXPORT_DATE - timedelta(days=round(days))
        return min(est, config.WINDOW_END), "relative_estimate"
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date(), "exact"
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12:
            return date(y, b, a), "exact"            # day first
        if b > 12:
            return date(y, a, b), "exact"            # month first (US export)
        return date(y, b, a), "ambiguous_day_month"  # Indian day/month order assumed
    m = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{1,2})", s)  # "Mar 14" with no year
    if m:
        for fmt in ("%b %d %Y", "%B %d %Y"):
            try:
                d = datetime.strptime(f"{s} 2000", fmt).date()  # 2000 is a leap year, so 29 Feb parses
            except ValueError:
                continue
            year = config.WINDOW_START.year if d.month >= config.WINDOW_START.month else config.WINDOW_END.year
            return d.replace(year=year), "year_inferred"
    return None, "unparsed"


# --------------------------------------------------------------------------
# Ratings
# --------------------------------------------------------------------------
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse_rating(raw: str) -> float | None:
    s = (raw or "").strip().lower()
    if not s:
        return None
    if "★" in s:
        return float(s.count("★"))
    if s in _WORDS:
        return float(_WORDS[s])
    m = re.match(r"(\d(?:\.\d)?)\s*(?:/\s*5|stars?)?$", s)
    if m:
        v = float(m.group(1))
        return v if 1 <= v <= 5 else None
    return None


# --------------------------------------------------------------------------
# Language (script detection plus Hinglish word markers)
# --------------------------------------------------------------------------
_SCRIPTS = [
    ("Tamil", r"[\u0B80-\u0BFF]"), ("Telugu", r"[\u0C00-\u0C7F]"), ("Kannada", r"[\u0C80-\u0CFF]"),
    ("Bengali", r"[\u0980-\u09FF]"), ("Devanagari", r"[\u0900-\u097F]"),
]
_MARATHI = re.compile(r"आहे|नाही|झाल|खूप|आणि|पण|मिळाल|केला|होते|आला|अजून|आम्ही|माझ")
_HINGLISH = re.compile(
    r"\b(hai|nahi|nahin|bahut|kya|yaar|wale|kuch|bhi|toh|mein|aur|bas|rehte|karte|kar|diya|"
    r"gaya|liya|zyada|accha|acha|bhai|kab|kyun|pe|se|ka|ki|ko|bola|abhi|tak)\b", re.I)


def detect_language(text: str) -> str:
    for name, pattern in _SCRIPTS:
        if re.search(pattern, text):
            if name == "Devanagari":
                return "Marathi" if _MARATHI.search(text) else "Hindi"
            return name
    if len(_HINGLISH.findall(text)) >= 2:
        return "Hinglish"
    return "English"


# --------------------------------------------------------------------------
# Noise (only high-confidence rules; borderline rows go to the model)
# --------------------------------------------------------------------------
NOISE_RULES = [
    ("brand_reply", r"we have shared this with the concerned team|please check your registered email|"
                    r"sorry for the inconvenience caused|kindly dm us|dm us your order|we'?d like to look into this|"
                    r"please share your order id|share your registered number|- ?zomato care$|team zomato$"),
    ("job_post", r"we'?re hiring|i'?ve joined zomato|joined zomato as|is looking for a .{0,40}(analyst|manager|engineer|executive|intern)|apply via|apply now|job opening"),
    ("spam", r"check out my channel|subscribe to my|referral code|earn ₹?\s?\d+.{0,20}(daily|from home)|dm me|bit\.ly|100% working|free zomato gold|click here|whatsapp me"),
    ("off_topic_stock", r"\bshares\b|stock|\bnse\b|\bbse\b|target price|profit booking|portfolio|\bq[1-4]\b results|share price|market cap"),
]
JUNK_TEXTS = {"test", "testing", "na", "n/a", "null", "none", "first", "lol", "nice video", "asdf", "asdfgh", "qwerty", "hi", "hello"}
_OWN_MENTION = re.compile(r"zomato|zomto|\bgold\b|this app", re.I)
_OTHER_MENTION = re.compile(r"swiggy|blinkit|district", re.I)


BRAND_ACCOUNTS = {"zomato", "zomatocare", "zomato care", "zomato support", "zomato_care"}


def rule_noise(text: str, brand: str, username: str = "") -> str | None:
    if normalise_user(username) in BRAND_ACCOUNTS:
        return "brand_reply"
    t = text.strip()
    if not t:
        return "empty"
    if len(re.sub(r"[\W_]+", "", t)) < 2 or t.lower().strip(" .!") in JUNK_TEXTS:
        return "junk"
    low = t.lower()
    for label, pattern in NOISE_RULES:
        if re.search(pattern, low):
            return label
    if brand in ("Swiggy", "Blinkit", "District"):
        return "other_brand"
    # Only when the other brand is the subject. "Switching to Swiggy" is a competitor
    # comparison about Zomato, which is valuable signal, not noise.
    if re.match(r"^\W*(swiggy|blinkit|district)\b", low) and not _OWN_MENTION.search(low):
        return "other_brand"
    return None


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------
_PREFIXES = re.compile(r"^(\[r/\w+\]\s*|comment on '[^']*':\s*|rant:\s*|reply:\s*|q:.*?a:\s*|updating my review:\s*)", re.I)


def dedupe_key(text: str) -> str:
    t = text.lower()
    for _ in range(3):
        t = _PREFIXES.sub("", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"[^\w\s₹]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# Platforms hide real names behind placeholders; these must never link two reviews.
PLACEHOLDER_USERS = {"a google user", "anonymous", "user", "guest", "deleted", "[deleted]", "customer", "iphone user", "android user"}


def normalise_user(u: str) -> str:
    return re.sub(r"^(@|u/)", "", (u or "").strip().lower())


def find_duplicates(df: pd.DataFrame) -> pd.Series:
    """Return duplicate_of (record_id of the first copy) for every repeated record.

    Rules, strictest first:
      1. same source ID and the same cleaned text (re-export of the same row)
      2. same user and near-identical text (cross-post or edited re-post)
    Identical short texts from DIFFERENT users ("Good app") are kept, by design.
    """
    from rapidfuzz import fuzz

    dup_of = pd.Series("", index=df.index, dtype=object)
    first_by_id_text: dict[tuple[str, str], str] = {}
    by_user: dict[str, list[tuple[str, str]]] = {}
    for idx, row in df.iterrows():
        if row.get("rule_noise_type"):
            continue  # noise is dropped anyway; brand replies repeat by design
        key = dedupe_key(row["text"])
        rid = row["record_id"]
        k1 = (row["source_id"], key)
        if row["source_id"] and k1 in first_by_id_text:
            dup_of[idx] = first_by_id_text[k1]
            continue
        first_by_id_text.setdefault(k1, rid)
        user = normalise_user(row["username"])
        if user and user not in PLACEHOLDER_USERS and len(key) >= 25:
            for other_key, other_rid in by_user.get(user, []):
                if fuzz.ratio(key, other_key) >= 92:
                    dup_of[idx] = other_rid
                    break
            if not dup_of[idx]:
                by_user.setdefault(user, []).append((key, rid))
    return dup_of


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def _slash_order(raw: str) -> str | None:
    m = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/\d{4}\s*", raw or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return "day_first" if a > 12 else "month_first" if b > 12 else None


def _resolve_ambiguous_dates(df: pd.DataFrame) -> None:
    """Dates like 03/04/2026 are ambiguous. Each export source uses one convention,
    so learn it from that source's unambiguous dates (e.g. 12/23/2025 means month first)."""
    orders = df.raw_date.map(_slash_order)
    for source, grp in orders.groupby(df.raw_source):
        counts = grp.value_counts()
        if counts.get("month_first", 0) > counts.get("day_first", 0):
            mask = (df.raw_source == source) & (df.date_quality == "ambiguous_day_month")
            for idx in df.index[mask]:
                a, b, y = map(int, df.at[idx, "raw_date"].strip().split("/"))
                df.at[idx, "date"] = date(y, a, b)
                df.at[idx, "date_quality"] = "ambiguous_resolved_by_source"


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_clean_table(paths: list[Path] | None = None) -> tuple[pd.DataFrame, dict]:
    paths = paths or sorted(config.RAW_DIR.glob("*.csv"))
    frames, report = [], {"files": [], "skipped_lines": {}}
    for p in paths:
        f = read_raw_file(p)
        frames.append(f)
        report["files"].append({"file": p.name, "records": len(f)})
        for k, v in f.attrs["skipped"].items():
            report["skipped_lines"][k] = report["skipped_lines"].get(k, 0) + v
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "record_id", [f"R{i + 1:05d}" for i in range(len(df))])

    df["text"] = [clean_text(f"{t}" if not ti else f"{ti}. {t}") for t, ti in zip(df.raw_text, df.title)]
    redacted = df.text.map(redact_personal_details)
    df["text"] = [r[0] for r in redacted]
    df["had_personal_details"] = [r[1] for r in redacted]
    df["username"] = df.username.map(lambda u: "" if _PHONE.search(u or "") or _EMAIL.search(u or "") else u)

    df["brand"] = df.raw_brand.map(normalise_brand)
    df["platform"] = [normalise_platform(p, s, i) for p, s, i in zip(df.raw_platform, df.raw_source, df.source_id)]
    parsed = df.raw_date.map(parse_date)
    df["date"] = [p[0] for p in parsed]
    df["date_quality"] = [p[1] for p in parsed]
    _resolve_ambiguous_dates(df)
    df["rating"] = df.raw_rating.map(parse_rating)
    df["language"] = df.text.map(detect_language)
    df["likes"] = df.raw_likes.str.extract(r"(\d+)", expand=False).fillna(0).astype(int)
    df["app_version"] = df.extra_info.str.extract(r"(\d{2}\.\d+\.\d+)", expand=False).fillna("")
    df["rule_noise_type"] = [rule_noise(t, b, u) or "" for t, b, u in zip(df.text, df.brand, df.username)]
    df["duplicate_of"] = find_duplicates(df)
    df["is_duplicate"] = df.duplicate_of != ""
    df["text_hash"] = df.text.map(text_hash)

    report.update({
        "raw_records": int(len(df)),
        "rule_noise": int((df.rule_noise_type != "").sum()),
        "duplicates": int(df.is_duplicate.sum()),
        "personal_details_redacted": int(df.had_personal_details.sum()),
        "date_quality": df.date_quality.value_counts().to_dict(),
        "ratings_present": int(df.rating.notna().sum()),
        "platform_spellings_unified": int(df.raw_platform.str.strip().nunique()),
        "brand_spellings_unified": int(df.raw_brand.str.strip().nunique()),
    })
    return df, report


if __name__ == "__main__":
    import json

    table, rep = build_clean_table()
    out = config.PROCESSED / "clean.parquet"
    table.to_parquet(out, index=False)
    (config.PROCESSED / "cleaning_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
