"""Daily refresh (the "advanced version": a scheduled job).

What it does each run:
  1. Moves any new export files from data/inbox/ into data/raw/.
  2. Re-runs cleaning. Labelling is cached by review text, so only NEW reviews
     cost a Gemini call.
  3. Recomputes trends and insights.
  4. Writes reports/alerts_YYYY-MM-DD.md listing new spikes and newly rising issues
     compared with the previous run.

Schedule it (Linux or macOS), every day at 7:10 in the morning:
    crontab -e
    10 7 * * * cd /path/to/voc-copilot && /usr/bin/python3 scripts/daily_refresh.py >> reports/refresh.log 2>&1

On Windows use Task Scheduler with the same command. A GitHub Actions version is in
.github/workflows/daily_refresh.yml (store GEMINI_API_KEY as a repository secret).

Collectors for live sources (Google Play, App Store, Reddit) are the next step:
each one only needs to drop a CSV with the same 11 columns into data/inbox/.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analytics, config, enrich, ingest, insights  # noqa: E402

STATE = config.PROCESSED / "refresh_state.json"


def main() -> None:
    moved = []
    for f in sorted(config.INBOX_DIR.glob("*.csv")):
        target = config.RAW_DIR / f.name
        shutil.move(str(f), target)
        moved.append(f.name)

    table, report = ingest.build_clean_table()
    table.to_parquet(config.PROCESSED / "clean.parquet", index=False)
    (config.PROCESSED / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))
    df = enrich.run("auto")
    tables = analytics.compute_all(df)
    insights.run()

    previous = json.loads(STATE.read_text()) if STATE.exists() else {"alerts": [], "rising": [], "records": 0}
    alerts = [f"{a['issue']}|{a['month']}" for a in tables["spike_alerts"]]
    rising = tables["trend"][tables["trend"].status.isin(["Rising", "New"])].issue.tolist()
    new_alerts = [a for a in alerts if a not in previous["alerts"]]
    new_rising = [r for r in rising if r not in previous["rising"]]

    lines = [f"# Voice of Customer refresh, {date.today().isoformat()}", "",
             f"- New files: {', '.join(moved) or 'none'}",
             f"- Records now: {report['raw_records']} (previous run: {previous['records']})",
             f"- Labelling engine: {df.engine.iloc[0]}", "", "## New spike alerts"]
    lines += [f"- {a.replace('|', ' in ')}" for a in new_alerts] or ["- None"]
    lines += ["", "## Newly rising issues"] + ([f"- {r}" for r in new_rising] or ["- None"])
    out = config.REPORTS / f"alerts_{date.today().isoformat()}.md"
    out.write_text("\n".join(lines))
    STATE.write_text(json.dumps({"alerts": alerts, "rising": rising, "records": report["raw_records"]}))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
