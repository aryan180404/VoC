"""Run the whole Voice of Customer pipeline in one command.

    python run_pipeline.py                 # Gemini if GEMINI_API_KEY is set, otherwise rules
    python run_pipeline.py --engine rules  # free, offline baseline
    python run_pipeline.py --trial 200     # try Gemini on 200 records first to check cost and quality

Steps: clean -> label -> insights -> evaluate (if the answer key is present).
"""
from __future__ import annotations

import argparse
import json
import time

from src import config, enrich, evaluate, ingest, insights, llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["auto", "gemini", "rules"], default="auto")
    ap.add_argument("--trial", type=int, help="label only the first N records (does not overwrite results)")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    print("1/4 Cleaning raw files ...")
    table, report = ingest.build_clean_table()
    table.to_parquet(config.PROCESSED / "clean.parquet", index=False)
    (config.PROCESSED / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"    {report['raw_records']} records, {report['rule_noise']} noise, {report['duplicates']} duplicates, "
          f"{report['personal_details_redacted']} with personal details removed")

    engine = a.engine
    if engine == "auto":
        engine = "gemini" if config.llm_available() else "rules"
    print(f"2/4 Labelling with: {engine}" + ("" if config.llm_available() else "  (no GEMINI_API_KEY found, using the rules baseline)"))
    try:
        df = enrich.run(engine, a.trial, progress=lambda d, n: print(f"    {d}/{n} labelled", end="\r"))
    except llm.FatalLLMError as exc:
        print(f"\nSTOPPED: {exc}")
        raise SystemExit(1)
    print(f"\n    label sources: {df.label_source.value_counts().to_dict()}")
    if a.trial:
        print("Trial finished. Results saved to data/processed/enriched_trial_*.parquet; full results unchanged.")
        from src.llm import USAGE
        print(f"Gemini usage: {USAGE.calls} calls, {USAGE.input_tokens} input tokens, {USAGE.output_tokens} output tokens")
        return

    print("3/4 Writing evidence-backed insights ...")
    for ins in insights.run():
        flag = "" if ins["verification"]["numbers_verified"] and ins["verification"]["citations_verified"] else "  [CHECK]"
        print(f"    - {ins['headline'][:110]}{flag}")

    if not a.skip_eval and (config.EVAL_DIR / "zomato_voice_of_customer_answer_key.csv").exists():
        print("4/4 Scoring against the answer key ...")
        res = evaluate.run()
        for eng, r in res["engines"].items():
            h = r["classification_holdout"]
            print(f"    {eng:7s} sentiment {h['sentiment_accuracy']:.1%} | primary issue {h['primary_issue_correct']:.1%} "
                  f"| issue F1 {h['issue_micro']['f1']:.2f}")
        print(f"    incidents detected: {sum(x['detected'] for x in res['incident_detection'])}/4")
        print("    evaluation sheet: reports/evaluation_sheet.xlsx")
    print(f"Done in {time.time() - t0:.0f}s. Start the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
