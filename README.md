# Voice of Customer Copilot

**Turns messy, multi-platform customer feedback into evidence-backed product decisions.**
Built on a 5,237-record synthetic Zomato dataset covering Google Play, App Store, X, Reddit, YouTube, Quora, complaint forums, LinkedIn and news comments, in 8 languages.

> Not an AI summariser. The model labels each review; code counts; every insight links to real reviews; a product manager makes the call.

## Six modules

| Module | What it answers |
|---|---|
| 1. Feedback Hub | What did we receive, and what did cleaning change? |
| 2. Pain Points | What do customers complain about, where, and what do they praise? |
| 3. Investigation | What patterns sit inside a problem, what might cause them, and what should we check or ask? |
| 4. Trends & Releases | What is rising, spiking or declining? Did something change after a release? |
| 5. Decision Engine | Which problem first, and why? Approve, modify or reject, then export to Jira or Notion and track the outcome |
| 6. Evaluation Centre | How accurate is it, where does it fail, and do the dashboard numbers match the truth? |

**Deploy it:** see [DEPLOY.md](DEPLOY.md) (GitHub + Streamlit Community Cloud, about 15 minutes).

## Quick start (laptop)

```bash
pip install -r requirements.txt
cp .env.example .env            # then paste your Gemini key into .env
python run_pipeline.py --trial 200   # optional: test Gemini on 200 records first
python run_pipeline.py               # full run: clean → label → insights → evaluate
streamlit run app.py
```

With no key, everything runs on the free rule-based baseline, so you can demo the tool offline.
Labels are cached, so re-running costs nothing for reviews already labelled.

**Daily refresh:** `python scripts/daily_refresh.py` picks up new files dropped into `data/inbox/`. The cron and GitHub Actions setup is described in the script.

**Tests:** `python -m pytest -q` (12 guardrail tests, using a fake Gemini).

## How it works

```
raw exports ──▶ cleaning (code) ──▶ labelling (Gemini Flash, one review at a time, validated)
                                         │  fallback: rule-based classifier
                                         ▼
                 counts · rates · spikes · ranking (code) ──▶ insights (Gemini Pro writes, code verifies)
                                         ▼
                      dashboard ──▶ product manager decision ──▶ outcome tracking
```

## Results (rule-based baseline; Gemini results appear after your first run)

| | Score |
|---|---|
| Noise removal F1 | 0.99 |
| Duplicates merged correctly (recall / precision) | 0.89 / 0.91 |
| Personal details removed | 100% |
| Sentiment accuracy, 70% holdout split | 87.2% |
| Primary issue correct, holdout | 75.1% |
| Planted incidents detected in the right month | 4 of 4 |
| False alarm on the control release | None |
| Monthly negative share versus truth | within 1.9 points on average |
| Top-5 problems matching the truth | 2 of 5. The keyword baseline over-counts filler words. This is the gap Gemini should close |

The answer key audit found **147 Hinglish rows with self-contradictory sentiment labels** (see `reports/answer_key_audit.csv`).

## Portfolio deliverables

| Deliverable | Where |
|---|---|
| Working demo | `streamlit run app.py` (deploy free on Streamlit Community Cloud) |
| Loom video | Script in `docs/demo_video_script.md` |
| Evaluation sheet | `reports/evaluation_sheet.xlsx` |
| Real-user feedback | Kit and change log in `docs/user_validation_kit.md` |
| Product requirements | `docs/product_requirements.md` |

## Project map

```
app.py                      dashboard (six modules)
run_pipeline.py             one-command pipeline
src/ingest.py               cleaning, personal-detail removal, duplicates, noise
src/taxonomy.py             locked category list
src/enrich.py               Gemini labelling, prompt, validation, cache, fallback
src/rules_classifier.py     baseline and fallback classifier
src/llm.py                  Gemini client: model choice, pacing, retries, usage
src/analytics.py            all numbers: rates, spikes, ranking, release impact
src/insights.py             evidence packs, hypotheses, citation checks, "ask the data"
src/evaluate.py             scoring, answer key audit, evaluation sheet
scripts/daily_refresh.py    scheduled job
data/context/releases.csv   release dates (INVENTED, matching the planted incidents)
data/eval/                  answer key (synthetic; read only by evaluate.py, never by the labeller)
.github/workflows/          Gemini pipeline run (manual) and daily refresh (scheduled)
```

## Data note

All records are synthetic. The layout and categories are modelled on [Pp2k05/ZOMATO-ECOSYSTEM-ANALYSIS](https://github.com/Pp2k05/ZOMATO-ECOSYSTEM-ANALYSIS). The four incidents and the release dates are invented, not real Zomato events.
