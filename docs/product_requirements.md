# Voice of Customer Copilot: Product Requirements Document

**Owner:** Ari · **Status:** Version 1 built, awaiting Gemini run and user validation · **Last updated:** 12 November 2026

> **Positioning.** A continuous product-intelligence tool that turns scattered customer feedback into evidence-backed product opportunities, prioritisation decisions and measurable outcomes.
> It supports product decisions. It does not replace the product manager: the model investigates and recommends, and the product manager decides.

---

## 1. Problem

Customer feedback for a consumer app like Zomato arrives from many places at once: Play Store and App Store reviews, X (Twitter), Reddit, YouTube comments, Quora, complaint forums and news comment sections. It comes in many languages and in many messy formats.

A product manager today does one of three things:

1. **Reads a sample.** Fast, but biased toward whatever is loudest this week.
2. **Asks an analyst for a dashboard.** Takes days, and shows *what* without *why* or *so what*.
3. **Pastes reviews into a chatbot.** Fast and fluent, but the numbers can't be checked, quotes may be made up, and the themes change every time it runs.

**The real job:** *"Tell me which customer problems matter most right now, show me the proof, help me work out why they happen, and tell me if what we shipped helped."*

## 2. Users

| User | Situation | What they need |
|---|---|---|
| **Primary: product manager** (consumer app, 2 to 6 squads' worth of surface) | Weekly prioritisation, roadmap reviews, incident follow-up | A ranked list of problems with reasons, real quotes, trends, a way to record decisions |
| Secondary: product leader | Monthly reviews | Trust in the numbers; what changed since last month |
| Secondary: customer support lead | Support quality | Which support failures drive negative feedback |

## 3. Goals and non-goals

**Goals (version 1)**
- Clean and unify messy multi-platform feedback automatically, removing personal details before any model sees it.
- Label every review with sentiment, specific issues, severity and churn risk, using a fixed category list.
- Rank problems with an explanation a product manager can challenge.
- Attach real, checkable customer quotes to every insight.
- Detect what is changing: spikes, rising and declining issues, before-and-after a release.
- Measure the model's reliability openly against an answer key.

**Non-goals (version 1)**
- Proving root causes. Reviews show symptoms; causes are always labelled as hypotheses.
- Acting on its own: it creates no tickets and changes no roadmap without a person approving.
- Live scraping of every platform. Version 1 reads exported files; collectors drop files into an inbox folder.
- Replacing quantitative product analytics such as funnels and crash logs. It points to which ones to check.

## 4. What good output looks like

The rule that separates this from an AI summariser:

| Bad output | Good output |
|---|---|
| "Users are unhappy with Gold." | "193 of 3,293 negative reviews (5.9%) describe Gold charges. They rose from 1.1% of all feedback (June to August) to 15.4% (September to October). The main patterns are auto-renewal without a reminder and the free-delivery distance cap. Example: `R02455` 'Zomato Gold renewed automatically, no notification, no easy way to cancel.'" |
| "Delivery is a problem." | Same, plus the patterns inside it (rain delays with a rain fee still charged; sarcastic 'great job, 90 minutes'), hypotheses to test, the data that would confirm each one, and how many reviews *praise* delivery |

**Every insight must answer "based on what?"** It needs a count, a share, a trend, and quotes that a reviewer can open.

## 5. Workflow (input → processing → output → human review)

```
 Exports (CSV, any platform)          ┌─────────────── Code only ───────────────┐
 ─────────────────────────────▶  1. FEEDBACK HUB
                                    read · fix encoding · unify platform/brand ·
                                    parse 12 date formats · parse ratings ·
                                    remove phone numbers and emails · drop noise ·
                                    merge duplicates · detect language
                                                     │
                                ┌──── Gemini (Flash): labels ONE review at a time ────┐
                                 2. INTELLIGENCE ENGINE
                                    sentiment · 1 to 3 issues from a locked list · severity ·
                                    churn risk · sarcasm · feature request · exact evidence quote
                                    Guardrails: schema check, quote must exist in review,
                                    missing records retried, then rules fallback
                                                     │
                                ┌─────────────── Code only ───────────────┐
                                 counts · shares · trends · spikes (significance-tested) ·
                                 opportunity score with reasons · release before/after
                                                     │
                                ┌──── Gemini (Pro): writes, never counts ────┐
                                 3. INVESTIGATION
                                    names patterns · hypotheses with 2+ citations ·
                                    data to check · interview questions · opportunity
                                    Guardrails: numbers checked against evidence, citations checked
                                                     │
                                 4. TRENDS & RELEASES   5. DECISION ENGINE   6. EVALUATION CENTRE
                                                     │
                                 HUMAN REVIEW: product manager approves, modifies or rejects;
                                 adds impact, confidence, effort; records shipped date
                                                     │
                                 Closed loop: did the problem's share of feedback fall after shipping?
```

## 6. Requirements and priority

| Feature | Priority | Status |
|---|---|---|
| Feedback ingestion and cleaning | Must have | Built |
| Categorisation, sentiment, clustering | Must have | Built (Gemini plus rules fallback) |
| Evidence-backed insights (quotes, record IDs) | Must have | Built |
| Evaluation framework | Must have | Built |
| Pain point dashboard | Must have | Built |
| Trend intelligence (rate-based, spikes, rising and declining) | Should have | Built |
| Root-cause investigation (hypotheses, data to check, interview questions) | Should have | Built |
| Prioritisation engine (weighted score plus reach, impact, confidence and effort) | Should have | Built |
| Release intelligence (before and after) | Could have | Built, with invented release dates |
| Export to Jira or Notion | Could have | CSV export built; direct connection later |
| Daily scheduled refresh | Could have | Script plus scheduler setup built |
| Closed-loop outcome tracking | Long-term vision | First version built |

## 7. How the language model is used, and why that way

| Decision | Reason |
|---|---|
| **The model labels; code counts.** | Models are unreliable at counting thousands of items and cannot show their working. Labels can be checked one by one; totals then follow exactly. |
| **Locked category list** (adapted from the reference project Pp2k05/ZOMATO-ECOSYSTEM-ANALYSIS) | Free-form themes drift between runs, which makes month-on-month trends meaningless. |
| **Small, fast model for labelling; stronger model only for writing insights** | Labelling 5,000 reviews is high-volume and simple; insight writing is low-volume and needs reasoning. |
| **Batches of 25, cached by review text** | Cuts cost; identical texts are labelled once; a daily refresh only pays for new reviews. |
| **Text beats star rating** | About 1 in 70 reviews have a rating that contradicts the text; these are flagged. |
| **Personal details removed before labelling** | Phone numbers and emails never leave the machine. |

## 8. Evaluation (how we know it works)

The answer key is read only by `src/evaluate.py`. The labeller never sees it.

| What we measure | How | Target |
|---|---|---|
| Noise removal | Precision, recall, F1 against answer key | F1 ≥ 0.90 |
| Duplicate merging | Is each true duplicate grouped with its original? | Recall ≥ 0.85, precision ≥ 0.80 |
| Personal details removed | Recall | 100% |
| Sentiment | Accuracy and macro F1 on a 70% **holdout** split | ≥ 85% / ≥ 0.80 |
| Issue labels | Primary issue correct; multi-label micro F1 | ≥ 80% / ≥ 0.75 |
| Robustness | Sentiment accuracy on sarcastic reviews, rating-text conflicts, each language | ≥ 75% |
| **Dashboard numbers** | Copilot's monthly negative share and issue shares compared with the same numbers computed from the answer key | Within 2 points |
| **Incident detection** | Does the trend engine flag the 4 planted incidents in the right month? | 4 of 4 |
| **False alarms** | A control release with no planted change must show no clear change | No signal |
| Citation accuracy | Do cited reviews exist and really support the claim? | 100% exist; ≥ 90% support (human-rated) |
| Insight usefulness | 5-point rubric: accuracy, evidence, specificity, actionability, honesty | Average ≥ 4 |

**Answer key audit.** Repeated copies of the same sentence should carry the same label. The audit found **147 rows across 10 Hinglish sentences** where the answer key gives contradictory sentiment. One example: "Khana basi tha, pet kharab ho gaya" ("the food was stale and gave me an upset stomach") is labelled Positive 22 times and Negative 15 times. Scores are reported with and without these rows. Before any final claims, the dataset generator should be fixed.

### Results so far (rule-based baseline; Gemini run pending)

| Metric | Baseline score |
|---|---|
| Noise removal F1 | 0.99 |
| Duplicate recall / precision | 0.89 / 0.91 |
| Personal details removed | 100% |
| Platform / date month correct | 100% / 96% |
| Sentiment accuracy (holdout) | 87.2% (88.6% excluding contradictory answer-key rows) |
| Primary issue correct (holdout) | 75.1% |
| Planted incidents detected | 4 of 4 |
| Control release false alarm | None |
| Monthly negative share error | 1.9 points |
| Top-5 problems matching the truth | 2 of 5 ← the main reason a language model is needed |

## 9. Failure modes and guardrails

| Failure mode | Guardrail | Fallback |
|---|---|---|
| Model invents numbers | Model never counts; headline numbers checked against the evidence pack | Code-written headline shown; flag on screen |
| Model invents a quote | Quote must appear word for word in the review | Quote dropped, row marked unverified |
| Model cites a review that does not exist | Every record ID checked | Citation removed; hypothesis flagged if fewer than 2 remain |
| Model skips records in a batch | Missing IDs re-asked in a smaller batch | Rules classifier, marked `rules_fallback` |
| Gemini outage or quota limit | Retries with backoff, request pacing | Whole run continues on rules; the dashboard says so |
| Sarcasm misread | Explicit prompt rule | Measured separately in the Evaluation Centre |
| Volume growth looks like a worse product | All trends are shares of feedback | Spikes need lift ≥ 1.6×, at least 12 complaints, p ≤ 0.01 |
| Mix effects: one surge makes other issues' shares fall | Called out in release analysis | Read alongside raw counts |
| Correlation read as causation | Release analysis labelled "association", with written caveat | Product manager judgement |
| Over-trusting low-evidence issues | Evidence level shown on every problem (Low under 20 reviews) | Low-evidence items marked "signal, not conclusion" |

## 10. Success metrics

**User value:** time from export to a prioritised problem list (target: under 15 minutes, compared with days); share of weekly product decisions that cite the tool; product managers' rating of "I trust these numbers" (target ≥ 4 of 5).

**Business value (in a real deployment):** time to detect a release-related problem (target: within 1 week of release); decline in the share of feedback about an issue after a shipped fix.

**AI quality:** the evaluation targets in section 8, tracked on every prompt change (the prompt version is stored with each cached label).

## 11. Real-user validation plan

Put the working demo in front of **two founders or product managers**. The script is in `docs/user_validation_kit.md`. We ask:

- Does it help you prioritise better?
- Are the insights useful?
- Which outputs do you not trust?
- What confused you?
- What would you improve?

Findings go into a change log in that document.

## 12. Decision log

| Date | Decision | Why |
|---|---|---|
| 12 Nov 2026 | Model labels, code counts | Checkable numbers; the central reliability choice |
| 12 Nov 2026 | Kept the reference project's two-level categories, adding Gold and feature request issues | Comparable with prior work; Gold is a known pain point |
| 12 Nov 2026 | "Switching to Swiggy" is a competitor comparison, not noise | It is churn signal about Zomato |
| 12 Nov 2026 | Same text from different users is not a duplicate; placeholder names like "A Google user" never link reviews | Otherwise real volume is undercounted |
| 12 Nov 2026 | Relative dates ("3 weeks ago") estimated from the export date and labelled as estimates | Keeps them in trends without pretending they are exact |
| 12 Nov 2026 | Report holdout scores; audit the answer key | Avoids grading our own homework |
