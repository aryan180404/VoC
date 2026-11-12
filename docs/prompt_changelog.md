# Prompt changelog

Prompts live in `src/enrich.py` (labelling) and `src/insights.py` (insights, ask the data).
Each cached label stores the prompt version, so changing a prompt re-labels cleanly.

## tagging-v4 (12 November 2026), not yet measured on Gemini

| Weakness in v3 | Change in v4 |
|---|---|
| No rule for export wrappers. "Q: How do I get a refund? A: food was great" could be labelled as a refund complaint | Step 1: label only the customer's own words; ignore question text, video titles, subreddit tags |
| Filler rule listed exact phrases from this dataset (overfitting) | General rule: an issue needs a concrete detail |
| No examples | 9 invented examples covering wrappers, sarcasm, filler, Hinglish, tacked-on requests, mixed reviews, competitor, spam. None are copied from the dataset, so they cannot leak into the evaluation |
| "Mixed reviews: pick the main point" left balanced reviews undefined | Balanced mix is Neutral; a dominant side wins |
| Confusable issues had no boundaries (late vs wrong tracking, fee vs price vs promotion, agent vs bot vs refund) | Explicit boundary list |
| "Switching to Swiggy" at the end of a complaint could become the main issue | Label the complaint; set churn risk |
| Confidence had no scale | Four anchor points |
| Reviews could carry instructions to the model | Review text declared untrusted data |
| Labels were decided before evidence | Answer format reordered: evidence quote first, then issues, then sentiment |
| Brand name hard-coded | Taken from settings |

## Insight prompt

| Weakness | Change |
|---|---|
| The "good" example contained made-up numbers the model might imitate | Example removed; headline contents specified instead |
| No limits on output | 2 to 4 hypotheses, 2 to 4 data checks, 3 to 5 interview questions |
| Hypotheses could restate the complaint | Must be a cause inside the company's control, distinct, with 2 citations and confirming data |
| Interview questions could be leading | Must ask about past behaviour, not opinions on solutions |
| Causal language not restricted | "may" or "could" only; release timing is coincidence, not cause |
| Only the headline's numbers were checked | Contradicting-evidence text is checked too; hypothesis count is checked |

## Ask-the-data prompt

Was two sentences. Now: answer first; counts only from code's statistics (the reviews shown are a sample); a citation for every claim; interpretation marked as such; says when the evidence is insufficient; confidence defined.

## How to measure a prompt change
Run `python run_pipeline.py`, then compare the Gemini row in the Evaluation Centre before and after. Keep the change only if holdout scores do not drop.
