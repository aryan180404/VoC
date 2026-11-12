# Loom demo script (about 4 minutes)

**0:00 The problem (20 s).** "Product managers get feedback from ten places in seven languages. Most AI tools summarise it into confident paragraphs you can't check. I built a copilot where every number is counted by code and every insight links to real reviews."

**0:20 Feedback Hub (30 s).** Show the cleaning funnel: 5,237 raw records. Noise, duplicates and personal details are removed. "Switching to Swiggy" is kept, because it's churn signal, not noise. Open the record explorer to show a raw row next to its cleaned version.

**0:50 Overview (30 s).** "What should the team look at this week?" Point out:
- negative share rising from about 60% to about 68%
- the spike alerts
- the ranked problems, each with the reason for its rank

**1:20 Investigation (60 s).** Open Gold charges.
- Headline with counts and trend
- The reliability panel
- Patterns: open one and read a quote with its record ID
- Hypotheses, labelled as hypotheses, with what would confirm each
- Interview questions and contradicting evidence

Ask one question in "Ask the data".

**2:20 Trends and releases (30 s).** Select the September Gold auto-renewal release and show the before and after. Then select the control release (new checkout flow) and show that it finds nothing. "It doesn't cry wolf."

**2:50 Decision Engine (30 s).**
- Observed data versus my assumptions
- Approve one item and export it for Jira
- Mention the closed loop: once a fix ships, the tool checks whether that complaint's share of feedback falls

**3:20 Evaluation Centre (35 s).**
- Scores on a holdout split
- Gemini against the rules baseline
- Dashboard numbers against the truth
- 4 of 4 planted incidents caught

"I also audited the answer key itself and found 147 rows with contradictory labels."

**3:55 Close (5 s).** "It's a decision-support tool, not an autonomous product manager. The AI investigates; I decide."
