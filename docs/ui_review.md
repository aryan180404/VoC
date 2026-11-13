# Interface review

Checked against Vercel's Web Interface Guidelines (fetched 13 November 2026) and against design notes for
product tools in the Awesome Design collection (PostHog, Linear, Intercom, Stripe, Sentry). Rules that only
apply to React code (hydration, `onPaste`, virtualised lists, image sizes) do not apply to this Streamlit app.

## Findings

### app.py
- app.py:74 decorative status dots missing `aria-hidden="true"`
- app.py:78 page not in the URL; a shared link always opens Overview (guideline: URL reflects state)
- app.py:244 "..." → "…" in loading text
- app.py:403 "..." → "…" in loading text
- app.py:292 placeholder should show an example and end with "…"
- app.py:400 placeholder should end with "…"
- app.py:chat_input placeholder should end with "…"
- app.py:252 "Clear conversation" deletes the whole chat at once; needs a confirmation step
- app.py:249 chat error states the problem but not the next step
- Headings and buttons use sentence case, not Title Case. Kept on purpose: it is the house style chosen for
  a calm internal tool, and it is applied consistently.

### src/ui.py
- src/ui.py:160 page and section titles are `<div>`s with no heading role; screen readers see no structure
- src/ui.py:65 headings missing `text-wrap: balance`
- tabular figures only on the headline numbers; missing on panels, tables, badges and release tiles
- no `prefers-reduced-motion` rule
- no `touch-action: manipulation` or intentional tap highlight on controls
- focus ring defined for buttons and the menu only; missing on expanders, links and chips
- review text and long issue names have no wrapping rule; a long unbroken string can overflow
- Streamlit notice boxes all look the same; information, warning and error need distinct but quiet styles
- nothing marks which text Gemini wrote versus what code computed

## Ideas taken from the reference designs (principles only, no brand styling copied)
- One accent colour used sparingly: primary button, focus ring, links, selection (Linear, PostHog).
- Hierarchy from a light surface ladder and hairline borders, not shadows (Linear, Intercom, PostHog).
- Tabular figures in every numeric cell (Stripe).
- Slight negative tracking on large titles only; body text at normal spacing (Linear, Intercom, Stripe).
- Quiet tinted notice boxes with a side rail, one tint per meaning (PostHog's documentation callouts).
- AI output gets its own restrained marker, used only on AI content and never as decoration (Intercom's
  separate colour for its AI product).
- Line height 1.5 on working surfaces (Sentry).
