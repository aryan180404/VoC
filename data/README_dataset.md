# Zomato Voice of Customer Dataset (Synthetic, Messy)

**This is synthetic data, written for practice.** No row is a real review or a real person. The layout and categories are modelled on the public project [Pp2k05/ZOMATO-ECOSYSTEM-ANALYSIS](https://github.com/Pp2k05/ZOMATO-ECOSYSTEM-ANALYSIS). That project scraped real reviews from Google Play, the App Store and Reddit (December 2025 to June 2026). This dataset copies none of its review text.

## Files
| File | What it is | Who sees it |
|---|---|---|
| `zomato_voice_of_customer_raw_messy.csv` | 5,237 raw feedback records: the "messy export" a product manager would hand over | Give this to the large language model |
| `zomato_voice_of_customer_answer_key.csv` | The correct answer for every record: sentiment, category, issue, noise, duplicates, true date and platform | Keep hidden. Use it only to score the model's output |

## Sources covered (Source column in every row)
Google Play (about 39%), X/Twitter (about 14%), Reddit (about 14%), Apple App Store (about 11%), YouTube comments (about 9%), Quora (about 5%), consumercomplaints.in (about 4%), MouthShut (about 2%), LinkedIn comments, and news-site comments. Time window: December 2025 to October 2026.

## Built-in patterns your copilot should find
- Negative share rises from about 60% in December 2025 to the high 60s, matching the trend in the reference project.
- Four **invented** incident spikes (not real events):
  - A: app crash on the coupon page after an update, March 2026
  - B: platform fee hike complaints, June 2026
  - C: monsoon delays plus rain fee, July 2026
  - D: Gold membership auto-renewal without notice, September 2026
- Gold complaints: support for paying members, delivery distance cap, hidden fees.
- About 380 feature requests hidden inside complaints.

## Messiness included
- 3 header rows repeated mid-file, plus blank lines and a comment line at the top.
- About 110 exact duplicates and about 80 near-duplicates or cross-posts. These are marked in `duplicate_of`. Identical short texts from different users, such as "Good app", are *not* duplicates.
- Brand written 9 ways, including a typo ("zomto"), a blank brand, and the parent company name.
- Platform written 4 to 7 ways each.
- Dates in about 12 formats: Unix timestamps, Twitter format, US format, "3 weeks ago", and "Mar 14" with no year.
- Ratings as 1, 1.0, 1/5, ★☆☆☆☆, "one", "1 star", or missing. Some ratings contradict their text.
- Typos, ALL CAPS, encoding garbage (â‚¹ in place of ₹), web text codes (&amp;), stray whitespace, line breaks inside reviews, emojis, and sarcasm.
- Hinglish, Hindi, Marathi, Tamil, Telugu, Kannada and Bengali text.
- About 190 noise rows: spam, brand replies, stock-market chatter, Blinkit, District and Swiggy posts, job posts, junk, empty rows.
- About 108 rows with fake personal details to remove: phone numbers such as 9876543210, and @example.com emails.

## Category taxonomy (from the reference project, extended)
**Customer Support & Resolution:** Agent Helpfulness, Self-Service & Bot Effectiveness, Refund & Policy Disputes, Premium/Gold Support Failure

**Fulfillment & Logistics:** Speed of Delivery, Order Tracking Accuracy, Staff Professionalism, Location & City Availability, Restaurant Acceptance Issues

**Product & Service Quality:** Order Accuracy (Missing/Wrong), Food Quality (Stale/Cold), Food Safety (Foreign Object), Item Condition & Packaging

**Value & Pricing:** Fees & Surcharges, Price & Affordability, Promotions & Loyalty, Gold Hidden Fees/Platform Charges

**Ease of Use & Digital Experience:** App Stability & Bugs, Checkout & Payment Process, App/Website Navigation, Account & Login, Feature Request

**Competitor Comparisons**

**Uncategorized/General Feedback:** No Specific Issue Stated
