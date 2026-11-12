"""The locked category list.

Adapted from the reference project (Pp2k05/ZOMATO-ECOSYSTEM-ANALYSIS) and
extended for Gold membership and feature requests. Locking the list is a
product decision: free-form model-invented themes drift between runs and make
month-on-month trends meaningless.
"""
from __future__ import annotations

TAXONOMY: dict[str, dict[str, str]] = {
    "Customer Support & Resolution": {
        "Agent Helpfulness": "Human support agents are unhelpful, scripted, slow to respond, or close tickets without solving the problem.",
        "Self-Service & Bot Effectiveness": "The chatbot or help section loops, gives canned answers, or has no path to a human.",
        "Refund & Policy Disputes": "Refund delayed, denied, partial, or stuck; disputes over cancellation or compensation policy.",
        "Premium/Gold Support Failure": "Paying Gold members say they get no better support than everyone else.",
    },
    "Fulfillment & Logistics": {
        "Speed of Delivery": "Late or slow delivery, long waits, orders stuck.",
        "Order Tracking Accuracy": "Live tracking, estimated time, or rider location is wrong.",
        "Staff Professionalism": "Delivery partner behaviour: rude, refuses to come up, asks to mark delivered early, asks for extra money.",
        "Location & City Availability": "Service not available in an area, wrong address pin, distance cap.",
        "Restaurant Acceptance Issues": "Restaurant cancels after accepting, item unavailable after ordering.",
    },
    "Product & Service Quality": {
        "Order Accuracy (Missing/Wrong)": "Missing items, wrong items, wrong quantity.",
        "Food Quality (Stale/Cold)": "Food stale, cold, bad taste, spoiled (or praised as fresh and hot).",
        "Food Safety (Foreign Object)": "Hair, plastic, insects, or other foreign objects; food poisoning.",
        "Item Condition & Packaging": "Spilled, leaking, crushed, poorly packed (or praised packaging).",
    },
    "Value & Pricing": {
        "Fees & Surcharges": "Platform fee, delivery fee, rain fee, packaging fee, surge charges.",
        "Price & Affordability": "Menu prices higher than in the restaurant, food delivery too expensive overall.",
        "Promotions & Loyalty": "Coupons, offers, discounts, cashback, loyalty rewards (often positive).",
        "Gold Hidden Fees/Platform Charges": "Gold members still charged fees, Gold auto-renewal, Gold benefits removed or hidden.",
    },
    "Ease of Use & Digital Experience": {
        "App Stability & Bugs": "Crashes, freezes, errors, app stuck after an update.",
        "Checkout & Payment Process": "Payment failed, money debited but order not placed, cash on delivery issues.",
        "App/Website Navigation": "Confusing interface, hard to find things, redesign complaints (or praise for an easy app).",
        "Account & Login": "Login, one-time password, account blocked or hacked.",
        "Feature Request": "The customer asks for a new feature or option and no other problem is the main point.",
    },
    "Competitor Comparisons": {
        "Competitor Comparisons": "Compares Zomato with Swiggy, Magicpin, ONDC, or other apps, or says they are switching.",
    },
    "Uncategorized/General Feedback": {
        "No Specific Issue Stated": "General praise or complaint with no specific issue (\"good app\", \"worst experience\").",
    },
}

NOT_RELEVANT = "Not Relevant"

L2_TO_L1: dict[str, str] = {l2: l1 for l1, l2s in TAXONOMY.items() for l2 in l2s}
ALL_L2 = list(L2_TO_L1)
ALL_L1 = list(TAXONOMY)

SENTIMENTS = ["Positive", "Neutral", "Negative"]
NOISE_TYPES = [
    "spam", "brand_reply", "job_post", "off_topic_stock", "other_brand", "junk", "empty",
]
JOURNEY_STAGES = ["Discovery & Ordering", "Payment", "Delivery", "After Delivery & Support", "Membership", "General"]


def taxonomy_for_prompt() -> str:
    lines = []
    for l1, l2s in TAXONOMY.items():
        lines.append(f"## {l1}")
        for l2, desc in l2s.items():
            lines.append(f"- {l2}: {desc}")
    return "\n".join(lines)
