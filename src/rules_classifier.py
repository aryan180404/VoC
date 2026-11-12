"""Rule-based fallback classifier.

Why this exists (a product decision, not a shortcut):
  1. Guardrail: if Gemini is down, rate-limited or returns invalid output for a
     batch, those records still get a label, clearly marked label_source="rules".
  2. Baseline: the evaluation compares Gemini against this, so we can show what
     the language model actually adds instead of assuming it is better.
It is deliberately simple keyword matching. It does not understand sarcasm well
and it does not read Tamil, Telugu, Kannada or Bengali beyond a few words.
"""
from __future__ import annotations

import re

from .taxonomy import L2_TO_L1

# Each issue: list of regular expressions (lower-case text). Kept domain-generic.
ISSUE_PATTERNS: dict[str, list[str]] = {
    "Speed of Delivery": [r"\b\d{2,3} ?(mins?|minutes?)\b(?! ?(late|without penalty))", r"\d+ minute late", r"stuck for \d+",
                          r"delivered in \d+", r"on time", r"took (an|one|two|\d+) hours?", r"got it at \d", r"came after \d+",
                          r"delivery (was )?(very |so )?(late|slow|fast|quick)", r"late delivery", r"देर से", r"जल्दी", r"लेट",
                          r"ತಡವಾಗಿ", r"उशिरा", r"waiting an hour", r"never had a late order", r"fast delivery"],
    "Order Tracking Accuracy": [r"tracking", r"marked (as )?delivered", r"on the map", r"'on the way'", r"status (jumped|shows|showed)",
                                r"estimated time", r"eta\b", r"shown near"],
    "Staff Professionalism": [r"delivery (partner|guy|boy|executive)", r"\brider\b(?!.*lost)", r"shouted", r"rude", r"polite",
                              r"mark the order as delivered", r"asked (me )?for (extra|tip)", r"upar aane"],
    "Location & City Availability": [r"not delivering to your location", r"address pin", r"serve my (town|city|area)", r"within \d+ ?km",
                                     r"outside (the )?\d+ ?km", r"not available in my", r"distance"],
    "Restaurant Acceptance Issues": [r"restaurant (cancelled|canceled|accepted)", r"cancelled my order after accepting",
                                     r"item not available", r"show them as open"],
    "Order Accuracy (Missing/Wrong)": [r"missing", r"wrong (item|order|quantity)", r"received 1|recieved 1", r"something completely different",
                                       r"non-veg item", r"kuch aur hi aa gaya", r"got something else"],
    "Food Quality (Stale/Cold)": [r"\bcold\b", r"stale", r"fresh and hot", r"food quality", r"tasteless", r"soggy", r"spoiled", r"thanda",
                                  r"\bbasi\b", r"ठंडा", r"ঠান্ডা", r"டேஸ்ட்", r"food (was|is) (great|amazing|delicious|bad|awful)"],
    "Food Safety (Foreign Object)": [r"\bhair\b", r"insect", r"cockroach", r"plastic piece", r"food poisoning", r"foreign object",
                                     r"pet kharab", r"health issue"],
    "Item Condition & Packaging": [r"packag(ing|e)", r"leaked|leaking|spilled", r"smashed|crushed", r"seal", r"প্যাকেজিং"],
    "Fees & Surcharges": [r"platform fee", r"rain fee", r"packaging fee", r"delivery fee", r"surge", r"convenience fee",
                          r"charges hai", r"naya charge", r"small order fee"],
    "Price & Affordability": [r"higher than dine-in", r"menu prices", r"expensive", r"costs ₹", r"luxury", r"in the restaurant and ₹",
                              r"prices on the app", r"overpriced"],
    "Promotions & Loyalty": [r"offers?\b", r"coupon(?!.*crash)", r"discount", r"cashback", r"saved ₹", r"loyal", r"gold is (totally )?worth",
                             r"ఆఫర్స్"],
    "Gold Hidden Fees/Platform Charges": [r"auto[- ]?renew", r"renewed automatically", r"gold.{0,40}(fee|charge|debited|₹)",
                                          r"gold says free delivery", r"gold.{0,30}only within"],
    "App Stability & Bugs": [r"crash", r"stuck on the splash", r"\bbug", r"freez", r"cart got cleared", r"app (is )?broken",
                             r"version \d+\.\d+", r"error", r"not loading", r"hangs"],
    "Checkout & Payment Process": [r"payment (failed|was charged|keeps)", r"money (got )?debited", r"charged twice", r"\bupi\b",
                                   r"cash on delivery", r"checkout"],
    "App/Website Navigation": [r"\bui\b", r"search bar", r"filters?\b", r"push notifications|notifications (are|about)", r"easy app", r"easy to use", r"veg mode",
                               r"interface", r"navigation", r"schedule order", r"design"],
    "Account & Login": [r"\botp\b", r"log ?in", r"account (got )?(blocked|hacked|suspended)", r"sign ?in"],
    "Agent Helpfulness": [r"\bagents?\b", r"support (agent|executive|team)", r"customer care wale", r"support se baat",
                          r"good support", r"great support", r"copy pasted", r"we understand your concern", r"scripted", r"real human"],
    "Self-Service & Bot Effectiveness": [r"\bbot\b", r"chatbot", r"help section", r"no customer care number", r"closed my ticket",
                                         r"cannot escalate", r"sirf bot"],
    "Refund & Policy Disputes": [r"refund(?!.*(sirf|only) ₹)", r"refnd", r"wallet", r"compensat", r"cancellation (fee|policy)",
                                 r"पैसे परत", r"रिफंड", r"money back"],
    "Premium/Gold Support Failure": [r"priority support", r"gold member.{0,60}support", r"pay for gold.{0,60}support",
                                     r"support treats me the same"],
    "Competitor Comparisons": [r"switching to swiggy|moving to (ondc|magicpin)|swiggy (refunds|is better|was cheaper)",
                               r"compared the same order", r"swiggy and zomato|zomato (vs|or) swiggy"],
}
_COMPILED = {k: [re.compile(p) for p in v] for k, v in ISSUE_PATTERNS.items()}

FEATURE_REQUEST = re.compile(
    r"please add|can you add|need an option|please allow|please show|i'?d love to see|bring back|would be (great|nice) if|"
    r"wish (there|you|zomato)|should (add|have|allow)|add an option|add a filter|feature request|give us an option|let us")

POSITIVE = re.compile(
    r"\b(good|great|awesome|amazing|excellent|love|loved|nice|polite|kind|fresh|hot|impressed|worth it|keep it up|thanks?|"
    r"thank you|useful|easy|best|fast|quick|on time|saved|happy|perfect|mast)\w*|अच्छा|బాగ|நல்லா|❤️|👍|🙏|😍")
NEGATIVE = re.compile(
    r"\b(worst|bad|poor|pathetic|terrible|horrible|disappointed|disappointing|late|cold|stale|missing|wrong|rude|shouted|never|"
    r"crash|broken|failed|debited|hike|hiked|fake|lie|useless|ridiculous|disgusting|uninstall|fed up|unacceptable|"
    r"shady|robbery|scam|annoying|refund|stuck|not|no|cannot|can't|bakwas|bekaar|basi|kharab|nahi|khraab)\w*|😡|😤|😭|🤬|"
    r"देर|खराब|नहीं|ঠান্ডা|খারাপ|ತಡವಾಗಿ|उशिरा|नाहीत|லேட்|🤡")
SARCASM = re.compile(
    r"great job zomato 👏|👏👏|🙃|truly world class|love paying|thanks for (nothing|making)|10/10 customer service|"
    r"wow,? amazing service|amazing, the bot|what a (great|wonderful)")
NEUTRAL_HINT = re.compile(r"\b(okay|ok|average|fine|nothing special|decent|so-so|sometimes good|theek)\b")


def classify(text: str, rating: float | None = None) -> dict:
    t = text.lower()
    issues = [l2 for l2, pats in _COMPILED.items() if any(p.search(t) for p in pats)]
    # Gold support beats generic agent or bot labels; Gold fees beat generic fees.
    if "Premium/Gold Support Failure" in issues:
        issues = [i for i in issues if i not in ("Agent Helpfulness", "Self-Service & Bot Effectiveness")]
    if "Gold Hidden Fees/Platform Charges" in issues:
        issues = [i for i in issues if i not in ("Fees & Surcharges", "Refund & Policy Disputes", "Promotions & Loyalty")]
    is_feature_request = bool(FEATURE_REQUEST.search(t))
    if not issues:
        issues = ["Feature Request"] if is_feature_request else ["No Specific Issue Stated"]
    issues = issues[:2]

    pos, neg = len(POSITIVE.findall(t)), len(NEGATIVE.findall(t))
    sarcastic = bool(SARCASM.search(t))
    if sarcastic:
        sentiment = "Negative"
    elif issues == ["Feature Request"]:
        sentiment = "Neutral"
    elif neg > pos:
        sentiment = "Negative"
    elif pos > neg:
        sentiment = "Positive"
    elif NEUTRAL_HINT.search(t):
        sentiment = "Neutral"
    elif rating is not None:
        sentiment = "Negative" if rating <= 2 else "Positive" if rating >= 4 else "Neutral"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "l2_issues": issues,
        "l1_categories": sorted({L2_TO_L1[i] for i in issues}),
        "is_feature_request": is_feature_request,
        "feature_request_text": "",
        "severity": 3 if re.search(r"food poisoning|insect|hair|plastic|debited|charged twice|blocked|unacceptable", t) else 2 if sentiment == "Negative" else 1,
        "churn_risk": bool(re.search(r"uninstall|switching|never (ordering|using)|moving to|deleting", t)),
        "is_sarcastic": sarcastic,
        "evidence_quote": "",
        "confidence": 0.5,
        "label_source": "rules",
    }
