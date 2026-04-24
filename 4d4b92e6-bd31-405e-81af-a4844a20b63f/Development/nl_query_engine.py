
import re

# ─────────────────────────────────────────────
# Natural Language Query Engine for Real Estate
# Parses: neighborhood, intent, risk_level
# ─────────────────────────────────────────────

# Known neighborhoods (from dataset)
NEIGHBORHOODS = [
    "Downtown", "Midtown", "Uptown", "Suburbs", "Waterfront"
]

# Intent patterns: action → canonical intent
INTENT_PATTERNS = {
    "buy": [
        r"\bbuy\b", r"\bpurchase\b", r"\bacquire\b", r"\bget a home\b",
        r"\bget a house\b", r"\bshould i buy\b", r"\bbest time to buy\b"
    ],
    "sell": [
        r"\bsell\b", r"\blisting\b", r"\bput.*on the market\b",
        r"\boffload\b", r"\bshould i sell\b", r"\btime to sell\b"
    ],
    "invest": [
        r"\binvest\b", r"\binvestment\b", r"\brental\b", r"\brent out\b",
        r"\bcash flow\b", r"\broi\b", r"\bpassive income\b", r"\bflip\b",
        r"\bflipping\b", r"\bappreciation\b"
    ],
    "compare": [
        r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bbetter\b",
        r"\bwhich is\b", r"\bwhich neighborhood\b", r"\bdifference between\b",
        r"\brank\b", r"\bbest neighborhood\b"
    ],
}

# Risk level signals
RISK_HIGH = [
    r"\bhigh.risk\b", r"\baggressive\b", r"\bspeculat\b", r"\bflip\b",
    r"\bflipping\b", r"\bmaximize return\b", r"\bmax roi\b", r"\bleverage\b",
    r"\bundervalued\b", r"\bgamble\b"
]
RISK_LOW = [
    r"\bsafe\b", r"\bstable\b", r"\blow.risk\b", r"\bconservative\b",
    r"\blong.term\b", r"\bsecure\b", r"\bfamily\b", r"\bretirement\b",
    r"\bpassive\b", r"\bsteady\b"
]


def extract_neighborhood(text: str) -> str | None:
    """Match a neighborhood name (case-insensitive) from the query."""
    for nbhd in NEIGHBORHOODS:
        if re.search(rf"\b{re.escape(nbhd)}\b", text, re.IGNORECASE):
            return nbhd
    return None


def extract_intent(text: str) -> str:
    """Return the canonical intent with the most pattern matches."""
    scores = {intent: 0 for intent in INTENT_PATTERNS}
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                scores[intent] += 1
    best_intent = max(scores, key=lambda k: scores[k])
    # If no patterns matched at all, default to 'buy' (most common)
    return best_intent if scores[best_intent] > 0 else "buy"


def extract_risk(text: str) -> str:
    """Classify risk preference as high / low / moderate."""
    high_hits = sum(1 for p in RISK_HIGH if re.search(p, text, re.IGNORECASE))
    low_hits  = sum(1 for p in RISK_LOW  if re.search(p, text, re.IGNORECASE))
    if high_hits > low_hits:
        return "high"
    if low_hits > high_hits:
        return "low"
    return "moderate"


def parse_query(user_query: str) -> dict:
    """
    Main parser: extract neighborhood, intent, and risk_level from free text.

    Parameters
    ----------
    user_query : str  – natural language real-estate question

    Returns
    -------
    parsed_query : dict with keys:
        neighborhood  – matched neighborhood name or None
        intent        – buy | sell | invest | compare
        risk_level    – low | moderate | high
        raw_query     – original text (for transparency)
    """
    return {
        "neighborhood": extract_neighborhood(user_query),
        "intent":       extract_intent(user_query),
        "risk_level":   extract_risk(user_query),
        "raw_query":    user_query,
    }


# ─────────────────────────────────────────────
# Test suite – 5 diverse sample queries
# ─────────────────────────────────────────────

sample_queries = [
    "Should I buy a home in Downtown right now?",
    "Is it a good time to sell my Waterfront property before rates go up?",
    "I want to invest in a rental in the Suburbs for stable passive income.",
    "Can you compare Midtown vs Uptown — which is the better neighborhood for flipping houses?",
    "What's the safest long-term neighborhood for a family? I'm thinking Uptown.",
]

DIVIDER_NL = "─" * 60
print(DIVIDER_NL)
print("  🏠  Natural Language Real-Estate Query Engine")
print(DIVIDER_NL)

parsed_results = []

for i, q in enumerate(sample_queries, 1):
    result = parse_query(q)
    parsed_results.append(result)

    print(f"\nQuery {i}: \"{q}\"")
    print(f"  neighborhood : {result['neighborhood'] or '(not specified)'}")
    print(f"  intent       : {result['intent']}")
    print(f"  risk_level   : {result['risk_level']}")

print(f"\n{DIVIDER_NL}")
print(f"  ✅  All {len(sample_queries)} queries parsed successfully.")
print(DIVIDER_NL)

# Expose the last single-query result as parsed_query (as per ticket spec)
user_query = sample_queries[0]
parsed_query = parse_query(user_query)

print(f"\n📋 parsed_query (from ticket spec query):")
for k, v in parsed_query.items():
    print(f"  {k}: {v}")
