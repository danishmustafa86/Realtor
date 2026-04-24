
# ─────────────────────────────────────────────────────────────────────────────
# NL REPORT — Full Natural Language Results Experience
# Readable by anyone. No jargon. Zero formulas. 100% human.
# ─────────────────────────────────────────────────────────────────────────────
# Upstream:
#   boss_agent         → recommendations_df, market_heat_index, macro_data
#   nl_query_engine    → parsed_query
#   neighborhood_rankings → neighborhood_rankings

import pandas as pd

# ── 1. Pull the queried neighborhood & recommendation ─────────────────────────
_nlr_queried_nbhd = parsed_query.get("neighborhood") or recommendations_df.iloc[0]["neighborhood"]
_nlr_intent       = parsed_query.get("intent", "buy")
_nlr_risk         = parsed_query.get("risk_level", "moderate")
_nlr_raw_query    = parsed_query.get("raw_query", "")

_nlr_nbhd_row = recommendations_df[recommendations_df["neighborhood"] == _nlr_queried_nbhd]
if _nlr_nbhd_row.empty:
    _nlr_nbhd_row = recommendations_df.iloc[[0]]
    _nlr_queried_nbhd = recommendations_df.iloc[0]["neighborhood"]

_nlr_rec  = _nlr_nbhd_row.iloc[0]["recommendation"]
_nlr_heat = float(market_heat_index)

# ── 2. Emoji indicator for recommendation ─────────────────────────────────────
_nlr_rec_emoji = {
    "Strong Buy": "🟢",
    "Buy":        "🟢",
    "Hold":       "🟡",
    "Sell":       "🔴",
}.get(_nlr_rec, "⚪")

# ── 3. Market condition (plain English) ───────────────────────────────────────
if _nlr_heat >= 8:
    _nlr_market_summary = (
        "Right now the market is running very hot. Homes are selling fast and "
        "prices are high — a great time to sell, but a tough one for buyers."
    )
elif _nlr_heat >= 7:
    _nlr_market_summary = (
        "Right now the market is on the warmer side, leaning toward sellers. "
        "There's strong demand out there, which means prices stay high and "
        "good deals can be harder to find."
    )
elif _nlr_heat >= 4:
    _nlr_market_summary = (
        "The market is pretty balanced at the moment. Neither buyers nor sellers "
        "have a huge upper hand, which often means more room to negotiate."
    )
else:
    _nlr_market_summary = (
        "The market is on the cooler side right now — less competition and more "
        "room to get a good price, which works in a buyer's favor."
    )

# ── 4. Latest macro snapshot (plain English, no formulas) ────────────────────
_nlr_latest   = macro_data.sort_values("date").iloc[-1]
_nlr_mortgage = float(_nlr_latest["MORTGAGE30US"])
_nlr_houst    = float(_nlr_latest["HOUST"])
_nlr_cpi_yoy  = float(_nlr_latest["CPI_YOY"]) if not pd.isna(_nlr_latest["CPI_YOY"]) else 0.0

if _nlr_mortgage <= 4.0:
    _nlr_mort_plain = (
        f"Mortgage rates are sitting at around {_nlr_mortgage:.1f}% — "
        "historically low, so borrowing money is cheap right now."
    )
elif _nlr_mortgage <= 6.5:
    _nlr_mort_plain = (
        f"Mortgage rates are around {_nlr_mortgage:.1f}% — moderate, "
        "not the cheapest we've seen, but manageable for most buyers."
    )
else:
    _nlr_mort_plain = (
        f"Mortgage rates are elevated at around {_nlr_mortgage:.1f}%, "
        "meaning monthly payments are noticeably higher than a few years ago."
    )

if _nlr_houst >= 1600:
    _nlr_houst_plain = (
        "Builders are putting up new homes at a solid pace, "
        "so supply is gradually improving."
    )
elif _nlr_houst >= 1200:
    _nlr_houst_plain = (
        "New home construction is ticking along at a moderate rate — "
        "not a flood of new supply, but not a drought either."
    )
else:
    _nlr_houst_plain = (
        "New home construction is running below normal, meaning there "
        "aren't many fresh listings hitting the market."
    )

# ── 5. Top 3 neighborhood comparisons (plain English) ────────────────────────
_nlr_ranked   = neighborhood_rankings.sort_values("neighborhood_rank").reset_index(drop=True)
_nlr_all_ppsf = _nlr_ranked["avg_price_per_sqft"].values

def _nlr_describe_nbhd(idx: int) -> str:
    _r    = _nlr_ranked.iloc[idx]
    _n    = _r["neighborhood"]
    _ppsf = _r["avg_price_per_sqft"]
    _dens = _r["bargain_density"]
    _rank = int(_r["neighborhood_rank"])
    _pct  = (_ppsf - _nlr_all_ppsf.min()) / max(_nlr_all_ppsf.max() - _nlr_all_ppsf.min(), 1)
    _ptag = (
        "one of the pricier spots" if _pct >= 0.7
        else "one of the more affordable options" if _pct <= 0.3
        else "mid-range in terms of price"
    )
    _dtag = (
        "and it has the most hidden-value deals right now" if _dens >= 0.22
        else "with a decent number of good-value listings available" if _dens >= 0.15
        else "though bargains are a bit harder to find there"
    )
    return f"#{_rank} {_n} is {_ptag} {_dtag}."

_nlr_top3 = [_nlr_describe_nbhd(i) for i in range(min(3, len(_nlr_ranked)))]

# ── 6. Confidence statement (based on boss_agent recommendations count) ───────
# We use the spread of recommendations as a proxy for confidence.
# If all neighborhoods get the same clear signal → higher confidence.
_nlr_recs       = recommendations_df["recommendation"].value_counts()
_nlr_top_signal = _nlr_recs.index[0]
_nlr_top_count  = int(_nlr_recs.iloc[0])
_nlr_total_nbhd = len(recommendations_df)
_nlr_agreement  = _nlr_top_count / _nlr_total_nbhd  # 1.0 = perfect agreement

if _nlr_agreement >= 0.8:
    _nlr_conf_level = "fairly confident"
    _nlr_conf_emoji = "🟢"
    _nlr_conf_detail = (
        f"{_nlr_top_count} out of {_nlr_total_nbhd} neighborhoods point to "
        f"the same signal ({_nlr_top_signal}), which shows strong consistency "
        "across the market — not just one neighborhood, but a broad trend."
    )
elif _nlr_agreement >= 0.5:
    _nlr_conf_level = "moderately confident"
    _nlr_conf_emoji = "🟡"
    _nlr_conf_detail = (
        f"{_nlr_top_count} out of {_nlr_total_nbhd} neighborhoods agree on "
        f"{_nlr_top_signal}. There's a clear leaning, but some parts of the "
        "market tell a slightly different story."
    )
else:
    _nlr_conf_level = "cautiously optimistic"
    _nlr_conf_emoji = "🟡"
    _nlr_conf_detail = (
        "The neighborhoods we tracked show mixed signals, so we'd encourage "
        "you to treat this as a starting point for your own research."
    )

# ── 7. Recommendation in plain English ────────────────────────────────────────
if _nlr_rec in ("Strong Buy", "Buy"):
    _nlr_rec_plain = (
        f"We think {_nlr_queried_nbhd} is worth buying into right now. "
        "The fundamentals look solid and the timing works in your favor."
    )
elif _nlr_rec == "Hold":
    _nlr_rec_plain = (
        f"If you already own in {_nlr_queried_nbhd}, we'd say hold tight for now. "
        "It's not the moment to rush into buying or selling — patience will pay off."
    )
else:  # Sell
    _nlr_rec_plain = (
        f"Our analysis suggests that right now may be a good time to sell "
        f"in {_nlr_queried_nbhd}. The market is warm and prices are about as "
        "good as they're likely to get in the near term."
    )

_nlr_risk_note = {
    "high":     "Given you're open to higher risk, this is a moment where bold moves can pay off.",
    "low":      "Since you prefer playing it safe, this aligns with a cautious, steady approach.",
    "moderate": "For someone with a balanced risk appetite like yours, this fits comfortably.",
}.get(_nlr_risk, "")

# ── 8. Box-drawing helpers ────────────────────────────────────────────────────
_NLR_W   = 70
_NLR_BOX = "╔" + "═" * (_NLR_W - 2) + "╗"
_NLR_BOT = "╚" + "═" * (_NLR_W - 2) + "╝"
_NLR_SEP = "╟" + "─" * (_NLR_W - 2) + "╢"
_NLR_PAD = "║"

def _nlr_wrap(text: str, width: int = _NLR_W - 4) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]

def _nlr_box_lines(texts: list) -> None:
    for text in texts:
        for wl in _nlr_wrap(text):
            print(f"{_NLR_PAD}  {wl:<{_NLR_W - 4}}{_NLR_PAD}")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT THE FULL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print()
print(_NLR_BOX)
print(f"{_NLR_PAD}{'':^{_NLR_W - 2}}{_NLR_PAD}")
print(f"{_NLR_PAD}{'🏠  YOUR REAL ESTATE ADVISOR REPORT':^{_NLR_W - 2}}{_NLR_PAD}")
print(f"{_NLR_PAD}{'':^{_NLR_W - 2}}{_NLR_PAD}")
print(_NLR_BOT)
print()

# ── Section 1: What You Asked ─────────────────────────────────────────────────
print("📋  WHAT YOU ASKED")
print("─" * _NLR_W)
print(f"    You asked: \"{_nlr_raw_query}\"")
print(f"    We looked at {_nlr_queried_nbhd} for you, with a focus on your goal to {_nlr_intent}.")
print()

# ── Section 2: Our Recommendation (boxed) ────────────────────────────────────
print(_NLR_BOX)
_nlr_hdr = f"  {_nlr_rec_emoji}  OUR RECOMMENDATION FOR {_nlr_queried_nbhd.upper()}: {_nlr_rec.upper()}  "
print(f"{_NLR_PAD}{_nlr_hdr:^{_NLR_W - 2}}{_NLR_PAD}")
print(_NLR_SEP)
print(f"{_NLR_PAD}{'':^{_NLR_W - 2}}{_NLR_PAD}")
_nlr_box_lines([_nlr_rec_plain, "", _nlr_risk_note])
print(f"{_NLR_PAD}{'':^{_NLR_W - 2}}{_NLR_PAD}")
print(_NLR_BOT)
print()

# ── Section 3: Market Summary (3 plain sentences, no math) ───────────────────
print("📈  WHAT'S HAPPENING IN THE MARKET RIGHT NOW")
print("─" * _NLR_W)
print(f"  • {_nlr_market_summary}")
print(f"  • {_nlr_mort_plain}")
print(f"  • {_nlr_houst_plain}")
print()

# ── Section 4: Top 3 Neighborhood Comparisons ────────────────────────────────
print("🏘️  HOW THE TOP 3 NEIGHBORHOODS STACK UP")
print("─" * _NLR_W)
for _nlr_line in _nlr_top3:
    print(f"  • {_nlr_line}")
print()

# ── Section 5: Confidence Statement ──────────────────────────────────────────
print("🎯  HOW CONFIDENT ARE WE?")
print("─" * _NLR_W)
print(f"  {_nlr_conf_emoji}  We are {_nlr_conf_level} in this recommendation.")
_nlr_box_lines([f"     {_nlr_conf_detail}"])
print()
print(f"     Real estate is complex — treat this as a helpful guide,")
print(f"     not a guarantee. Consider chatting with a local agent too.")
print()
print("═" * _NLR_W)
print("  ✅  Report complete. Good luck with your real estate journey! 🏡")
print("═" * _NLR_W)
print()

# ── Output variable ───────────────────────────────────────────────────────────
final_nl_report = (
    f"REAL ESTATE ADVISOR REPORT\n\n"
    f"You asked: {_nlr_raw_query}\n"
    f"Neighborhood: {_nlr_queried_nbhd} | Intent: {_nlr_intent} | Risk: {_nlr_risk}\n\n"
    f"RECOMMENDATION: {_nlr_rec_emoji} {_nlr_rec.upper()}\n"
    f"{_nlr_rec_plain}\n"
    f"{_nlr_risk_note}\n\n"
    f"MARKET SNAPSHOT\n"
    f"• {_nlr_market_summary}\n"
    f"• {_nlr_mort_plain}\n"
    f"• {_nlr_houst_plain}\n\n"
    f"TOP 3 NEIGHBORHOODS\n"
    + "".join(f"• {_l}\n" for _l in _nlr_top3)
    + f"\nCONFIDENCE\n"
    f"{_nlr_conf_emoji} We are {_nlr_conf_level} in this recommendation.\n"
    f"{_nlr_conf_detail}"
)
