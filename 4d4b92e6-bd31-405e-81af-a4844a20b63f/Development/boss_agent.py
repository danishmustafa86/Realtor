import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# BOSS AGENT — Synthesis & Recommendation Engine
# Combines macro signals (Economist) with micro rankings (Fleet Appraiser)
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Inputs from upstream agents ──────────────────────────────────────────
# market_heat_index : float  1–10 (from economist_agent)
# neighborhood_rankings : DataFrame with avg_value_score, composite_score etc.

# Use avg_value_score (0–10 scale from appraiser fleet) as property_value_score
_rankings = neighborhood_rankings.copy()
_rankings = _rankings.rename(columns={"avg_value_score": "property_value_score"})

# ── 2. Recommendation logic ──────────────────────────────────────────────────
def _recommend(heat_idx: float, prop_score: float) -> tuple[str, str]:
    """
    Returns (recommendation, logic_explanation) based on:
      - market_heat_index : macro market temperature (1–10)
      - property_value_score : neighborhood investment quality (0–10)
    
    Decision matrix:
      heat <= 4 AND prop >= 7  → Hold       (cool market, already good assets)
      heat >= 7 AND prop >= 7  → Strong Buy (hot market demand + quality asset)
      heat >= 7 AND prop < 7   → Sell       (hot market, below-average quality)
      heat < 7 AND prop >= 7   → Buy        (warm market, quality assets available)
      heat < 7 AND prop < 7    → Sell       (weak fundamentals, no upside catalyst)
      intermediate (4–7 heat)  → Buy/Hold blend
    """
    h = heat_idx
    p = prop_score

    if h <= 4 and p >= 7:
        rec  = "Hold"
        expl = (f"Cool market (heat={h:.1f}/10) limits near-term appreciation, "
                f"but strong asset quality (score={p:.2f}) supports retention. "
                "Await a more active market before deploying capital.")

    elif h >= 7 and p >= 7:
        rec  = "Strong Buy"
        expl = (f"Hot market (heat={h:.1f}/10) indicates robust demand and rising "
                f"prices. High property quality (score={p:.2f}) amplifies upside. "
                "Favorable risk-reward — act decisively before further appreciation.")

    elif h >= 7 and p < 7:
        rec  = "Sell"
        expl = (f"Hot market (heat={h:.1f}/10) creates a seller's window, but "
                f"below-average asset quality (score={p:.2f}) limits long-term "
                "value. Capitalise on peak pricing before a cyclical correction.")

    elif 4 < h < 7 and p >= 7:
        rec  = "Buy"
        expl = (f"Balanced market (heat={h:.1f}/10) provides measured entry "
                f"conditions. High property quality (score={p:.2f}) indicates "
                "strong fundamentals with room to grow as market heats further.")

    elif 4 < h < 7 and p < 7:
        rec  = "Sell"
        expl = (f"Warm market (heat={h:.1f}/10) but below-average asset quality "
                f"(score={p:.2f}) yields a mediocre risk profile. Rotate capital "
                "into higher-quality opportunities before conditions cool.")

    else:  # h <= 4 and p < 7
        rec  = "Sell"
        expl = (f"Cool market (heat={h:.1f}/10) and weak asset quality "
                f"(score={p:.2f}) — no near-term catalyst for appreciation. "
                "Exit position and reallocate.")

    return rec, expl


# ── 3. Apply logic to every neighborhood ────────────────────────────────────
_results = []
for _, _row in _rankings.iterrows():
    _rec, _expl = _recommend(market_heat_index, _row["property_value_score"])
    _results.append({
        "neighborhood":        _row["neighborhood"],
        "recommendation":      _rec,
        "market_heat_index":   round(float(market_heat_index), 2),
        "property_value_score": round(float(_row["property_value_score"]), 3),
        "composite_score":     round(float(_row["composite_score"]), 4),
        "neighborhood_rank":   int(_row["neighborhood_rank"]),
        "logic_explanation":   _expl,
    })

recommendations_df = pd.DataFrame(_results).sort_values("neighborhood_rank").reset_index(drop=True)

# ── 4. Print recommendations ─────────────────────────────────────────────────
_DIVIDER = "=" * 80
print(_DIVIDER)
print("🤖  BOSS AGENT — INVESTMENT RECOMMENDATIONS")
print(_DIVIDER)
print(f"   📊 Market Heat Index  : {market_heat_index:.2f} / 10")
_heat_label = (
    "🟢 Cool – Buyer-Friendly"  if market_heat_index < 4 else
    "🟡 Warm – Balanced Market" if market_heat_index < 6 else
    "🟠 Hot – Seller's Market"  if market_heat_index < 8 else
    "🔴 Very Hot – High Stress"
)
print(f"   🌡️  Market Condition   : {_heat_label}")
print(f"   🏘️  Neighborhoods      : {len(recommendations_df)}")
print(_DIVIDER)

for _, _r in recommendations_df.iterrows():
    _icon = {"Strong Buy": "🟢", "Buy": "🔵", "Hold": "🟡", "Sell": "🔴"}.get(_r["recommendation"], "⚪")
    print(f"\n  #{int(_r['neighborhood_rank'])}  {_r['neighborhood']}")
    print(f"      {_icon}  Recommendation      : {_r['recommendation']}")
    print(f"         Market Heat Index  : {_r['market_heat_index']}/10")
    print(f"         Property Value     : {_r['property_value_score']}/10")
    print(f"         Composite Score    : {_r['composite_score']}")
    print(f"         Logic: {_r['logic_explanation']}")

print(f"\n{_DIVIDER}")
print(f"✅ Recommendations generated for all {len(recommendations_df)} neighborhoods.")

# Summary tally
_tally = recommendations_df["recommendation"].value_counts()
print("\n📋  Summary:")
for _rec_type, _count in _tally.items():
    _icon = {"Strong Buy": "🟢", "Buy": "🔵", "Hold": "🟡", "Sell": "🔴"}.get(_rec_type, "⚪")
    print(f"   {_icon}  {_rec_type:12s} : {_count} neighborhood(s)")
print(_DIVIDER)
