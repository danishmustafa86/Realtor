import pandas as pd
import numpy as np

# Filter properties for this neighborhood slice
_slice_df = properties_df[properties_df["neighborhood"] == neighborhood_slice].copy()

# ── Metrics ──────────────────────────────────────────────────────────────────
_avg_value_score  = round(_slice_df["property_value_score"].mean(), 3)
_undervalued_count = int(_slice_df["is_bargain_outlier"].sum())
_total_listings   = len(_slice_df)
_avg_ppsf         = round(_slice_df["price_per_sqft"].mean(), 2)
_median_ppsf      = round(_slice_df["price_per_sqft"].median(), 2)

# neighborhood_rank computed as composite:
#   higher avg_value_score → higher rank candidate
#   more undervalued properties → more opportunity (slight bonus)
# Score combines normalized value_score and bargain density
_bargain_density  = _undervalued_count / _total_listings  # fraction of bargains
_composite        = round(_avg_value_score + (_bargain_density * 2), 4)  # weighted composite

# Package the slice result as a dict — Aggregator will collect these
slice_result = {
    "neighborhood":        neighborhood_slice,
    "avg_value_score":     _avg_value_score,
    "undervalued_count":   _undervalued_count,
    "total_listings":      _total_listings,
    "avg_price_per_sqft":  _avg_ppsf,
    "median_price_per_sqft": _median_ppsf,
    "bargain_density":     round(_bargain_density, 4),
    "composite_score":     _composite,
    "neighborhood_rank":   None,   # assigned in Aggregator after all slices collected
}

print(f"📍 [{neighborhood_slice}]  avg_score={_avg_value_score}  "
      f"undervalued={_undervalued_count}/{_total_listings}  composite={_composite}")
