
import pandas as pd
import numpy as np
import io

# ── 1. Load dataset from upstream variable ───────────────────────────────────
properties_df = pd.read_csv(io.StringIO(real_estate_csv)).copy()

# ── 2. Calculate price_per_sqft per property ─────────────────────────────────
properties_df["price_per_sqft"] = (properties_df["price"] / properties_df["sqft"]).round(2)

# ── 3. Compute neighborhood-level stats ──────────────────────────────────────
nbhd_stats = (
    properties_df
    .groupby("neighborhood")["price_per_sqft"]
    .agg(
        nbhd_mean_ppsf="mean",
        nbhd_std_ppsf="std",
        nbhd_median_ppsf="median",
        nbhd_count="count",
    )
    .reset_index()
)

properties_df = properties_df.merge(nbhd_stats, on="neighborhood", how="left")

# ── 4. Z-score within each neighborhood ──────────────────────────────────────
properties_df["ppsf_zscore"] = (
    (properties_df["price_per_sqft"] - properties_df["nbhd_mean_ppsf"])
    / properties_df["nbhd_std_ppsf"]
).round(4)

# ── 5. Flag outliers: z-score < -1 (significantly below nbhd average) ────────
properties_df["is_bargain_outlier"] = properties_df["ppsf_zscore"] < -1.0

# ── 6. Property Value Score (1–10) ────────────────────────────────────────────
# Score inversely related to z-score relative to neighborhood:
#   - z-score >> 0  → high score (priced above avg = high value)
#   - z-score << 0  → low score  (priced below avg = potential deal / distressed)
# Clamp to [-3, +3] then linearly map to [1, 10]
z_clamped = properties_df["ppsf_zscore"].clip(-3, 3)
properties_df["property_value_score"] = (
    ((z_clamped + 3) / 6 * 9 + 1).round(1)
)
properties_df["property_value_score"] = properties_df["property_value_score"].clip(1, 10)

# ── 7. Neighborhood summary dataframe ────────────────────────────────────────
nbhd_outlier_counts = (
    properties_df
    .groupby("neighborhood")["is_bargain_outlier"]
    .sum()
    .reset_index()
    .rename(columns={"is_bargain_outlier": "bargain_outlier_count"})
)

avg_score = (
    properties_df
    .groupby("neighborhood")["property_value_score"]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={"property_value_score": "avg_value_score"})
)

nbhd_price = (
    properties_df
    .groupby("neighborhood")
    .agg(
        avg_price=("price", "mean"),
        avg_sqft=("sqft", "mean"),
        total_listings=("price", "count"),
    )
    .round(0)
    .reset_index()
)

neighborhood_summary = (
    nbhd_stats
    .merge(nbhd_outlier_counts, on="neighborhood")
    .merge(avg_score, on="neighborhood")
    .merge(nbhd_price, on="neighborhood")
    .sort_values("nbhd_mean_ppsf", ascending=False)
    .reset_index(drop=True)
)

neighborhood_summary["nbhd_mean_ppsf"] = neighborhood_summary["nbhd_mean_ppsf"].round(2)
neighborhood_summary["nbhd_std_ppsf"]  = neighborhood_summary["nbhd_std_ppsf"].round(2)
neighborhood_summary["nbhd_median_ppsf"] = neighborhood_summary["nbhd_median_ppsf"].round(2)

# ── 8. Print results ──────────────────────────────────────────────────────────
print("=" * 70)
print("🏠  APPRAISER AGENT — REAL ESTATE VALUATION REPORT")
print("=" * 70)

print(f"\n📊 Dataset: {len(properties_df)} properties | {properties_df['neighborhood'].nunique()} neighborhoods\n")

print("── NEIGHBORHOOD SUMMARY ─────────────────────────────────────────────")
display_cols = [
    "neighborhood", "total_listings", "avg_price",
    "nbhd_mean_ppsf", "nbhd_median_ppsf", "nbhd_std_ppsf",
    "avg_value_score", "bargain_outlier_count"
]
print(neighborhood_summary[display_cols].rename(columns={
    "neighborhood": "Neighborhood",
    "total_listings": "Listings",
    "avg_price": "Avg Price ($)",
    "nbhd_mean_ppsf": "Mean $/sqft",
    "nbhd_median_ppsf": "Median $/sqft",
    "nbhd_std_ppsf": "Std $/sqft",
    "avg_value_score": "Avg Score (1-10)",
    "bargain_outlier_count": "Bargain Outliers",
}).to_string(index=False))

print(f"\n── BARGAIN OUTLIER PROPERTIES (z-score < -1) ────────────────────────")
outliers = properties_df[properties_df["is_bargain_outlier"]].sort_values("ppsf_zscore")
print(f"   Total flagged: {len(outliers)} properties\n")
print(outliers[[
    "neighborhood", "price", "sqft", "bedrooms", "bathrooms",
    "price_per_sqft", "nbhd_mean_ppsf", "ppsf_zscore", "property_value_score"
]].rename(columns={
    "price_per_sqft": "$/sqft",
    "nbhd_mean_ppsf": "Nbhd Mean $/sqft",
    "ppsf_zscore": "Z-Score",
    "property_value_score": "Value Score",
}).to_string(index=True))

print(f"\n── SCORE DISTRIBUTION ───────────────────────────────────────────────")
score_bins = pd.cut(properties_df["property_value_score"], bins=[0,2,4,6,8,10], labels=["1-2","3-4","5-6","7-8","9-10"])
print(score_bins.value_counts().sort_index().to_string())

print(f"\n── SAMPLE PROPERTIES_DF (first 5 rows) ──────────────────────────────")
print(properties_df[[
    "neighborhood","price","sqft","price_per_sqft",
    "ppsf_zscore","is_bargain_outlier","property_value_score"
]].head(5).to_string(index=True))

print("\n✅ Appraiser Agent complete.")
