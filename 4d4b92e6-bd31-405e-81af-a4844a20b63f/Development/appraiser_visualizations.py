
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ── Zerve design system ───────────────────────────────────────────────────────
BG       = "#1D1D20"
PRIMARY  = "#fbfbff"
SECONDARY= "#909094"
COLORS   = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF"]
HIGHLIGHT= "#ffd400"
DANGER   = "#f04438"
SUCCESS  = "#17b26a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": SECONDARY, "axes.labelcolor": PRIMARY,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "text.color": PRIMARY, "grid.color": "#333337",
    "font.family": "sans-serif",
})

nbhd_order = neighborhood_summary.sort_values("nbhd_mean_ppsf", ascending=True)["neighborhood"].tolist()

# ─── Chart 1: Avg Price per SqFt by Neighborhood (horizontal bar) ────────────
fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor=BG)
ax1.set_facecolor(BG)

bars = ax1.barh(
    nbhd_order,
    [neighborhood_summary.loc[neighborhood_summary["neighborhood"] == n, "nbhd_mean_ppsf"].values[0] for n in nbhd_order],
    color=COLORS[:len(nbhd_order)],
    height=0.55,
)
for bar, val in zip(bars, [neighborhood_summary.loc[neighborhood_summary["neighborhood"] == n, "nbhd_mean_ppsf"].values[0] for n in nbhd_order]):
    ax1.text(val + 2, bar.get_y() + bar.get_height() / 2, f"${val:.0f}", va="center", fontsize=10, color=PRIMARY)

ax1.set_xlabel("Average Price per SqFt ($)", fontsize=11)
ax1.set_title("Average Price per SqFt by Neighborhood", fontsize=14, fontweight="bold", pad=14)
ax1.set_xlim(0, max([b.get_width() for b in bars]) * 1.18)
ax1.grid(axis="x", alpha=0.3)
ax1.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("chart1_avg_ppsf.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.show()

# ─── Chart 2: Property Value Score distribution (histogram) ──────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor=BG)
ax2.set_facecolor(BG)

ax2.hist(
    properties_df["property_value_score"],
    bins=20, color="#A1C9F4", edgecolor=BG, linewidth=0.6, alpha=0.88
)
ax2.axvline(5.5, color=HIGHLIGHT, linestyle="--", linewidth=1.6, label="Midpoint (5.5)")
ax2.axvline(
    properties_df["property_value_score"].mean(),
    color=SUCCESS, linestyle="-", linewidth=1.8,
    label=f"Mean ({properties_df['property_value_score'].mean():.1f})"
)

ax2.set_xlabel("Property Value Score (1–10)", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title("Distribution of Property Value Scores", fontsize=14, fontweight="bold", pad=14)
ax2.legend(framealpha=0.2, labelcolor=PRIMARY, fontsize=10)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("chart2_score_dist.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.show()

# ─── Chart 3: Scatter — Price/SqFt vs Z-Score, colored by outlier status ─────
fig3, ax3 = plt.subplots(figsize=(11, 6), facecolor=BG)
ax3.set_facecolor(BG)

normal_props  = properties_df[~properties_df["is_bargain_outlier"]]
bargain_props = properties_df[properties_df["is_bargain_outlier"]]

ax3.scatter(normal_props["ppsf_zscore"],  normal_props["price_per_sqft"],
            color="#A1C9F4", alpha=0.7, s=55, label="Normal", zorder=3)
ax3.scatter(bargain_props["ppsf_zscore"], bargain_props["price_per_sqft"],
            color=DANGER, alpha=0.9, s=90, marker="v", label="Bargain Outlier (z < −1)", zorder=4)

ax3.axvline(-1, color=HIGHLIGHT, linestyle="--", linewidth=1.5, alpha=0.8, label="z = −1 threshold")
ax3.axhline(properties_df["price_per_sqft"].mean(), color=SECONDARY, linestyle=":", linewidth=1.2, alpha=0.6)

ax3.set_xlabel("Neighborhood Z-Score (price/sqft)", fontsize=11)
ax3.set_ylabel("Price per SqFt ($)", fontsize=11)
ax3.set_title("Bargain Outlier Detection: Price/SqFt vs. Neighborhood Z-Score", fontsize=13, fontweight="bold", pad=14)
ax3.legend(framealpha=0.2, labelcolor=PRIMARY, fontsize=10)
ax3.grid(alpha=0.2)
ax3.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("chart3_outlier_scatter.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.show()

# ─── Chart 4: Bargain Outlier Count by Neighborhood (bar) ────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor=BG)
ax4.set_facecolor(BG)

nbhd_labels = neighborhood_summary["neighborhood"].tolist()
outlier_cnts = neighborhood_summary["bargain_outlier_count"].tolist()

bars4 = ax4.bar(nbhd_labels, outlier_cnts, color=[DANGER if c > 4 else "#FFB482" for c in outlier_cnts],
                width=0.5, edgecolor=BG)
for bar, cnt in zip(bars4, outlier_cnts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, str(cnt),
             ha="center", va="bottom", fontsize=12, color=PRIMARY, fontweight="bold")

ax4.set_ylabel("Number of Bargain Outlier Properties", fontsize=11)
ax4.set_title("Bargain Outliers Flagged per Neighborhood", fontsize=14, fontweight="bold", pad=14)
ax4.set_ylim(0, max(outlier_cnts) + 1.5)
ax4.grid(axis="y", alpha=0.3)
ax4.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("chart4_outliers_by_nbhd.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.show()

print("✅ All 4 visualizations rendered.")
