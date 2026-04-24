
import pandas as pd
import numpy as np

np.random.seed(42)

neighborhoods = {
    "Riverside Heights":   {"base_price": 450000, "base_sqft": 1800, "std": 80000},
    "Maple Grove":         {"base_price": 320000, "base_sqft": 1500, "std": 60000},
    "Sunset Park":         {"base_price": 580000, "base_sqft": 2100, "std": 100000},
    "Old Town":            {"base_price": 270000, "base_sqft": 1300, "std": 50000},
    "Lakeview Terrace":    {"base_price": 710000, "base_sqft": 2500, "std": 130000},
}

records = []
n_per_neighborhood = 30

for nbhd, params in neighborhoods.items():
    for _ in range(n_per_neighborhood):
        sqft = int(np.random.normal(params["base_sqft"], params["base_sqft"] * 0.12))
        sqft = max(700, sqft)
        bedrooms = int(np.clip(np.round(sqft / 550 + np.random.normal(0, 0.5)), 1, 6))
        bathrooms = round(max(1.0, bedrooms * 0.6 + np.random.normal(0, 0.3)), 1)

        # Price correlates with sqft + neighborhood + noise
        price = int(np.random.normal(
            params["base_price"] + (sqft - params["base_sqft"]) * (params["base_price"] / params["base_sqft"]) * 0.8,
            params["std"]
        ))
        price = max(100000, price)

        records.append({
            "neighborhood": nbhd,
            "price": price,
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
        })

raw_df = pd.DataFrame(records)

# Add a few deliberate outlier bargains (significantly underpriced)
bargain_indices = np.random.choice(len(raw_df), size=8, replace=False)
for idx in bargain_indices:
    raw_df.loc[idx, "price"] = int(raw_df.loc[idx, "price"] * np.random.uniform(0.35, 0.55))

# Save as CSV string (in-memory, stored as variable)
real_estate_csv = raw_df.to_csv(index=False)

print(f"✅ Synthetic real estate dataset generated: {len(raw_df)} properties across {raw_df['neighborhood'].nunique()} neighborhoods")
print(f"   Deliberate bargain outliers injected at indices: {sorted(bargain_indices.tolist())}")
print(f"\nSample rows:")
print(raw_df.head(8).to_string(index=True))
print(f"\nData types:\n{raw_df.dtypes}")
print(f"\nPrice range: ${raw_df['price'].min():,} — ${raw_df['price'].max():,}")
print(f"Sqft range: {raw_df['sqft'].min()} — {raw_df['sqft'].max()}")
