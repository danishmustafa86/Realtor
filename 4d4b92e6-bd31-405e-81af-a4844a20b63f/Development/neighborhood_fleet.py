import pandas as pd

# Get unique neighborhoods and fan out — one slice per neighborhood
unique_neighborhoods = sorted(properties_df["neighborhood"].unique().tolist())
print(f"🚀 Spreading fleet across {len(unique_neighborhoods)} neighborhoods: {unique_neighborhoods}")

# Spread creates one parallel execution per neighborhood
neighborhood_slice = spread(unique_neighborhoods)
