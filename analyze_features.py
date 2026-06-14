import pandas as pd

# 1. Load your final 50k data
dataset_name = "synthetic"
# Change this if your label is different now
parquet_path = f"outputs/{dataset_name}/baseline_cohort_person_level_phenotypes.parquet" 

print("Loading data...")
df = pd.read_parquet(parquet_path)

# 2. Define the features you care about
feature_cols = [
    "mean_active_n", "poly_month_prop", "mean_turnover", "burden_slope",
    "n_ingredient_eras", "n_distinct_ingredients", "early_disc_90_rate", 
    "restart_180_rate", "switch_60_rate", "median_era_days",
    "prop_NoRx", "prop_Initiation", "prop_StableMono", "prop_StableLowPoly"
]

# 3. Calculate the mean of each feature, grouped by cluster!
print("Calculating cluster profiles...")
cluster_profiles = df.groupby("trajectory_cluster")[feature_cols].mean().T

# Format it to look nice (round to 3 decimal places)
cluster_profiles = cluster_profiles.round(3)

# 4. Save to CSV so you can copy/paste it into your Thesis Word Document!
save_path = f"outputs/{dataset_name}/cluster_feature_profiles.csv"
cluster_profiles.to_csv(save_path)

print(f"Success! Feature importance table saved to: {save_path}")
print("\nPreview of the table:")
print(cluster_profiles.head(10))