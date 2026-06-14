import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np

# ---------------------------------------------------------
# 1. Load the Data (Using synthetic_50k as the best example)
# ---------------------------------------------------------
dataset_name = "synthetic"
run_label = "baseline_cohort"
base_dir = Path("outputs") / dataset_name

print(f"Loading data from {base_dir}...")
features_df = pd.read_parquet(base_dir / f"{run_label}_person_level_phenotypes.parquet")
eras_df = pd.read_parquet(base_dir / f"{run_label}_eras.parquet")

# Convert dates to datetime objects for plotting
eras_df['era_start_date'] = pd.to_datetime(eras_df['era_start_date'])
eras_df['era_end_date'] = pd.to_datetime(eras_df['era_end_date'])
eras_df['era_duration'] = (eras_df['era_end_date'] - eras_df['era_start_date']).dt.days

# Load Vocabulary to get real names
try:
    concepts = pd.read_csv("synthetic_data/CONCEPT.csv", usecols=['concept_id', 'concept_name'])
    concept_dict = dict(zip(concepts['concept_id'], concepts['concept_name']))
    eras_df['drug_name'] = eras_df['ingredient_concept_id'].map(concept_dict)
    eras_df['drug_name'] = eras_df['drug_name'].fillna(eras_df['ingredient_concept_id'].astype(str))
except FileNotFoundError:
    eras_df['drug_name'] = eras_df['ingredient_concept_id'].astype(str)

# ---------------------------------------------------------
# 2. Identify Dominant Groups and Sample Patients
# ---------------------------------------------------------
# Find dominant drug per person to group them
dominant_drug = eras_df.loc[eras_df.groupby('person_id')['era_duration'].idxmax()]
dominant_drug = dominant_drug[['person_id', 'drug_name']]
features_df = features_df.merge(dominant_drug, on='person_id', how='left')

# Get the top 5 most common drug groups (the biggest UMAP islands)
top_groups = features_df[features_df['disc_evaluable'] == True]['drug_name'].value_counts().nlargest(5).index

# Sample 5 random patients from each of these top groups
sampled_patients = []
for group in top_groups:
    patients_in_group = features_df[features_df['drug_name'] == group]['person_id'].unique()
    # Pick 3 random patients using a fixed seed for reproducibility
    np.random.seed(42) 
    sampled = np.random.choice(patients_in_group, size=min(5, len(patients_in_group)), replace=False)
    for p in sampled:
        sampled_patients.append({'person_id': p, 'group': group})

sample_df = pd.DataFrame(sampled_patients)

# Filter the eras dataframe to ONLY include our sampled patients
plot_eras = eras_df[eras_df['person_id'].isin(sample_df['person_id'])].copy()

# ---------------------------------------------------------
# 3. Plot the Gantt Chart Timelines
# ---------------------------------------------------------
fig, axes = plt.subplots(nrows=len(top_groups), figsize=(12, 10), sharex=True)
fig.suptitle("Representative Patient Trajectories by Dominant Drug Group (synthetic 1k)", fontsize=16, fontweight='bold', y=0.95)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Standard distinct colors

for idx, group in enumerate(top_groups):
    ax = axes[idx]
    group_color = colors[idx % len(colors)]
    
    # Get the 3 patients for this specific group
    group_patients = sample_df[sample_df['group'] == group]['person_id'].tolist()
    
    y_ticks = []
    y_labels = []
    
    for y_pos, person in enumerate(group_patients):
        patient_eras = plot_eras[plot_eras['person_id'] == person]
        
        # Draw a line for every era this patient had
        for _, row in patient_eras.iterrows():
            start = row['era_start_date']
            end = row['era_end_date']
            
            # If the era drug matches the group dominant drug, color it solidly. 
            # If they took a *different* background drug, make it grey.
            color = group_color if row['drug_name'] == group else 'lightgrey'
            linewidth = 8 if row['drug_name'] == group else 4
            
            ax.plot([start, end], [y_pos, y_pos], color=color, linewidth=linewidth, solid_capstyle='butt')
            
        y_ticks.append(y_pos)
        y_labels.append(f"Patient {str(person)[-4:]}") # Just show last 4 digits of ID for clean look
        
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_title(f"UMAP Island: {group}", loc='left', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

# Formatting the X-axis as dates
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[-1].set_xlabel("Timeline (Year-Month)")

plt.tight_layout()
plt.subplots_adjust(top=0.88)
save_path = "outputs/representative_trajectories_gantt.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved successfully to: {save_path}")
plt.show()