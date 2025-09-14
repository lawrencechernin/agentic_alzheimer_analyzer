#!/usr/bin/env python3
"""
Figure 1: Raw scores showing the age paradox
Objective performance vs Self-awareness across age groups
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from improvements.ashford_policy import apply_ashford

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("Generating Figure 1: Raw Scores...")

# Load MemTrax
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
memtrax_q = apply_ashford(memtrax, accuracy_threshold=0.60)

# Aggregate MemTrax by subject
memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
    'CorrectPCT': ['mean', 'std'],
    'CorrectResponsesRT': ['mean', 'std', 'count']
}).reset_index()

memtrax_agg.columns = ['SubjectCode', 'accuracy_mean', 'accuracy_std', 
                       'RT_mean', 'RT_std', 'test_count']
memtrax_agg['CognitiveScore'] = memtrax_agg['RT_mean'] / (memtrax_agg['accuracy_mean'] + 0.01)

# Load ECOG Self-Report
ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv', low_memory=False)
if 'Code' in ecog.columns:
    ecog = ecog.rename(columns={'Code': 'SubjectCode'})
if 'TimepointCode' in ecog.columns:
    ecog_baseline = ecog[ecog['TimepointCode'] == 'm00'].copy()
else:
    ecog_baseline = ecog

ecog_cols = [c for c in ecog_baseline.columns if c.startswith('QID') and 
             ecog_baseline[c].dtype in ['float64', 'int64']]
if ecog_cols:
    ecog_baseline['ECOG_self_score'] = ecog_baseline[ecog_cols].mean(axis=1)
    ecog_small = ecog_baseline[['SubjectCode', 'ECOG_self_score']].drop_duplicates()

# Load demographics and merge
from improvements.demographics_enrichment import enrich_demographics

merged = memtrax_agg.merge(ecog_small, on='SubjectCode', how='inner')
merged = enrich_demographics(DATA_DIR, merged)

# Filter to those with age data
merged = merged[merged['Age_Baseline'].notna() & 
                (merged['Age_Baseline'] >= 40) & 
                (merged['Age_Baseline'] <= 90)]

# Calculate statistics by age group
age_groups = [
    ('40-49', 40, 50, 45),
    ('50-59', 50, 60, 55),
    ('60-69', 60, 70, 65),
    ('70-79', 70, 80, 75),
    ('80+', 80, 95, 85)
]

results = []
for label, min_age, max_age, mid_age in age_groups:
    mask = (merged['Age_Baseline'] >= min_age) & (merged['Age_Baseline'] < max_age)
    subset = merged[mask]
    
    if len(subset) > 10:
        results.append({
            'Age': label,
            'Age_Mid': mid_age,
            'N': len(subset),
            'CognitiveScore': subset['CognitiveScore'].mean(),
            'CognitiveScore_SE': subset['CognitiveScore'].sem(),
            'ECOG_Self': subset['ECOG_self_score'].mean(),
            'ECOG_Self_SE': subset['ECOG_self_score'].sem(),
        })

df_results = pd.DataFrame(results)

# Create Figure 1: Raw Scores
fig, ax1 = plt.subplots(figsize=(12, 8))

color1 = '#FF6B6B'  # Red for objective
color2 = '#4ECDC4'  # Teal for subjective

ax1_twin = ax1.twinx()

# Plot objective performance (higher = worse)
line1 = ax1.plot(df_results['Age_Mid'], df_results['CognitiveScore'], 
         marker='o', color=color1, linewidth=4, markersize=12, 
         label='MemTrax Score (objective)', zorder=3)
ax1.fill_between(df_results['Age_Mid'], 
                  df_results['CognitiveScore'] - df_results['CognitiveScore_SE'],
                  df_results['CognitiveScore'] + df_results['CognitiveScore_SE'],
                  alpha=0.2, color=color1)

# Plot self-reported problems
line2 = ax1_twin.plot(df_results['Age_Mid'], df_results['ECOG_Self'], 
              marker='s', color=color2, linewidth=4, markersize=12, 
              label='ECOG Self-Report', linestyle='--', zorder=3)
ax1_twin.fill_between(df_results['Age_Mid'],
                       df_results['ECOG_Self'] - df_results['ECOG_Self_SE'],
                       df_results['ECOG_Self'] + df_results['ECOG_Self_SE'],
                       alpha=0.2, color=color2)

# Formatting
ax1.set_xlabel('Age Group', fontsize=16, fontweight='bold')
ax1.set_ylabel('MemTrax Cognitive Score\n(Higher = Worse Performance)', 
               color=color1, fontsize=14, fontweight='bold')
ax1_twin.set_ylabel('ECOG Self-Report Score\n(Higher = More Complaints)', 
                    color=color2, fontsize=14, fontweight='bold')

ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1_twin.tick_params(axis='y', labelcolor=color2, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

ax1.set_title('THE AGE PARADOX: Objective Decline vs Self-Awareness\n', 
              fontsize=18, fontweight='bold')

# Add subtitle with sample size
subtitle = f"N = {merged.shape[0]:,} participants"
ax1.text(0.5, 0.97, subtitle, transform=ax1.transAxes, 
         ha='center', fontsize=12, style='italic')

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(df_results['Age_Mid'])
ax1.set_xticklabels(df_results['Age'], fontsize=12)

# Add sample sizes below each point
for i, row in df_results.iterrows():
    ax1.annotate(f'n={row["N"]:,}', 
                 xy=(row['Age_Mid'], ax1.get_ylim()[0]), 
                 xytext=(0, -25), textcoords='offset points',
                 ha='center', fontsize=10, color='gray')

# Set y-axis limits for better visualization
ax1.set_ylim(0.8, 1.25)
ax1_twin.set_ylim(2.35, 2.55)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
           loc='upper left', fontsize=12, framealpha=0.9)

# Add arrows to emphasize the paradox
ax1.annotate('', xy=(82, 1.15), xytext=(82, 0.90),
            arrowprops=dict(arrowstyle='->', color=color1, lw=2))
ax1.text(83, 1.02, 'Worsening', rotation=90, va='center', 
         color=color1, fontweight='bold', fontsize=11)

ax1_twin.annotate('', xy=(77, 2.44), xytext=(77, 2.50),
            arrowprops=dict(arrowstyle='->', color=color2, lw=2))
ax1_twin.text(78, 2.47, 'Fewer\nComplaints', ha='left', va='center',
              color=color2, fontweight='bold', fontsize=11)

# Add text box explaining the paradox
textstr = 'The Paradox:\nAs cognitive ability objectively declines,\nself-awareness of problems decreases'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.98, 0.15, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('bhr_memtrax_results/age_paradox_raw_scores.png', dpi=150, bbox_inches='tight')
print("Saved: bhr_memtrax_results/age_paradox_raw_scores.png")
plt.show()
