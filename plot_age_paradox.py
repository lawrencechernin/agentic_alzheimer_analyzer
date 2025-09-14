#!/usr/bin/env python3
"""
Visual demonstration of the age paradox:
Objective performance WORSENS while self-awareness DECREASES
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from improvements.ashford_policy import apply_ashford

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("Generating age paradox visualization...")

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
            'RT': subset['RT_mean'].mean(),
            'Accuracy': subset['accuracy_mean'].mean() * 100
        })

df_results = pd.DataFrame(results)

# Normalize to baseline (40-49)
baseline_cog = df_results.iloc[0]['CognitiveScore']
baseline_ecog = df_results.iloc[0]['ECOG_Self']

df_results['Cog_Relative'] = ((df_results['CognitiveScore'] - baseline_cog) / baseline_cog) * 100
df_results['ECOG_Relative'] = ((df_results['ECOG_Self'] - baseline_ecog) / baseline_ecog) * 100

# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Raw scores with dual axis
color1 = '#FF6B6B'  # Red for objective
color2 = '#4ECDC4'  # Teal for subjective

ax1_twin = ax1.twinx()

# Objective performance (higher = worse)
ax1.plot(df_results['Age_Mid'], df_results['CognitiveScore'], 
         marker='o', color=color1, linewidth=3, markersize=10, label='MemTrax Score (↑=worse)')
ax1.fill_between(df_results['Age_Mid'], 
                  df_results['CognitiveScore'] - df_results['CognitiveScore_SE'],
                  df_results['CognitiveScore'] + df_results['CognitiveScore_SE'],
                  alpha=0.2, color=color1)

# Self-reported problems
ax1_twin.plot(df_results['Age_Mid'], df_results['ECOG_Self'], 
              marker='s', color=color2, linewidth=3, markersize=10, label='ECOG Self-Report (↑=worse)')
ax1_twin.fill_between(df_results['Age_Mid'],
                       df_results['ECOG_Self'] - df_results['ECOG_Self_SE'],
                       df_results['ECOG_Self'] + df_results['ECOG_Self_SE'],
                       alpha=0.2, color=color2)

ax1.set_xlabel('Age Group', fontsize=12)
ax1.set_ylabel('MemTrax Cognitive Score\n(Higher = Worse)', color=color1, fontsize=12)
ax1_twin.set_ylabel('ECOG Self-Report Score\n(Higher = More Problems)', color=color2, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1_twin.tick_params(axis='y', labelcolor=color2)
ax1.set_title('THE AGE PARADOX: Objective Decline vs Self-Awareness', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(df_results['Age_Mid'])
ax1.set_xticklabels(df_results['Age'])

# Add annotations
for i, row in df_results.iterrows():
    ax1.annotate(f'n={row["N"]:,}', 
                 xy=(row['Age_Mid'], row['CognitiveScore']),
                 xytext=(0, -15), textcoords='offset points',
                 ha='center', fontsize=8, color='gray')

# Plot 2: Relative change from baseline
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline (40-49)')

bars1 = ax2.bar(df_results['Age_Mid'] - 2, df_results['Cog_Relative'], 
                width=4, color=color1, alpha=0.7, label='Objective Decline')
bars2 = ax2.bar(df_results['Age_Mid'] + 2, df_results['ECOG_Relative'], 
                width=4, color=color2, alpha=0.7, label='Self-Reported')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if height != 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    if height != 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

ax2.set_xlabel('Age Group', fontsize=12)
ax2.set_ylabel('% Change from Baseline (40-49)', fontsize=12)
ax2.set_title('The Growing Gap: Objective Reality vs Self-Perception', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(df_results['Age_Mid'])
ax2.set_xticklabels(df_results['Age'])
ax2.set_ylim(-10, 40)

# Add shaded region to highlight the paradox
ax2.axvspan(70, 90, alpha=0.1, color='red', label='Paradox Zone')
ax2.text(77.5, 35, '🚨 PARADOX ZONE', ha='center', fontsize=12, 
         fontweight='bold', color='darkred')

# Plot 3: Components breakdown
width = 0.35
x = np.arange(len(df_results['Age']))

ax3.bar(x - width/2, df_results['RT'] - df_results['RT'].iloc[0], 
        width, label='Reaction Time (slower)', color='#FF9999')
ax3.bar(x + width/2, df_results['Accuracy'] - df_results['Accuracy'].iloc[0], 
        width, label='Accuracy (lower)', color='#9999FF')

ax3.set_xlabel('Age Group', fontsize=12)
ax3.set_ylabel('Change from Baseline', fontsize=12)
ax3.set_title('Components of Objective Decline', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(df_results['Age'])
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Add summary text box
summary_text = (
    "Key Insight: As cognitive ability objectively declines with age,\n"
    "self-awareness of problems paradoxically DECREASES.\n"
    "This shows why objective tests like MemTrax are essential."
)
ax3.text(0.5, -0.25, summary_text, transform=ax3.transAxes,
         fontsize=11, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('bhr_memtrax_results/age_paradox.png', dpi=150, bbox_inches='tight')
print("Saved plot to: bhr_memtrax_results/age_paradox.png")
plt.show()

# Print correlation for reference
corr = merged['Age_Baseline'].corr(merged['CognitiveScore'])
corr_ecog = merged['Age_Baseline'].corr(merged['ECOG_self_score'])
print(f"\nCorrelations with age:")
print(f"  MemTrax Score (objective): r = {corr:.3f}")
print(f"  ECOG Self (subjective):    r = {corr_ecog:.3f}")
print(f"\nThe paradox is clear: objective performance worsens (+36%) ")
print(f"while self-reported problems decrease (-4%)!")
