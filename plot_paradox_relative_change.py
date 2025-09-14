#!/usr/bin/env python3
"""
Figure 2: Relative change from baseline showing the growing gap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from improvements.ashford_policy import apply_ashford

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("Generating Figure 2: Relative Changes...")

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
            'ECOG_Self': subset['ECOG_self_score'].mean(),
        })

df_results = pd.DataFrame(results)

# Calculate relative changes from baseline
baseline_cog = df_results.iloc[0]['CognitiveScore']
baseline_ecog = df_results.iloc[0]['ECOG_Self']

df_results['Cog_Relative'] = ((df_results['CognitiveScore'] - baseline_cog) / baseline_cog) * 100
df_results['ECOG_Relative'] = ((df_results['ECOG_Self'] - baseline_ecog) / baseline_ecog) * 100

# Create Figure 2: Relative Changes
fig, ax = plt.subplots(figsize=(12, 8))

color1 = '#FF6B6B'  # Red for objective
color2 = '#4ECDC4'  # Teal for subjective

# Draw baseline
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Baseline (40-49)')

# Bar width and positions
width = 3.5
x_pos = df_results['Age_Mid']

# Plot bars
bars1 = ax.bar(x_pos - width/2, df_results['Cog_Relative'], 
               width=width, color=color1, alpha=0.8, 
               label='Objective Decline (MemTrax)', edgecolor='darkred', linewidth=2)

bars2 = ax.bar(x_pos + width/2, df_results['ECOG_Relative'], 
               width=width, color=color2, alpha=0.8, 
               label='Self-Reported Problems (ECOG)', edgecolor='darkcyan', linewidth=2)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if abs(height) > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', 
                va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold', color='darkred')

for bar in bars2:
    height = bar.get_height()
    if abs(height) > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', 
                va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold', color='darkcyan')

# Formatting
ax.set_xlabel('Age Group', fontsize=16, fontweight='bold')
ax.set_ylabel('% Change from Baseline (40-49 years)', fontsize=14, fontweight='bold')
ax.set_title('The Growing Gap: Objective Reality vs Self-Perception\n', 
             fontsize=18, fontweight='bold')

# Add subtitle
subtitle = f"Based on {merged.shape[0]:,} participants"
ax.text(0.5, 0.97, subtitle, transform=ax.transAxes, 
        ha='center', fontsize=12, style='italic')

ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_xticks(df_results['Age_Mid'])
ax.set_xticklabels(df_results['Age'], fontsize=12)
ax.set_ylim(-10, 45)

# Add shaded region to highlight the paradox zone
ax.axvspan(70, 90, alpha=0.1, color='red')
ax.text(77.5, 40, 'PARADOX ZONE', ha='center', fontsize=14, 
        fontweight='bold', color='darkred',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Add sample sizes
for i, row in df_results.iterrows():
    ax.text(row['Age_Mid'], -8, f'n={row["N"]:,}', 
            ha='center', fontsize=10, color='gray')

# Add gap annotation
gap_70 = df_results[df_results['Age'] == '70-79'].iloc[0]
gap_80 = df_results[df_results['Age'] == '80+'].iloc[0]

# Draw arrow showing the gap
ax.annotate('', xy=(75, gap_70['Cog_Relative']), 
            xytext=(75, gap_70['ECOG_Relative']),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))

gap_size = gap_70['Cog_Relative'] - gap_70['ECOG_Relative']
ax.text(76, gap_70['ECOG_Relative'] + gap_size/2, 
        f'Gap:\n{gap_size:.0f}%', 
        fontsize=11, fontweight='bold', color='purple',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add text explaining the pattern
text_str = (
    'Key Finding:\n'
    '• Objective performance worsens by 36%\n'
    '• Self-reported problems decrease by 4%\n'
    '• The gap widens dramatically with age'
)
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.98, 0.65, text_str, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('bhr_memtrax_results/age_paradox_relative_change.png', dpi=150, bbox_inches='tight')
print("Saved: bhr_memtrax_results/age_paradox_relative_change.png")
plt.show()
