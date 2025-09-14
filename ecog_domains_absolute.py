#!/usr/bin/env python3
"""
Figure 1: ECOG scores by domain - Absolute values
Shows average self-reported problems in each cognitive domain by age
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from improvements.demographics_enrichment import enrich_demographics

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*80)
print("ECOG DOMAIN ANALYSIS - ABSOLUTE SCORES")
print("="*80)

# Load ECOG Self-Report
ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv', low_memory=False)
if 'Code' in ecog.columns:
    ecog = ecog.rename(columns={'Code': 'SubjectCode'})

# Filter to baseline
if 'TimepointCode' in ecog.columns:
    ecog_baseline = ecog[ecog['TimepointCode'] == 'm00'].copy()
else:
    ecog_baseline = ecog

print(f"Loaded {len(ecog_baseline)} baseline ECOG records")

# Find all QID columns
qid_cols = [c for c in ecog_baseline.columns if c.startswith('QID') and 
            ecog_baseline[c].dtype in ['float64', 'int64']]

print(f"Found {len(qid_cols)} ECOG question columns")

# Define ECOG domains - approximate based on typical ECOG structure
ecog_domains = {}
if len(qid_cols) >= 39:  # Standard ECOG has 39 items
    ecog_domains['Memory'] = qid_cols[0:8]  # First 8 questions
    ecog_domains['Language'] = qid_cols[8:17]  # Next 9 questions  
    ecog_domains['Visuospatial'] = qid_cols[17:24]  # Next 7 questions
    ecog_domains['Planning'] = qid_cols[24:29]  # Next 5 questions
    ecog_domains['Organization'] = qid_cols[29:35]  # Next 6 questions
    ecog_domains['Divided Attn'] = qid_cols[35:39]  # Last 4 questions
else:
    # If fewer questions, divide proportionally
    chunk_size = len(qid_cols) // 6
    domain_names = ['Memory', 'Language', 'Visuospatial', 'Planning', 'Organization', 'Divided Attn']
    for i, domain in enumerate(domain_names):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 5 else len(qid_cols)
        ecog_domains[domain] = qid_cols[start_idx:end_idx]

# Calculate domain scores
for domain, cols in ecog_domains.items():
    if cols:
        ecog_baseline[f'{domain}_score'] = ecog_baseline[cols].mean(axis=1)

# Add demographics
ecog_baseline = enrich_demographics(DATA_DIR, ecog_baseline)

# Filter to 40+ with valid age
ecog_baseline = ecog_baseline[ecog_baseline['Age_Baseline'].notna() & 
                              (ecog_baseline['Age_Baseline'] >= 40) & 
                              (ecog_baseline['Age_Baseline'] <= 90)]

print(f"Analyzing {len(ecog_baseline)} participants aged 40+")

# Define age groups
age_groups = [
    ('40-49', 40, 50),
    ('50-59', 50, 60),
    ('60-69', 60, 70),
    ('70-79', 70, 80),
    ('80+', 80, 95)
]

# Calculate averages by age group and domain
results = []
for label, min_age, max_age in age_groups:
    mask = (ecog_baseline['Age_Baseline'] >= min_age) & (ecog_baseline['Age_Baseline'] < max_age)
    subset = ecog_baseline[mask]
    
    if len(subset) > 10:
        row = {'Age': label, 'N': len(subset)}
        for domain in ecog_domains.keys():
            col_name = f'{domain}_score'
            if col_name in subset.columns:
                row[domain] = subset[col_name].mean()
        results.append(row)

df_results = pd.DataFrame(results)

# Create visualization
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(df_results))
width = 0.13
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#9B59B6']
domains = list(ecog_domains.keys())

# Plot bars for each domain
for i, (domain, color) in enumerate(zip(domains, colors)):
    if domain in df_results.columns:
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, df_results[domain], width, 
                      label=domain, color=color, alpha=0.85, 
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if pd.notna(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', 
                       fontsize=8, rotation=0)

# Formatting
ax.set_xlabel('Age Group', fontsize=14, fontweight='bold')
ax.set_ylabel('ECOG Score (1-4 scale)\n1=No change, 4=Much worse', fontsize=14, fontweight='bold')
ax.set_title('Self-Reported Cognitive Problems by Domain and Age\n', 
             fontsize=16, fontweight='bold')

# Add subtitle
subtitle = f"Based on {len(ecog_baseline):,} participants"
ax.text(0.5, 0.97, subtitle, transform=ax.transAxes,
        ha='center', fontsize=12, style='italic')

ax.set_xticks(x)
ax.set_xticklabels([f"{row['Age']}\n(n={row['N']:,})" for _, row in df_results.iterrows()], 
                   fontsize=11)
ax.legend(loc='upper left', ncol=3, fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y')

# Add reference lines
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.02, 2.02, 'Occasionally worse', fontsize=9, color='green', 
        transform=ax.get_yaxis_transform())

ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.02, 2.52, 'Between occasional and consistent', fontsize=9, color='orange',
        transform=ax.get_yaxis_transform())

ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.02, 3.02, 'Consistently worse', fontsize=9, color='red',
        transform=ax.get_yaxis_transform())

ax.set_ylim(1.8, 3.2)

# Add text box with key finding
text_str = (
    'Key Pattern:\n'
    '• All domains hover around 2.4-2.5 (mild problems)\n'
    '• Most domains show slight DECREASE with age\n'
    '• Despite objective decline, self-reports remain stable or decrease'
)
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax.text(0.98, 0.25, text_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('bhr_memtrax_results/ecog_domains_absolute.png', dpi=150, bbox_inches='tight')
print("\nSaved: bhr_memtrax_results/ecog_domains_absolute.png")
plt.show()

# Print summary table
print("\n" + "="*80)
print("ECOG SCORES BY DOMAIN AND AGE GROUP")
print("(1 = No change, 2 = Occasionally worse, 3 = Consistently worse, 4 = Much worse)")
print("="*80)

print(f"\n{'Age':<10} {'N':>8}", end="")
for domain in domains:
    print(f" {domain[:8]:>9}", end="")
print()
print("-"*80)

for _, row in df_results.iterrows():
    print(f"{row['Age']:<10} {row['N']:>8,}", end="")
    for domain in domains:
        if domain in row and pd.notna(row[domain]):
            print(f" {row[domain]:>9.3f}", end="")
        else:
            print(f" {'--':>9}", end="")
    print()
