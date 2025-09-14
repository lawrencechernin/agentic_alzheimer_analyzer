#!/usr/bin/env python3
"""
Investigate why informant labels perform poorly
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

# Load all data
print("Loading data...")
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
demo = pd.read_csv(DATA_DIR / 'BHR_Demographics.csv', low_memory=False)

# Get baseline only
memtrax_baseline = memtrax[memtrax['TimepointCode'] == 'm00'] if 'TimepointCode' in memtrax.columns else memtrax
sp_ecog['TimepointCode'] = sp_ecog['TimepointCode'].str.replace('sp-', '') if 'TimepointCode' in sp_ecog.columns else 'm00'
sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'm00'] if 'TimepointCode' in sp_ecog.columns else sp_ecog
med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'] if 'TimepointCode' in med_hx.columns else med_hx

# Quality filter MemTrax
memtrax_q = memtrax_baseline[
    (memtrax_baseline['Status'] == 'Collected') &
    (memtrax_baseline['CorrectPCT'] >= 0.60) &
    (memtrax_baseline['CorrectResponsesRT'].between(0.5, 2.5))
]

# Get unique subjects
memtrax_subjects = set(memtrax_q['SubjectCode'].unique())
sp_ecog_subjects = set(sp_ecog_baseline['SubjectCode'].unique())
med_subjects = set(med_baseline['SubjectCode'].unique())

print("\n" + "="*70)
print("SUBJECT OVERLAP ANALYSIS")
print("="*70)
print(f"MemTrax subjects: {len(memtrax_subjects):,}")
print(f"SP-ECOG subjects: {len(sp_ecog_subjects):,}")
print(f"Medical Hx subjects: {len(med_subjects):,}")

# Overlaps
memtrax_sp = memtrax_subjects & sp_ecog_subjects
memtrax_med = memtrax_subjects & med_subjects
all_three = memtrax_subjects & sp_ecog_subjects & med_subjects

print(f"\nOverlaps:")
print(f"MemTrax ∩ SP-ECOG: {len(memtrax_sp):,} ({len(memtrax_sp)/len(memtrax_subjects)*100:.1f}% of MemTrax)")
print(f"MemTrax ∩ Medical: {len(memtrax_med):,} ({len(memtrax_med)/len(memtrax_subjects)*100:.1f}% of MemTrax)")
print(f"All three: {len(all_three):,}")

# Analyze characteristics of those with informant data
print("\n" + "="*70)
print("WHO GETS INFORMANT ASSESSMENTS?")
print("="*70)

# Get demographics for analysis
demo_baseline = demo[demo['TimepointCode'] == 'm00'] if 'TimepointCode' in demo.columns else demo
demo_baseline = demo_baseline.drop_duplicates('SubjectCode')

# Add age and education
if 'QID186' in demo_baseline.columns:
    demo_baseline['Age'] = demo_baseline['QID186']
if 'QID184' in demo_baseline.columns:
    demo_baseline['Education'] = demo_baseline['QID184']

# Mark who has informant data
demo_baseline['has_informant'] = demo_baseline['SubjectCode'].isin(sp_ecog_subjects)
demo_baseline['has_memtrax'] = demo_baseline['SubjectCode'].isin(memtrax_subjects)
demo_baseline['has_both'] = demo_baseline['SubjectCode'].isin(memtrax_sp)

# Compare demographics
for group_name, group_mask in [
    ('MemTrax only', demo_baseline['has_memtrax'] & ~demo_baseline['has_informant']),
    ('Informant only', demo_baseline['has_informant'] & ~demo_baseline['has_memtrax']),
    ('Both', demo_baseline['has_both'])
]:
    subset = demo_baseline[group_mask]
    if len(subset) > 0:
        print(f"\n{group_name} (n={len(subset):,}):")
        if 'Age' in subset.columns:
            print(f"  Age: {subset['Age'].mean():.1f} ± {subset['Age'].std():.1f}")
        if 'Education' in subset.columns:
            print(f"  Education: {subset['Education'].mean():.1f} ± {subset['Education'].std():.1f}")

# Check MCI rates in different groups
print("\n" + "="*70)
print("MCI RATES BY DATA AVAILABILITY")
print("="*70)

# Get MCI labels
MCI_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
mci = np.zeros(len(med_baseline), dtype=int)
for qid in MCI_QIDS:
    if qid in med_baseline.columns:
        mci |= (med_baseline[qid] == 1).values

med_baseline['mci'] = mci
med_baseline['has_informant'] = med_baseline['SubjectCode'].isin(sp_ecog_subjects)
med_baseline['has_memtrax'] = med_baseline['SubjectCode'].isin(memtrax_subjects)

print("MCI prevalence by group:")
print(f"  Has MemTrax only: {med_baseline[med_baseline['has_memtrax'] & ~med_baseline['has_informant']]['mci'].mean():.1%}")
print(f"  Has Informant only: {med_baseline[~med_baseline['has_memtrax'] & med_baseline['has_informant']]['mci'].mean():.1%}")
print(f"  Has Both: {med_baseline[med_baseline['has_memtrax'] & med_baseline['has_informant']]['mci'].mean():.1%}")

# Analyze SP-ECOG scores for those with both
print("\n" + "="*70)
print("SP-ECOG SCORES VS MEMTRAX PERFORMANCE")
print("="*70)

# Get subjects with both
overlap_subjects = list(memtrax_sp)[:1000]  # Sample for speed

# Get MemTrax performance
memtrax_perf = memtrax_q[memtrax_q['SubjectCode'].isin(overlap_subjects)].groupby('SubjectCode').agg({
    'CorrectPCT': 'mean',
    'CorrectResponsesRT': 'mean'
}).reset_index()

# Get SP-ECOG scores
qid_cols = [c for c in sp_ecog_baseline.columns if 'QID' in c and sp_ecog_baseline[c].dtype in [np.float64, np.int64]]
sp_scores = sp_ecog_baseline[sp_ecog_baseline['SubjectCode'].isin(overlap_subjects)].copy()
sp_scores['sp_mean'] = sp_scores[qid_cols].replace(8, np.nan).mean(axis=1)
sp_scores = sp_scores[['SubjectCode', 'sp_mean']]

# Merge
combined = memtrax_perf.merge(sp_scores, on='SubjectCode')

if len(combined) > 0:
    # Plot relationships
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(combined['CorrectPCT'], combined['sp_mean'], alpha=0.5)
    axes[0].set_xlabel('MemTrax Accuracy')
    axes[0].set_ylabel('SP-ECOG Mean Score')
    axes[0].set_title(f'Correlation: {combined["CorrectPCT"].corr(combined["sp_mean"]):.3f}')
    
    axes[1].scatter(combined['CorrectResponsesRT'], combined['sp_mean'], alpha=0.5)
    axes[1].set_xlabel('MemTrax RT (seconds)')
    axes[1].set_ylabel('SP-ECOG Mean Score')
    axes[1].set_title(f'Correlation: {combined["CorrectResponsesRT"].corr(combined["sp_mean"]):.3f}')
    
    plt.suptitle('MemTrax vs Informant Assessment')
    plt.tight_layout()
    plt.savefig('memtrax_informant_correlation.png', dpi=150)
    plt.show()
    
    print(f"\nCorrelations (n={len(combined)}):")
    print(f"  Accuracy vs SP-ECOG: {combined['CorrectPCT'].corr(combined['sp_mean']):.3f}")
    print(f"  RT vs SP-ECOG: {combined['CorrectResponsesRT'].corr(combined['sp_mean']):.3f}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. Limited overlap between MemTrax and informant data
2. Selection bias in who gets informant assessments
3. Weak correlation between objective performance and informant reports
4. Different constructs being measured
""")

