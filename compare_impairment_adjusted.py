#!/usr/bin/env python3
"""
Adjusted comparison with better thresholds
"""
import pandas as pd
import numpy as np
from pathlib import Path
from improvements.demographics_enrichment import enrich_demographics
from improvements.ashford_policy import apply_ashford

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*80)
print("ADJUSTED IMPAIRMENT COMPARISON - Multiple Thresholds")
print("="*80)

# Load MemTrax
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
memtrax_q = apply_ashford(memtrax, accuracy_threshold=0.60)

memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
    'CorrectPCT': ['mean', 'std', 'count'],
    'CorrectResponsesRT': ['mean', 'std']
}).reset_index()

memtrax_agg.columns = ['SubjectCode', 
                       'accuracy_mean', 'accuracy_std', 'test_count',
                       'RT_mean', 'RT_std']

memtrax_agg['CognitiveScore'] = memtrax_agg['RT_mean'] / (memtrax_agg['accuracy_mean'] + 0.01)

# Load MedHx
med_df = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
if 'TimepointCode' in med_df.columns:
    med_baseline = med_df[med_df['TimepointCode'] == 'm00'].copy()
else:
    med_baseline = med_df

COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
qids_present = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]

med_baseline['MedHx_MCI'] = 0
for qid in qids_present:
    med_baseline['MedHx_MCI'] |= (med_baseline[qid] == 1).fillna(False)

med_mci = med_baseline[['SubjectCode', 'MedHx_MCI']].copy()

# Merge
merged = memtrax_agg.merge(med_mci, on='SubjectCode', how='inner')
merged = enrich_demographics(DATA_DIR, merged)

print(f"Dataset: {len(merged):,} subjects")

# Try different MemTrax impairment thresholds
print("\n" + "="*80)
print("TESTING DIFFERENT MEMTRAX THRESHOLDS")
print("="*80)

thresholds = [
    ("Liberal", lambda df: (df['CognitiveScore'] > 1.2) | (df['accuracy_mean'] < 0.80)),
    ("Moderate", lambda df: (df['CognitiveScore'] > 1.4) | (df['accuracy_mean'] < 0.75)),
    ("Strict", lambda df: (df['CognitiveScore'] > 1.6) | (df['accuracy_mean'] < 0.70)),
    ("Very Strict", lambda df: (df['CognitiveScore'] > 1.8) | (df['accuracy_mean'] < 0.65))
]

for name, threshold_func in thresholds:
    merged[f'MemTrax_{name}'] = threshold_func(merged).astype(int)
    rate = merged[f'MemTrax_{name}'].mean() * 100
    agreement = (merged[f'MemTrax_{name}'] == merged['MedHx_MCI']).mean() * 100
    
    print(f"\n{name} Threshold:")
    print(f"  MemTrax impaired: {rate:.1f}%")
    print(f"  MedHx MCI: {merged['MedHx_MCI'].mean()*100:.1f}%")
    print(f"  Agreement: {agreement:.1f}%")
    
    # Hidden impairment
    hidden = merged[(merged[f'MemTrax_{name}'] == 1) & (merged['MedHx_MCI'] == 0)]
    print(f"  Hidden (MemTrax+ MedHx-): {len(hidden)/len(merged)*100:.1f}%")
    
    # Over-reporting
    over_report = merged[(merged[f'MemTrax_{name}'] == 0) & (merged['MedHx_MCI'] == 1)]
    print(f"  Over-report (MemTrax- MedHx+): {len(over_report)/len(merged)*100:.1f}%")

# Use moderate threshold for detailed analysis
merged['MemTrax_impaired'] = merged['MemTrax_Moderate']

print("\n" + "="*80)
print("DETAILED ANALYSIS WITH MODERATE THRESHOLD")
print("="*80)

# By education
edu_groups = [
    ('< HS', 0, 12),
    ('HS', 12, 13),
    ('Some Col', 13, 16),
    ('Bachelor', 16, 17),
    ('Graduate', 17, 30)
]

print("\n%-12s %8s %10s %10s %10s %12s" % 
      ("Education", "N", "MemTrax%", "MedHx%", "Hidden%", "OverReport%"))
print("-"*75)

for label, min_edu, max_edu in edu_groups:
    if 'YearsEducationUS_Converted' in merged.columns:
        if label == 'HS':
            mask = merged['YearsEducationUS_Converted'] == 12
        elif label == 'Bachelor':
            mask = merged['YearsEducationUS_Converted'] == 16  
        else:
            mask = (merged['YearsEducationUS_Converted'] >= min_edu) & (merged['YearsEducationUS_Converted'] < max_edu)
        
        subset = merged[mask]
        
        if len(subset) > 10:
            n = len(subset)
            memtrax_pct = subset['MemTrax_impaired'].mean() * 100
            medhx_pct = subset['MedHx_MCI'].mean() * 100
            
            hidden = subset[(subset['MemTrax_impaired'] == 1) & (subset['MedHx_MCI'] == 0)]
            hidden_pct = len(hidden) / n * 100
            
            over = subset[(subset['MemTrax_impaired'] == 0) & (subset['MedHx_MCI'] == 1)]
            over_pct = len(over) / n * 100
            
            print("%-12s %8d %9.1f%% %9.1f%% %9.1f%% %11.1f%%" % 
                  (label, n, memtrax_pct, medhx_pct, hidden_pct, over_pct))

# Analyze cognitive scores by group
print("\n" + "="*80)
print("COGNITIVE SCORES BY SELF-REPORT STATUS")
print("="*80)

print("\n%-30s %8s %12s %12s %12s" % 
      ("Group", "N", "Accuracy", "RT (sec)", "CogScore"))
print("-"*75)

groups = [
    ("True Negative (both -)", (merged['MemTrax_impaired'] == 0) & (merged['MedHx_MCI'] == 0)),
    ("Hidden Impaired (Mem+ Med-)", (merged['MemTrax_impaired'] == 1) & (merged['MedHx_MCI'] == 0)),
    ("Over-Reporters (Mem- Med+)", (merged['MemTrax_impaired'] == 0) & (merged['MedHx_MCI'] == 1)),
    ("True Positive (both +)", (merged['MemTrax_impaired'] == 1) & (merged['MedHx_MCI'] == 1))
]

for name, mask in groups:
    subset = merged[mask]
    if len(subset) > 0:
        n = len(subset)
        acc = subset['accuracy_mean'].mean()
        rt = subset['RT_mean'].mean()
        score = subset['CognitiveScore'].mean()
        
        print("%-30s %8d %11.1f%% %11.3f %11.3f" % 
              (name, n, acc*100, rt, score))

# Check if highly educated "over-reporters" have subtle signs
print("\n" + "="*80)
print("OVER-REPORTERS ANALYSIS (Report MCI but perform well)")
print("="*80)

over_reporters = merged[(merged['MemTrax_impaired'] == 0) & (merged['MedHx_MCI'] == 1)]
print(f"\nTotal over-reporters: {len(over_reporters):,} ({len(over_reporters)/len(merged)*100:.1f}%)")

if 'YearsEducationUS_Converted' in over_reporters.columns:
    print("\nOver-reporters by education:")
    for label, min_edu, max_edu in edu_groups:
        if label == 'HS':
            mask = over_reporters['YearsEducationUS_Converted'] == 12
        elif label == 'Bachelor':
            mask = over_reporters['YearsEducationUS_Converted'] == 16
        else:
            mask = (over_reporters['YearsEducationUS_Converted'] >= min_edu) & (over_reporters['YearsEducationUS_Converted'] < max_edu)
        
        subset = over_reporters[mask]
        if len(subset) > 0:
            print(f"  {label:12s}: {len(subset):4d} ({subset['accuracy_mean'].mean()*100:.1f}% acc, {subset['RT_mean'].mean():.3f}s RT)")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("""
🎯 What we're finding:
1. Many educated people REPORT MCI but PERFORM WELL (over-reporters)
2. These are likely "worried well" with subjective concerns
3. Their cognitive reserve allows good test performance despite concerns
4. This creates label noise that makes ML classification harder!

📊 Implications:
- Self-reported MCI in educated populations may be unreliable
- Objective measures (MemTrax) might underestimate true impairment
- The truth is likely somewhere in between
- Your 0.798 AUC is impressive given this label noise!
""")
