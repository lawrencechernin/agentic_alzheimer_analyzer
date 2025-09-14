#!/usr/bin/env python3
"""
Fixed: Comprehensive Impairment Comparison INCLUDING INFORMANT ECOG
Handles SP_ECOG with "sp-" prefixed timepoints and adjusted thresholds
"""
import pandas as pd
import numpy as np
from pathlib import Path
from improvements.demographics_enrichment import enrich_demographics
from improvements.ashford_policy import apply_ashford

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*90)
print("COMPREHENSIVE IMPAIRMENT COMPARISON - WITH INFORMANT DATA")
print("="*90)

# 1. Load Medical History as BASE
print("\nLoading Medical History (BASE)...")
med_df = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
if 'TimepointCode' in med_df.columns:
    med_baseline = med_df[med_df['TimepointCode'] == 'm00'].copy()
else:
    med_baseline = med_df

# Create MCI label
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
qids_present = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]

med_baseline['MedHx_MCI'] = 0
for qid in qids_present:
    med_baseline['MedHx_MCI'] |= (med_baseline[qid] == 1).fillna(False)

merged = med_baseline[['SubjectCode', 'MedHx_MCI']].drop_duplicates(subset=['SubjectCode'])
print(f"Base subjects (MedHx): {len(merged):,}")

# 2. Add MemTrax scores
print("\nAdding MemTrax scores...")
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
memtrax_q = apply_ashford(memtrax, accuracy_threshold=0.60)

memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
    'CorrectPCT': ['mean', 'std'],
    'CorrectResponsesRT': ['mean', 'std']
}).reset_index()

memtrax_agg.columns = ['SubjectCode', 'accuracy_mean', 'accuracy_std', 'RT_mean', 'RT_std']
memtrax_agg['CognitiveScore'] = memtrax_agg['RT_mean'] / (memtrax_agg['accuracy_mean'] + 0.01)

# MemTrax impairment (moderate threshold)
memtrax_agg['MemTrax_impaired'] = (
    (memtrax_agg['CognitiveScore'] > 1.4) | 
    (memtrax_agg['accuracy_mean'] < 0.75)
).astype(int)

merged = merged.merge(
    memtrax_agg[['SubjectCode', 'MemTrax_impaired']], 
    on='SubjectCode', 
    how='left'
)
print(f"  MemTrax data available: {merged['MemTrax_impaired'].notna().sum():,}")

# 3. Add ECOG Self-Report with ADJUSTED threshold
print("\nAdding ECOG Self-Report (threshold > 3.0)...")
try:
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
        ecog_baseline['ECOG_self_mean'] = ecog_baseline[ecog_cols].mean(axis=1)
        # ADJUSTED THRESHOLD: 3.0 instead of 2.5
        ecog_baseline['ECOG_self_impaired'] = (ecog_baseline['ECOG_self_mean'] > 3.0).astype(int)
        
        merged = merged.merge(
            ecog_baseline[['SubjectCode', 'ECOG_self_mean', 'ECOG_self_impaired']].drop_duplicates(subset=['SubjectCode']),
            on='SubjectCode',
            how='left'
        )
        print(f"  ECOG-Self data available: {merged['ECOG_self_impaired'].notna().sum():,}")
        print(f"  Mean ECOG-Self score: {merged['ECOG_self_mean'].mean():.2f}")
except:
    merged['ECOG_self_impaired'] = np.nan
    print("  No ECOG self data")

# 4. Add SP_ECOG Informant - FIXED for "sp-" prefix
print("\nAdding ECOG Informant Report (threshold > 3.0)...")
try:
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
    if 'Code' in sp_ecog.columns:
        sp_ecog = sp_ecog.rename(columns={'Code': 'SubjectCode'})
    
    # Handle "sp-m00" format
    if 'TimepointCode' in sp_ecog.columns:
        sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
        print(f"  Found {len(sp_ecog_baseline):,} informant records at baseline (sp-m00)")
    else:
        sp_ecog_baseline = sp_ecog
    
    sp_ecog_cols = [c for c in sp_ecog_baseline.columns if c.startswith('QID') and 
                    sp_ecog_baseline[c].dtype in ['float64', 'int64']]
    
    if sp_ecog_cols and len(sp_ecog_baseline) > 0:
        sp_ecog_baseline['ECOG_inf_mean'] = sp_ecog_baseline[sp_ecog_cols].mean(axis=1)
        # ADJUSTED THRESHOLD: 3.0 for informant too
        sp_ecog_baseline['ECOG_inf_impaired'] = (sp_ecog_baseline['ECOG_inf_mean'] > 3.0).astype(int)
        
        merged = merged.merge(
            sp_ecog_baseline[['SubjectCode', 'ECOG_inf_mean', 'ECOG_inf_impaired']].drop_duplicates(subset=['SubjectCode']),
            on='SubjectCode',
            how='left'
        )
        print(f"  ECOG-Informant data available: {merged['ECOG_inf_impaired'].notna().sum():,}")
        print(f"  Mean ECOG-Informant score: {merged['ECOG_inf_mean'].mean():.2f}")
    else:
        merged['ECOG_inf_impaired'] = np.nan
        print("  No valid informant data found")
except Exception as e:
    merged['ECOG_inf_impaired'] = np.nan
    print(f"  Error loading informant data: {e}")

# 5. Add demographics
print("\nAdding demographics...")
merged = enrich_demographics(DATA_DIR, merged)

print(f"\nFinal dataset: {len(merged):,} subjects")
print(f"Complete cases (all 4 measures): {merged[['MedHx_MCI', 'MemTrax_impaired', 'ECOG_self_impaired', 'ECOG_inf_impaired']].notna().all(axis=1).sum():,}")

# 6. Analysis by Age Groups
print("\n" + "="*90)
print("IMPAIRMENT RATES BY AGE GROUP (with adjusted ECOG thresholds)")
print("="*90)

age_groups = [
    ('18-44', 18, 45),
    ('45-54', 45, 55),
    ('55-64', 55, 65),
    ('65-74', 65, 75),
    ('75-84', 75, 85),
    ('85+', 85, 120)
]

print("\n%-12s %8s %12s %12s %14s %14s" % 
      ("Age Group", "N", "MedHx MCI", "MemTrax", "ECOG-Self>3", "ECOG-Inf>3"))
print("-"*85)

for label, min_age, max_age in age_groups:
    if 'Age_Baseline' in merged.columns:
        mask = (merged['Age_Baseline'] >= min_age) & (merged['Age_Baseline'] < max_age)
        subset = merged[mask]
        
        if len(subset) > 10:
            n = len(subset)
            
            medhx_pct = subset['MedHx_MCI'].mean() * 100
            
            memtrax_valid = subset['MemTrax_impaired'].dropna()
            memtrax_pct = memtrax_valid.mean() * 100 if len(memtrax_valid) > 0 else np.nan
            
            ecog_self_valid = subset['ECOG_self_impaired'].dropna()
            ecog_self_pct = ecog_self_valid.mean() * 100 if len(ecog_self_valid) > 0 else np.nan
            
            ecog_inf_valid = subset['ECOG_inf_impaired'].dropna()
            ecog_inf_pct = ecog_inf_valid.mean() * 100 if len(ecog_inf_valid) > 0 else np.nan
            
            # Format output
            memtrax_str = f"{memtrax_pct:.1f}%" if not np.isnan(memtrax_pct) else "N/A"
            ecog_self_str = f"{ecog_self_pct:.1f}%" if not np.isnan(ecog_self_pct) else "N/A"
            ecog_inf_str = f"{ecog_inf_pct:.1f}%" if not np.isnan(ecog_inf_pct) else "N/A"
            
            print("%-12s %8d %11.1f%% %11s %13s %13s" % 
                  (label, n, medhx_pct, memtrax_str, ecog_self_str, ecog_inf_str))

# Sample sizes
print("\nSample sizes by age:")
print("%-12s %8s %12s %12s %14s %14s" % 
      ("Age Group", "Total", "MedHx", "MemTrax", "ECOG-Self", "ECOG-Inf"))
print("-"*85)

for label, min_age, max_age in age_groups:
    if 'Age_Baseline' in merged.columns:
        mask = (merged['Age_Baseline'] >= min_age) & (merged['Age_Baseline'] < max_age)
        subset = merged[mask]
        if len(subset) > 10:
            n_total = len(subset)
            n_medhx = subset['MedHx_MCI'].notna().sum()
            n_memtrax = subset['MemTrax_impaired'].notna().sum()
            n_ecog_self = subset['ECOG_self_impaired'].notna().sum()
            n_ecog_inf = subset['ECOG_inf_impaired'].notna().sum()
            
            print("%-12s %8d %11d %11d %13d %13d" % 
                  (label, n_total, n_medhx, n_memtrax, n_ecog_self, n_ecog_inf))

# 7. Analysis by Education Level
print("\n" + "="*90)
print("IMPAIRMENT RATES BY EDUCATION LEVEL (with adjusted thresholds)")
print("="*90)

edu_groups = [
    ('< HS (<12)', 0, 12),
    ('HS (12)', 12, 13),
    ('Some Col (13-15)', 13, 16),
    ('Bachelor (16)', 16, 17),
    ('Graduate (17+)', 17, 30)
]

print("\n%-18s %8s %12s %12s %14s %14s" % 
      ("Education", "N", "MedHx MCI", "MemTrax", "ECOG-Self>3", "ECOG-Inf>3"))
print("-"*85)

for label, min_edu, max_edu in edu_groups:
    if 'YearsEducationUS_Converted' in merged.columns:
        if 'HS (12)' in label:
            mask = merged['YearsEducationUS_Converted'] == 12
        elif 'Bachelor (16)' in label:
            mask = merged['YearsEducationUS_Converted'] == 16
        else:
            mask = (merged['YearsEducationUS_Converted'] >= min_edu) & (merged['YearsEducationUS_Converted'] < max_edu)
        
        subset = merged[mask]
        
        if len(subset) > 10:
            n = len(subset)
            
            medhx_pct = subset['MedHx_MCI'].mean() * 100
            
            memtrax_valid = subset['MemTrax_impaired'].dropna()
            memtrax_pct = memtrax_valid.mean() * 100 if len(memtrax_valid) > 0 else np.nan
            
            ecog_self_valid = subset['ECOG_self_impaired'].dropna()
            ecog_self_pct = ecog_self_valid.mean() * 100 if len(ecog_self_valid) > 0 else np.nan
            
            ecog_inf_valid = subset['ECOG_inf_impaired'].dropna()
            ecog_inf_pct = ecog_inf_valid.mean() * 100 if len(ecog_inf_valid) > 0 else np.nan
            
            # Format output
            memtrax_str = f"{memtrax_pct:.1f}%" if not np.isnan(memtrax_pct) else "N/A"
            ecog_self_str = f"{ecog_self_pct:.1f}%" if not np.isnan(ecog_self_pct) else "N/A"
            ecog_inf_str = f"{ecog_inf_pct:.1f}%" if not np.isnan(ecog_inf_pct) else "N/A"
            
            print("%-18s %8d %11.1f%% %11s %13s %13s" % 
                  (label, n, medhx_pct, memtrax_str, ecog_self_str, ecog_inf_str))

# Sample sizes
print("\nSample sizes by education:")
print("%-18s %8s %12s %12s %14s %14s" % 
      ("Education", "Total", "MedHx", "MemTrax", "ECOG-Self", "ECOG-Inf"))
print("-"*85)

for label, min_edu, max_edu in edu_groups:
    if 'YearsEducationUS_Converted' in merged.columns:
        if 'HS (12)' in label:
            mask = merged['YearsEducationUS_Converted'] == 12
        elif 'Bachelor (16)' in label:
            mask = merged['YearsEducationUS_Converted'] == 16
        else:
            mask = (merged['YearsEducationUS_Converted'] >= min_edu) & (merged['YearsEducationUS_Converted'] < max_edu)
        
        subset = merged[mask]
        if len(subset) > 10:
            n_total = len(subset)
            n_medhx = subset['MedHx_MCI'].notna().sum()
            n_memtrax = subset['MemTrax_impaired'].notna().sum()
            n_ecog_self = subset['ECOG_self_impaired'].notna().sum()
            n_ecog_inf = subset['ECOG_inf_impaired'].notna().sum()
            
            print("%-18s %8d %11d %11d %13d %13d" % 
                  (label, n_total, n_medhx, n_memtrax, n_ecog_self, n_ecog_inf))

# 8. Summary
print("\n" + "="*90)
print("SUMMARY WITH ADJUSTED THRESHOLDS")
print("="*90)

# Overall rates
overall_medhx = merged['MedHx_MCI'].mean() * 100
overall_memtrax = merged['MemTrax_impaired'].dropna().mean() * 100 if merged['MemTrax_impaired'].notna().any() else np.nan
overall_ecog_self = merged['ECOG_self_impaired'].dropna().mean() * 100 if merged['ECOG_self_impaired'].notna().any() else np.nan
overall_ecog_inf = merged['ECOG_inf_impaired'].dropna().mean() * 100 if merged['ECOG_inf_impaired'].notna().any() else np.nan

print(f"\nOverall Impairment Rates:")
print(f"  MedHx MCI:           {overall_medhx:.1f}%")
print(f"  MemTrax (>1.4/<75%): {overall_memtrax:.1f}%" if not np.isnan(overall_memtrax) else "  MemTrax:            N/A")
print(f"  ECOG-Self (>3.0):    {overall_ecog_self:.1f}%" if not np.isnan(overall_ecog_self) else "  ECOG-Self:          N/A")
print(f"  ECOG-Informant (>3.0): {overall_ecog_inf:.1f}%" if not np.isnan(overall_ecog_inf) else "  ECOG-Informant:     N/A")

# Concordance for those with both self and informant
if 'ECOG_self_impaired' in merged.columns and 'ECOG_inf_impaired' in merged.columns:
    both_ecog = merged[['ECOG_self_impaired', 'ECOG_inf_impaired']].dropna()
    if len(both_ecog) > 0:
        agreement = (both_ecog['ECOG_self_impaired'] == both_ecog['ECOG_inf_impaired']).mean() * 100
        print(f"\nSelf vs Informant ECOG agreement: {agreement:.1f}% (N={len(both_ecog)})")
        
        # Check direction of disagreement
        self_only = ((both_ecog['ECOG_self_impaired'] == 1) & (both_ecog['ECOG_inf_impaired'] == 0)).sum()
        inf_only = ((both_ecog['ECOG_self_impaired'] == 0) & (both_ecog['ECOG_inf_impaired'] == 1)).sum()
        print(f"  Self reports impairment, informant doesn't: {self_only}")
        print(f"  Informant reports impairment, self doesn't: {inf_only}")

print("\n" + "="*90)
print("CUTOFF SPECIFICATIONS")
print("="*90)
print("""
📊 THRESHOLDS USED:
• MedHx MCI: Self-reported diagnosis (any of QID1-5,12,13,22,23 = Yes)
• MemTrax: CognitiveScore >1.4 OR Accuracy <75%
• ECOG-Self: Mean score >3.0 (adjusted from 2.5)
• ECOG-Informant: Mean score >3.0 (adjusted from 2.5)

🎯 ECOG Scale:
1 = Better or no change
2 = Questionable/occasionally
3 = Consistently a little worse
4 = Consistently much worse

Threshold >3.0 means "consistently worse" rather than just "sometimes"
""")
