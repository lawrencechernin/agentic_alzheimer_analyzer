#!/usr/bin/env python3
"""
Fixed: Comprehensive Impairment Comparison by Demographics
Shows % impaired across 4 measures: MedHx MCI, MemTrax, ECOG-Self, ECOG-Informant
"""
import pandas as pd
import numpy as np
from pathlib import Path
from improvements.demographics_enrichment import enrich_demographics
from improvements.ashford_policy import apply_ashford

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*90)
print("COMPREHENSIVE IMPAIRMENT COMPARISON ACROSS 4 MEASURES")
print("="*90)

# 1. Load Medical History as BASE (has most subjects)
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

# Start with medical history subjects
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

# 3. Add ECOG Self-Report
print("\nAdding ECOG Self-Report...")
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
        ecog_baseline['ECOG_self_impaired'] = (ecog_baseline['ECOG_self_mean'] > 2.5).astype(int)
        
        merged = merged.merge(
            ecog_baseline[['SubjectCode', 'ECOG_self_impaired']].drop_duplicates(subset=['SubjectCode']),
            on='SubjectCode',
            how='left'
        )
        print(f"  ECOG-Self data available: {merged['ECOG_self_impaired'].notna().sum():,}")
except:
    merged['ECOG_self_impaired'] = np.nan
    print("  No ECOG self data")

# 4. Add SP_ECOG Informant
print("\nAdding ECOG Informant Report...")
try:
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
    if 'Code' in sp_ecog.columns:
        sp_ecog = sp_ecog.rename(columns={'Code': 'SubjectCode'})
    if 'TimepointCode' in sp_ecog.columns:
        sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'm00'].copy()
    else:
        sp_ecog_baseline = sp_ecog
    
    sp_ecog_cols = [c for c in sp_ecog_baseline.columns if c.startswith('QID') and 
                    sp_ecog_baseline[c].dtype in ['float64', 'int64']]
    
    if sp_ecog_cols and len(sp_ecog_baseline) > 0:
        sp_ecog_baseline['ECOG_inf_mean'] = sp_ecog_baseline[sp_ecog_cols].mean(axis=1)
        sp_ecog_baseline['ECOG_inf_impaired'] = (sp_ecog_baseline['ECOG_inf_mean'] > 2.5).astype(int)
        
        merged = merged.merge(
            sp_ecog_baseline[['SubjectCode', 'ECOG_inf_impaired']].drop_duplicates(subset=['SubjectCode']),
            on='SubjectCode',
            how='left'
        )
        print(f"  ECOG-Informant data available: {merged['ECOG_inf_impaired'].notna().sum():,}")
    else:
        merged['ECOG_inf_impaired'] = np.nan
        print("  No ECOG informant data")
except:
    merged['ECOG_inf_impaired'] = np.nan
    print("  No ECOG informant data")

# 5. Add demographics
print("\nAdding demographics...")
merged = enrich_demographics(DATA_DIR, merged)

print(f"\nFinal dataset: {len(merged):,} subjects")
print(f"Complete cases (all 4 measures): {merged[['MedHx_MCI', 'MemTrax_impaired', 'ECOG_self_impaired', 'ECOG_inf_impaired']].notna().all(axis=1).sum():,}")

# 6. Analysis by Age Groups
print("\n" + "="*90)
print("IMPAIRMENT RATES BY AGE GROUP")
print("="*90)

age_groups = [
    ('18-44', 18, 45),
    ('45-54', 45, 55),
    ('55-64', 55, 65),
    ('65-74', 65, 75),
    ('75-84', 75, 85),
    ('85+', 85, 120)
]

print("\n%-12s %8s %12s %12s %12s %12s" % 
      ("Age Group", "N", "MedHx MCI", "MemTrax", "ECOG-Self", "ECOG-Inf"))
print("-"*75)

for label, min_age, max_age in age_groups:
    if 'Age_Baseline' in merged.columns:
        mask = (merged['Age_Baseline'] >= min_age) & (merged['Age_Baseline'] < max_age)
        subset = merged[mask]
        
        if len(subset) > 10:
            n = len(subset)
            
            # Calculate % impaired (handling NaN properly)
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
            
            print("%-12s %8d %11.1f%% %11s %11s %11s" % 
                  (label, n, medhx_pct, memtrax_str, ecog_self_str, ecog_inf_str))

# Show sample sizes for each measure
print("\nSample sizes by age:")
print("%-12s %8s %12s %12s %12s %12s" % 
      ("Age Group", "Total", "MedHx", "MemTrax", "ECOG-Self", "ECOG-Inf"))
print("-"*75)

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
            
            print("%-12s %8d %11d %11d %11d %11d" % 
                  (label, n_total, n_medhx, n_memtrax, n_ecog_self, n_ecog_inf))

# 7. Analysis by Education Level
print("\n" + "="*90)
print("IMPAIRMENT RATES BY EDUCATION LEVEL")
print("="*90)

edu_groups = [
    ('< HS (<12)', 0, 12),
    ('HS (12)', 12, 13),
    ('Some Col (13-15)', 13, 16),
    ('Bachelor (16)', 16, 17),
    ('Graduate (17+)', 17, 30)
]

print("\n%-18s %8s %12s %12s %12s %12s" % 
      ("Education", "N", "MedHx MCI", "MemTrax", "ECOG-Self", "ECOG-Inf"))
print("-"*75)

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
            
            print("%-18s %8d %11.1f%% %11s %11s %11s" % 
                  (label, n, medhx_pct, memtrax_str, ecog_self_str, ecog_inf_str))

# Show sample sizes
print("\nSample sizes by education:")
print("%-18s %8s %12s %12s %12s %12s" % 
      ("Education", "Total", "MedHx", "MemTrax", "ECOG-Self", "ECOG-Inf"))
print("-"*75)

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
            
            print("%-18s %8d %11d %11d %11d %11d" % 
                  (label, n_total, n_medhx, n_memtrax, n_ecog_self, n_ecog_inf))

# 8. Key Insights
print("\n" + "="*90)
print("KEY INSIGHTS")
print("="*90)

# Overall rates
overall_medhx = merged['MedHx_MCI'].mean() * 100
overall_memtrax = merged['MemTrax_impaired'].dropna().mean() * 100 if merged['MemTrax_impaired'].notna().any() else np.nan
overall_ecog_self = merged['ECOG_self_impaired'].dropna().mean() * 100 if merged['ECOG_self_impaired'].notna().any() else np.nan
overall_ecog_inf = merged['ECOG_inf_impaired'].dropna().mean() * 100 if merged['ECOG_inf_impaired'].notna().any() else np.nan

print(f"\nOverall Impairment Rates:")
print(f"  MedHx MCI:      {overall_medhx:.1f}%")
print(f"  MemTrax:        {overall_memtrax:.1f}%" if not np.isnan(overall_memtrax) else "  MemTrax:        N/A")
print(f"  ECOG-Self:      {overall_ecog_self:.1f}%" if not np.isnan(overall_ecog_self) else "  ECOG-Self:      N/A")
print(f"  ECOG-Informant: {overall_ecog_inf:.1f}%" if not np.isnan(overall_ecog_inf) else "  ECOG-Informant: N/A")

print("\n" + "="*90)
print("INTERPRETATION GUIDE")
print("="*90)

print("""
📊 CUTOFFS USED:
• MedHx MCI: Self-reported diagnosis (QID1-5, 12, 13, 22, 23)
• MemTrax: CogScore >1.4 OR Accuracy <75% (moderate threshold)
• ECOG-Self: Mean >2.5 (sometimes/consistently has problems)
• ECOG-Informant: Mean >2.5 (informant observes problems)

🔍 WHAT TO LOOK FOR:
1. Age gradient: All measures should increase with age
2. Education gradient: MemTrax impairment higher in low education
3. Discordance: ECOG-Self ~50% (too sensitive), others ~5-10%
4. Missing data: ECOG-Informant usually absent

⚠️ KEY PATTERNS:
• If MemTrax > MedHx → Hidden impairment (objective worse than subjective)
• If MedHx > MemTrax → Over-reporting or cognitive reserve masking
• ECOG-Self high across all groups → Captures anxiety, not just impairment
""")
