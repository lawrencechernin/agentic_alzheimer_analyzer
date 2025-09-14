#!/usr/bin/env python3
"""
Comprehensive Impairment Comparison by Demographics
Shows % impaired across 4 measures: MedHx MCI, MemTrax, ECOG-Self, ECOG-Informant
Bucketed by age groups and education levels
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

# 1. Load and process MemTrax
print("\nLoading MemTrax data...")
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
memtrax_q = apply_ashford(memtrax, accuracy_threshold=0.60)

# Aggregate MemTrax performance
memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
    'CorrectPCT': ['mean', 'std'],
    'CorrectResponsesRT': ['mean', 'std']
}).reset_index()

memtrax_agg.columns = ['SubjectCode', 'accuracy_mean', 'accuracy_std', 'RT_mean', 'RT_std']
memtrax_agg['CognitiveScore'] = memtrax_agg['RT_mean'] / (memtrax_agg['accuracy_mean'] + 0.01)

# Define MemTrax impairment (moderate threshold - best balance)
memtrax_agg['MemTrax_impaired'] = (
    (memtrax_agg['CognitiveScore'] > 1.4) | 
    (memtrax_agg['accuracy_mean'] < 0.75)
).astype(int)

print(f"MemTrax subjects: {len(memtrax_agg):,}")

# 2. Load Medical History MCI
print("\nLoading Medical History...")
med_df = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
if 'TimepointCode' in med_df.columns:
    med_baseline = med_df[med_df['TimepointCode'] == 'm00'].copy()
else:
    med_baseline = med_df

# Create MCI label from cognitive QIDs
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
qids_present = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]

med_baseline['MedHx_MCI'] = 0
for qid in qids_present:
    med_baseline['MedHx_MCI'] |= (med_baseline[qid] == 1).fillna(False)

med_mci = med_baseline[['SubjectCode', 'MedHx_MCI']].copy()
print(f"MedHx subjects: {len(med_mci):,}")

# 3. Load ECOG Self-Report
print("\nLoading ECOG Self-Report...")
try:
    ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv', low_memory=False)
    if 'Code' in ecog.columns:
        ecog = ecog.rename(columns={'Code': 'SubjectCode'})
    if 'TimepointCode' in ecog.columns:
        ecog_baseline = ecog[ecog['TimepointCode'] == 'm00'].copy()
    else:
        ecog_baseline = ecog
    
    # Calculate mean across ECOG items
    ecog_cols = [c for c in ecog_baseline.columns if c.startswith('QID') and 
                 ecog_baseline[c].dtype in ['float64', 'int64']]
    
    if ecog_cols:
        ecog_baseline['ECOG_self_mean'] = ecog_baseline[ecog_cols].mean(axis=1)
        # Using 2.5 cutoff (sometimes/consistently has problems)
        ecog_baseline['ECOG_self_impaired'] = (ecog_baseline['ECOG_self_mean'] > 2.5).astype(int)
        ecog_self = ecog_baseline[['SubjectCode', 'ECOG_self_mean', 'ECOG_self_impaired']].copy()
        print(f"ECOG-Self subjects: {len(ecog_self):,}")
    else:
        ecog_self = pd.DataFrame()
        print("No ECOG self data found")
except:
    ecog_self = pd.DataFrame()
    print("Error loading ECOG self data")

# 4. Load SP_ECOG Informant Report
print("\nLoading ECOG Informant Report...")
try:
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
    if 'Code' in sp_ecog.columns:
        sp_ecog = sp_ecog.rename(columns={'Code': 'SubjectCode'})
    if 'TimepointCode' in sp_ecog.columns:
        sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'm00'].copy()
    else:
        sp_ecog_baseline = sp_ecog
    
    # Calculate mean across SP_ECOG items
    sp_ecog_cols = [c for c in sp_ecog_baseline.columns if c.startswith('QID') and 
                    sp_ecog_baseline[c].dtype in ['float64', 'int64']]
    
    if sp_ecog_cols:
        sp_ecog_baseline['ECOG_inf_mean'] = sp_ecog_baseline[sp_ecog_cols].mean(axis=1)
        sp_ecog_baseline['ECOG_inf_impaired'] = (sp_ecog_baseline['ECOG_inf_mean'] > 2.5).astype(int)
        ecog_inf = sp_ecog_baseline[['SubjectCode', 'ECOG_inf_mean', 'ECOG_inf_impaired']].copy()
        print(f"ECOG-Informant subjects: {len(ecog_inf):,}")
    else:
        ecog_inf = pd.DataFrame()
        print("No ECOG informant data found")
except:
    ecog_inf = pd.DataFrame()
    print("Error loading ECOG informant data")

# 5. Merge all data sources
print("\n" + "-"*90)
print("Merging all data sources...")
merged = memtrax_agg.merge(med_mci, on='SubjectCode', how='outer')

if not ecog_self.empty:
    merged = merged.merge(ecog_self[['SubjectCode', 'ECOG_self_impaired']], 
                         on='SubjectCode', how='left')
else:
    merged['ECOG_self_impaired'] = np.nan

if not ecog_inf.empty:
    merged = merged.merge(ecog_inf[['SubjectCode', 'ECOG_inf_impaired']], 
                         on='SubjectCode', how='left')
else:
    merged['ECOG_inf_impaired'] = np.nan

# Add demographics
merged = enrich_demographics(DATA_DIR, merged)

print(f"Total merged subjects: {len(merged):,}")
print(f"Subjects with all 4 measures: {merged[['MemTrax_impaired', 'MedHx_MCI', 'ECOG_self_impaired', 'ECOG_inf_impaired']].notna().all(axis=1).sum():,}")

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
print("-"*90)

age_results = []
for label, min_age, max_age in age_groups:
    if 'Age_Baseline' in merged.columns:
        mask = (merged['Age_Baseline'] >= min_age) & (merged['Age_Baseline'] < max_age)
        subset = merged[mask]
        
        if len(subset) > 10:  # Need minimum sample size
            n = len(subset)
            
            # Calculate % impaired for each measure
            medhx_pct = subset['MedHx_MCI'].mean() * 100 if 'MedHx_MCI' in subset else np.nan
            memtrax_pct = subset['MemTrax_impaired'].mean() * 100 if 'MemTrax_impaired' in subset else np.nan
            ecog_self_pct = subset['ECOG_self_impaired'].mean() * 100 if 'ECOG_self_impaired' in subset else np.nan
            ecog_inf_pct = subset['ECOG_inf_impaired'].mean() * 100 if 'ECOG_inf_impaired' in subset else np.nan
            
            # Count valid N for each measure
            n_medhx = subset['MedHx_MCI'].notna().sum()
            n_memtrax = subset['MemTrax_impaired'].notna().sum()
            n_ecog_self = subset['ECOG_self_impaired'].notna().sum()
            n_ecog_inf = subset['ECOG_inf_impaired'].notna().sum()
            
            print("%-12s %8d %11.1f%% %11.1f%% %11.1f%% %11.1f%%" % 
                  (label, n, medhx_pct, memtrax_pct, ecog_self_pct, ecog_inf_pct))
            
            # Store for summary
            age_results.append({
                'group': label,
                'n': n,
                'medhx': medhx_pct,
                'memtrax': memtrax_pct,
                'ecog_self': ecog_self_pct,
                'ecog_inf': ecog_inf_pct
            })

# Print sample sizes for each measure
print("\n%-12s %8s %12s %12s %12s %12s" % 
      ("", "", "(N valid)", "(N valid)", "(N valid)", "(N valid)"))
for label, min_age, max_age in age_groups:
    if 'Age_Baseline' in merged.columns:
        mask = (merged['Age_Baseline'] >= min_age) & (merged['Age_Baseline'] < max_age)
        subset = merged[mask]
        if len(subset) > 10:
            n_medhx = subset['MedHx_MCI'].notna().sum()
            n_memtrax = subset['MemTrax_impaired'].notna().sum()
            n_ecog_self = subset['ECOG_self_impaired'].notna().sum()
            n_ecog_inf = subset['ECOG_inf_impaired'].notna().sum()
            
            print("%-12s %8s %12d %12d %12d %12d" % 
                  (label, "", n_medhx, n_memtrax, n_ecog_self, n_ecog_inf))

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
print("-"*90)

edu_results = []
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
            
            medhx_pct = subset['MedHx_MCI'].mean() * 100 if 'MedHx_MCI' in subset else np.nan
            memtrax_pct = subset['MemTrax_impaired'].mean() * 100 if 'MemTrax_impaired' in subset else np.nan
            ecog_self_pct = subset['ECOG_self_impaired'].mean() * 100 if 'ECOG_self_impaired' in subset else np.nan
            ecog_inf_pct = subset['ECOG_inf_impaired'].mean() * 100 if 'ECOG_inf_impaired' in subset else np.nan
            
            print("%-18s %8d %11.1f%% %11.1f%% %11.1f%% %11.1f%%" % 
                  (label, n, medhx_pct, memtrax_pct, ecog_self_pct, ecog_inf_pct))
            
            edu_results.append({
                'group': label,
                'n': n,
                'medhx': medhx_pct,
                'memtrax': memtrax_pct,
                'ecog_self': ecog_self_pct,
                'ecog_inf': ecog_inf_pct
            })

# Print sample sizes
print("\n%-18s %8s %12s %12s %12s %12s" % 
      ("", "", "(N valid)", "(N valid)", "(N valid)", "(N valid)"))
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
            n_medhx = subset['MedHx_MCI'].notna().sum()
            n_memtrax = subset['MemTrax_impaired'].notna().sum()
            n_ecog_self = subset['ECOG_self_impaired'].notna().sum()
            n_ecog_inf = subset['ECOG_inf_impaired'].notna().sum()
            
            print("%-18s %8s %12d %12d %12d %12d" % 
                  (label, "", n_medhx, n_memtrax, n_ecog_self, n_ecog_inf))

# 8. Summary Statistics
print("\n" + "="*90)
print("SUMMARY INSIGHTS")
print("="*90)

# Overall rates
overall_medhx = merged['MedHx_MCI'].mean() * 100
overall_memtrax = merged['MemTrax_impaired'].mean() * 100
overall_ecog_self = merged['ECOG_self_impaired'].mean() * 100
overall_ecog_inf = merged['ECOG_inf_impaired'].mean() * 100

print(f"\nOverall Impairment Rates:")
print(f"  MedHx MCI:      {overall_medhx:.1f}%")
print(f"  MemTrax:        {overall_memtrax:.1f}%")
print(f"  ECOG-Self:      {overall_ecog_self:.1f}%")
print(f"  ECOG-Informant: {overall_ecog_inf:.1f}%")

# Concordance between measures
print("\nConcordance Between Measures (% agreement):")
measures = [
    ('MedHx_MCI', 'MedHx MCI'),
    ('MemTrax_impaired', 'MemTrax'),
    ('ECOG_self_impaired', 'ECOG-Self'),
    ('ECOG_inf_impaired', 'ECOG-Inf')
]

for i, (col1, name1) in enumerate(measures):
    for col2, name2 in measures[i+1:]:
        if col1 in merged.columns and col2 in merged.columns:
            both_valid = merged[[col1, col2]].dropna()
            if len(both_valid) > 0:
                agreement = (both_valid[col1] == both_valid[col2]).mean() * 100
                correlation = both_valid[col1].corr(both_valid[col2])
                print(f"  {name1:12s} vs {name2:12s}: {agreement:5.1f}% agree (r={correlation:.3f})")

# Key patterns
print("\n" + "="*90)
print("KEY PATTERNS")
print("="*90)

print("""
📊 Age Patterns:
- MedHx MCI and MemTrax should increase with age
- ECOG-Self may be high across all ages (captures anxiety)
- ECOG-Informant most reliable but often missing

�� Education Patterns:
- Lower education → Higher MemTrax impairment (less cognitive reserve)
- Higher education → May have more MedHx MCI (worried well)
- ECOG-Self likely similar across education (subjective concerns)

⚠️ Expected Discordances:
- ECOG-Self ~50% (too sensitive, captures normal aging concerns)
- MemTrax < MedHx in educated (cognitive reserve compensates)
- MemTrax > MedHx in less educated (performance reveals hidden impairment)
""")

# Data completeness
print("\nData Completeness:")
for col, name in measures:
    if col in merged.columns:
        n_valid = merged[col].notna().sum()
        pct_valid = n_valid / len(merged) * 100
        print(f"  {name:15s}: {n_valid:6,} / {len(merged):6,} ({pct_valid:.1f}%)")
