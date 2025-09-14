#!/usr/bin/env python3
"""
Compare Multiple Impairment Indicators:
1. MemTrax objective performance (RT, accuracy)
2. Medical History self-reported MCI
3. ECOG self-reported cognitive concerns
4. SP_ECOG informant-reported concerns
"""
import pandas as pd
import numpy as np
from pathlib import Path
from improvements.demographics_enrichment import enrich_demographics
from improvements.ashford_policy import apply_ashford

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*80)
print("MULTI-SOURCE COGNITIVE IMPAIRMENT COMPARISON")
print("="*80)

# 1. Load and process MemTrax data
print("\n1. Loading MemTrax Performance Data...")
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)

# Apply quality filters
memtrax_q = apply_ashford(memtrax, accuracy_threshold=0.60)
print(f"MemTrax tests after quality filter: {len(memtrax_q):,}")

# Aggregate MemTrax performance per subject
memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
    'CorrectPCT': ['mean', 'std', 'count'],
    'CorrectResponsesRT': ['mean', 'std']
}).reset_index()

# Flatten column names
memtrax_agg.columns = ['SubjectCode', 
                       'accuracy_mean', 'accuracy_std', 'test_count',
                       'RT_mean', 'RT_std']

# Create composite cognitive score (higher = worse)
memtrax_agg['CognitiveScore'] = memtrax_agg['RT_mean'] / (memtrax_agg['accuracy_mean'] + 0.01)

print(f"Subjects with MemTrax data: {len(memtrax_agg):,}")
print(f"Mean accuracy: {memtrax_agg['accuracy_mean'].mean():.1%}")
print(f"Mean RT: {memtrax_agg['RT_mean'].mean():.3f}s")

# 2. Load Medical History self-reported MCI
print("\n2. Loading Medical History Self-Reported MCI...")
med_df = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)

# Get baseline only
if 'TimepointCode' in med_df.columns:
    med_baseline = med_df[med_df['TimepointCode'] == 'm00'].copy()
else:
    med_baseline = med_df

# Create MCI label from QIDs
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
qids_present = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]

med_baseline['MedHx_MCI'] = 0
for qid in qids_present:
    med_baseline['MedHx_MCI'] |= (med_baseline[qid] == 1).fillna(False)

med_mci = med_baseline[['SubjectCode', 'MedHx_MCI']].copy()
print(f"Subjects with MedHx: {len(med_mci):,}")
print(f"Self-reported MCI: {med_mci['MedHx_MCI'].mean()*100:.1f}%")

# 3. Load ECOG self-report
print("\n3. Loading ECOG Self-Report...")
ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv', low_memory=False)

if 'Code' in ecog.columns:
    ecog = ecog.rename(columns={'Code': 'SubjectCode'})

if 'TimepointCode' in ecog.columns:
    ecog_baseline = ecog[ecog['TimepointCode'] == 'm00'].copy()
else:
    ecog_baseline = ecog

# Calculate ECOG global mean (higher = worse)
ecog_cols = [c for c in ecog_baseline.columns if c.startswith('QID') and 
             ecog_baseline[c].dtype in ['float64', 'int64']]

if ecog_cols:
    ecog_baseline['ECOG_self_mean'] = ecog_baseline[ecog_cols].mean(axis=1)
    # Define impairment as mean > 2.5 (sometimes/consistently has problems)
    ecog_baseline['ECOG_self_impaired'] = (ecog_baseline['ECOG_self_mean'] > 2.5).astype(int)
    
    ecog_self = ecog_baseline[['SubjectCode', 'ECOG_self_mean', 'ECOG_self_impaired']].copy()
    print(f"Subjects with ECOG self: {len(ecog_self):,}")
    print(f"ECOG self-reported impairment (>2.5): {ecog_self['ECOG_self_impaired'].mean()*100:.1f}%")
else:
    ecog_self = pd.DataFrame({'SubjectCode': [], 'ECOG_self_mean': [], 'ECOG_self_impaired': []})

# 4. Load SP_ECOG informant report
print("\n4. Loading SP_ECOG Informant Report...")
sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)

if 'Code' in sp_ecog.columns:
    sp_ecog = sp_ecog.rename(columns={'Code': 'SubjectCode'})

if 'TimepointCode' in sp_ecog.columns:
    sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'm00'].copy()
else:
    sp_ecog_baseline = sp_ecog

# Calculate SP_ECOG global mean
sp_ecog_cols = [c for c in sp_ecog_baseline.columns if c.startswith('QID') and 
                sp_ecog_baseline[c].dtype in ['float64', 'int64']]

if sp_ecog_cols:
    sp_ecog_baseline['ECOG_informant_mean'] = sp_ecog_baseline[sp_ecog_cols].mean(axis=1)
    sp_ecog_baseline['ECOG_informant_impaired'] = (sp_ecog_baseline['ECOG_informant_mean'] > 2.5).astype(int)
    
    ecog_informant = sp_ecog_baseline[['SubjectCode', 'ECOG_informant_mean', 'ECOG_informant_impaired']].copy()
    print(f"Subjects with ECOG informant: {len(ecog_informant):,}")
    print(f"ECOG informant-reported impairment (>2.5): {ecog_informant['ECOG_informant_impaired'].mean()*100:.1f}%")
else:
    ecog_informant = pd.DataFrame({'SubjectCode': [], 'ECOG_informant_mean': [], 'ECOG_informant_impaired': []})

# 5. Merge all indicators
print("\n5. Merging All Indicators...")
merged = memtrax_agg.merge(med_mci, on='SubjectCode', how='inner')
merged = merged.merge(ecog_self, on='SubjectCode', how='left')
merged = merged.merge(ecog_informant, on='SubjectCode', how='left')

# Add demographics
merged = enrich_demographics(DATA_DIR, merged)

print(f"Final merged dataset: {len(merged):,} subjects")

# Define MemTrax impairment thresholds (based on typical cutoffs)
# Using composite score and accuracy
merged['MemTrax_impaired'] = ((merged['CognitiveScore'] > 1.5) | 
                              (merged['accuracy_mean'] < 0.70)).astype(int)

print(f"\nMemTrax impaired (objective): {merged['MemTrax_impaired'].mean()*100:.1f}%")

# 6. Compare impairment indicators
print("\n" + "="*80)
print("6. CONCORDANCE BETWEEN INDICATORS")
print("="*80)

indicators = ['MemTrax_impaired', 'MedHx_MCI', 'ECOG_self_impaired', 'ECOG_informant_impaired']

print("\nPairwise Agreement (% of subjects with same classification):")
print("-"*60)
for i, ind1 in enumerate(indicators):
    for ind2 in indicators[i+1:]:
        if ind1 in merged.columns and ind2 in merged.columns:
            both_present = merged[[ind1, ind2]].dropna()
            if len(both_present) > 0:
                agreement = (both_present[ind1] == both_present[ind2]).mean() * 100
                correlation = both_present[ind1].corr(both_present[ind2])
                print(f"{ind1:25s} vs {ind2:25s}: {agreement:5.1f}% agree (r={correlation:.3f})")

# 7. Analyze by Age Groups
print("\n" + "="*80)
print("7. IMPAIRMENT BY AGE GROUP")
print("="*80)

age_groups = [
    ('45-54', 45, 55),
    ('55-64', 55, 65),
    ('65-74', 65, 75),
    ('75-84', 75, 85),
    ('85+', 85, 120)
]

print("\n%-10s %8s %12s %12s %12s %12s" % 
      ("Age", "N", "MemTrax%", "MedHx%", "ECOG-S%", "ECOG-I%"))
print("-"*75)

for label, min_age, max_age in age_groups:
    if 'Age_Baseline' in merged.columns:
        mask = (merged['Age_Baseline'] >= min_age) & (merged['Age_Baseline'] < max_age)
        subset = merged[mask]
        
        if len(subset) > 10:
            n = len(subset)
            memtrax_pct = subset['MemTrax_impaired'].mean() * 100
            medhx_pct = subset['MedHx_MCI'].mean() * 100
            
            ecog_self_pct = subset['ECOG_self_impaired'].mean() * 100 if 'ECOG_self_impaired' in subset else np.nan
            ecog_inf_pct = subset['ECOG_informant_impaired'].mean() * 100 if 'ECOG_informant_impaired' in subset else np.nan
            
            print("%-10s %8d %11.1f%% %11.1f%% %11.1f%% %11.1f%%" % 
                  (label, n, memtrax_pct, medhx_pct, ecog_self_pct, ecog_inf_pct))

# 8. Analyze by Education
print("\n" + "="*80)
print("8. IMPAIRMENT BY EDUCATION LEVEL")
print("="*80)

edu_groups = [
    ('< HS', 0, 12),
    ('HS', 12, 13),
    ('Some Col', 13, 16),
    ('Bachelor', 16, 17),
    ('Graduate', 17, 30)
]

print("\n%-10s %8s %12s %12s %12s %12s" % 
      ("Education", "N", "MemTrax%", "MedHx%", "ECOG-S%", "ECOG-I%"))
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
            
            ecog_self_pct = subset['ECOG_self_impaired'].mean() * 100 if 'ECOG_self_impaired' in subset else np.nan
            ecog_inf_pct = subset['ECOG_informant_impaired'].mean() * 100 if 'ECOG_informant_impaired' in subset else np.nan
            
            print("%-10s %8d %11.1f%% %11.1f%% %11.1f%% %11.1f%%" % 
                  (label, n, memtrax_pct, medhx_pct, ecog_self_pct, ecog_inf_pct))

# 9. Find Hidden Impairment (MemTrax+ but MedHx-)
print("\n" + "="*80)
print("9. HIDDEN IMPAIRMENT ANALYSIS")
print("="*80)

# Cases where MemTrax shows impairment but self-report doesn't
hidden_impaired = merged[(merged['MemTrax_impaired'] == 1) & (merged['MedHx_MCI'] == 0)]
print(f"\nHidden impairment (MemTrax+ but MedHx-): {len(hidden_impaired):,} ({len(hidden_impaired)/len(merged)*100:.1f}%)")

# Analyze hidden impairment by education
if len(hidden_impaired) > 0 and 'YearsEducationUS_Converted' in hidden_impaired.columns:
    print("\nHidden impairment by education:")
    for label, min_edu, max_edu in edu_groups:
        if label == 'HS':
            mask_all = merged['YearsEducationUS_Converted'] == 12
            mask_hidden = hidden_impaired['YearsEducationUS_Converted'] == 12
        elif label == 'Bachelor':
            mask_all = merged['YearsEducationUS_Converted'] == 16
            mask_hidden = hidden_impaired['YearsEducationUS_Converted'] == 16
        else:
            mask_all = (merged['YearsEducationUS_Converted'] >= min_edu) & (merged['YearsEducationUS_Converted'] < max_edu)
            mask_hidden = (hidden_impaired['YearsEducationUS_Converted'] >= min_edu) & (hidden_impaired['YearsEducationUS_Converted'] < max_edu)
        
        n_total = mask_all.sum()
        n_hidden = mask_hidden.sum()
        
        if n_total > 0:
            pct_hidden = n_hidden / n_total * 100
            print(f"  {label:10s}: {n_hidden:4d}/{n_total:5d} ({pct_hidden:5.1f}%) have hidden impairment")

# 10. Summary insights
print("\n" + "="*80)
print("10. KEY INSIGHTS")
print("="*80)

print("""
🔍 What we're seeing:
- MemTrax (objective) vs MedHx (self-report) discordance reveals hidden impairment
- ECOG informant reports may be more reliable than self-reports
- Higher education groups likely have more hidden impairment

📊 Expected patterns:
- MemTrax should catch more impairment than self-report (cognitive reserve masking)
- Informant ECOG should align better with MemTrax than self-ECOG
- Hidden impairment should be higher in educated groups
""")

# Calculate overall statistics
memtrax_rate = merged['MemTrax_impaired'].mean() * 100
medhx_rate = merged['MedHx_MCI'].mean() * 100

print(f"\nOverall Impairment Rates:")
print(f"  MemTrax (objective):  {memtrax_rate:.1f}%")
print(f"  MedHx (self-report):  {medhx_rate:.1f}%")

if memtrax_rate > medhx_rate:
    print(f"  → MemTrax detects {memtrax_rate - medhx_rate:.1f}% MORE impairment than self-report!")
    print(f"  → This hidden impairment is likely due to cognitive reserve and under-reporting")
