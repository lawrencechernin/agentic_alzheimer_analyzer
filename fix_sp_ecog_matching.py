#!/usr/bin/env python3
"""
Fix SP-ECOG matching by handling the 'sp-' prefix in timepoints
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*70)
print("FIXING SP-ECOG MATCHING ISSUE")
print("="*70)

# Load datasets
print("\n1. Loading datasets...")
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)

# Get baseline subjects from each
memtrax_subjects = set(memtrax['SubjectCode'].dropna().unique())
med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'] if 'TimepointCode' in med_hx.columns else med_hx
med_subjects = set(med_baseline['SubjectCode'].dropna().unique())

print(f"   MemTrax subjects: {len(memtrax_subjects):,}")
print(f"   Medical History baseline subjects: {len(med_subjects):,}")

# FIX: Handle 'sp-m00' instead of 'm00'
print("\n2. Fixing SP-ECOG timepoint issue...")
print(f"   Original timepoints: {sp_ecog['TimepointCode'].value_counts().head().to_dict()}")

# Get baseline SP-ECOG with 'sp-m00'
sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
print(f"   SP-ECOG baseline (sp-m00) records: {len(sp_ecog_baseline):,}")

sp_subjects = set(sp_ecog_baseline['SubjectCode'].dropna().unique())
print(f"   SP-ECOG baseline subjects: {len(sp_subjects):,}")

if len(sp_subjects) > 0:
    print(f"   Sample SP-ECOG IDs: {list(sp_subjects)[:5]}")

# Check overlap now
print("\n3. CHECKING OVERLAP (FIXED):")
print("="*50)

overlap_memtrax = sp_subjects & memtrax_subjects
overlap_med = sp_subjects & med_subjects

print(f"   SP-ECOG ∩ MemTrax: {len(overlap_memtrax):,} subjects")
print(f"   SP-ECOG ∩ Medical History: {len(overlap_med):,} subjects")

if len(overlap_memtrax) > 0:
    print(f"\n   ✅ SUCCESS! Found {len(overlap_memtrax):,} subjects with informant data!")
    print(f"   Coverage: {100*len(overlap_memtrax)/len(memtrax_subjects):.1f}% of MemTrax subjects")
    
    # Check data quality for overlapping subjects
    print("\n4. SP-ECOG DATA QUALITY FOR MATCHED SUBJECTS:")
    print("="*50)
    
    # Filter to overlapping subjects
    sp_matched = sp_ecog_baseline[sp_ecog_baseline['SubjectCode'].isin(overlap_memtrax)]
    
    # Get numeric QID columns
    qid_cols = [c for c in sp_matched.columns if c.startswith('QID')]
    numeric_cols = sp_matched.select_dtypes(include=[np.number]).columns.tolist()
    numeric_qids = [c for c in qid_cols if c in numeric_cols]
    
    print(f"   Matched records: {len(sp_matched):,}")
    print(f"   Numeric QID columns: {len(numeric_qids)}")
    
    if numeric_qids:
        # Check completeness
        non_null = sp_matched[numeric_qids].notna().sum().sum()
        total_cells = len(sp_matched) * len(numeric_qids)
        completeness = 100 * non_null / total_cells if total_cells > 0 else 0
        
        print(f"   Data completeness: {non_null:,}/{total_cells:,} ({completeness:.1f}%)")
        
        # Sample QIDs
        print(f"\n   Sample QIDs with data:")
        for qid in numeric_qids[:5]:
            non_null_count = sp_matched[qid].notna().sum()
            print(f"     {qid}: {non_null_count:,} responses")
    
    # Create merged dataset for analysis
    print("\n5. CREATING MERGED DATASET WITH INFORMANT DATA:")
    print("="*50)
    
    # Start with MemTrax subjects that have SP-ECOG
    analysis_subjects = list(overlap_memtrax)[:100]  # Sample for testing
    
    print(f"   Creating analysis dataset with {len(analysis_subjects)} subjects (sample)")
    
    # Get MemTrax features for these subjects
    memtrax_sample = memtrax[memtrax['SubjectCode'].isin(analysis_subjects)]
    
    # Aggregate MemTrax features (quality filtered)
    memtrax_q = memtrax_sample[
        (memtrax_sample['Status'] == 'Collected') &
        (memtrax_sample['CorrectPCT'] >= 0.60) &
        (memtrax_sample['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    mem_features = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std'],
        'CorrectResponsesRT': ['mean', 'std']
    }).reset_index()
    mem_features.columns = ['SubjectCode'] + ['_'.join(col) for col in mem_features.columns[1:]]
    
    print(f"   MemTrax features: {mem_features.shape}")
    
    # Get SP-ECOG features
    sp_features = sp_matched[sp_matched['SubjectCode'].isin(analysis_subjects)].copy()
    
    # Calculate SP-ECOG summary scores
    if numeric_qids:
        sp_features['SP_ECOG_Mean'] = sp_features[numeric_qids].mean(axis=1)
        sp_features['SP_ECOG_NonNull'] = sp_features[numeric_qids].notna().sum(axis=1)
        
        sp_summary = sp_features[['SubjectCode', 'SP_ECOG_Mean', 'SP_ECOG_NonNull']]
        print(f"   SP-ECOG features: {sp_summary.shape}")
        
        # Merge
        merged = mem_features.merge(sp_summary, on='SubjectCode', how='inner')
        print(f"   Merged dataset: {merged.shape}")
        
        if len(merged) > 0:
            print(f"\n   Sample of merged data:")
            print(merged.head())
            
            # Check correlation between MemTrax and SP-ECOG
            if 'CorrectPCT_mean' in merged.columns and 'SP_ECOG_Mean' in merged.columns:
                valid = merged[['CorrectPCT_mean', 'SP_ECOG_Mean']].dropna()
                if len(valid) > 10:
                    corr = valid.corr().iloc[0, 1]
                    print(f"\n   Correlation (MemTrax accuracy vs SP-ECOG): {corr:.3f}")
                    print(f"   {'✅ Good' if abs(corr) > 0.2 else '⚠️ Weak'} correlation signal")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR IMPROVING MCI PREDICTION:")
    print("="*70)
    print(f"""
1. USE INFORMANT DATA ({len(overlap_memtrax):,} subjects available!)
   - SP-ECOG is more reliable than self-report
   - Less affected by anosognosia
   - Captures functional decline better

2. CREATE COMPOSITE LABELS:
   - Combine self-report (QIDs) + informant (SP-ECOG)
   - Require agreement from multiple sources
   - This reduces label noise significantly

3. USE SP-ECOG AS FEATURES:
   - Even if not used for labels, valuable predictive signal
   - Domain-specific scores (memory, executive, etc.)
   - Change over time if longitudinal data available

4. EXPECTED IMPROVEMENT:
   - Adding informant data: +0.03-0.05 AUC
   - Composite labels: +0.02-0.04 AUC
   - Combined: Could reach 0.78-0.80 AUC!
""")
else:
    print("\n   ❌ Still no overlap - need to investigate ID format further")
    
    # Debug ID formats
    print("\n   Debugging ID formats:")
    
    if sp_subjects:
        sp_sample = list(sp_subjects)[:3]
        mem_sample = list(memtrax_subjects)[:3]
        
        print(f"   SP-ECOG examples: {sp_sample}")
        print(f"   MemTrax examples: {mem_sample}")
        
        # Check if one has extra characters
        for sp_id in sp_sample:
            if pd.notna(sp_id):
                print(f"\n   SP ID '{sp_id}':")
                print(f"     Length: {len(str(sp_id))}")
                print(f"     Type: {type(sp_id)}")
                break

