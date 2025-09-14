#!/usr/bin/env python3
"""
Check SP_ECOG (informant) data availability
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*60)
print("CHECKING SP_ECOG (INFORMANT) DATA")
print("="*60)

# Load SP_ECOG
sp_ecog_path = DATA_DIR / 'BHR_SP_ECog.csv'
if not sp_ecog_path.exists():
    print(f"ERROR: {sp_ecog_path} not found!")
    exit(1)

print(f"\n✓ Found: {sp_ecog_path}")

sp_ecog = pd.read_csv(sp_ecog_path, low_memory=False)
print(f"Shape: {sp_ecog.shape}")

# Check for subject column
if 'Code' in sp_ecog.columns:
    sp_ecog = sp_ecog.rename(columns={'Code': 'SubjectCode'})

print(f"Unique subjects: {sp_ecog['SubjectCode'].nunique():,}")

# Check timepoints
if 'TimepointCode' in sp_ecog.columns:
    print(f"\nTimepoint distribution:")
    print(sp_ecog['TimepointCode'].value_counts().head(10))
    
    baseline = sp_ecog[sp_ecog['TimepointCode'] == 'm00']
    print(f"\nBaseline (m00) records: {len(baseline):,}")
    print(f"Unique baseline subjects: {baseline['SubjectCode'].nunique():,}")
else:
    baseline = sp_ecog
    print(f"No timepoint column - treating all as baseline")

# Check QID columns
qid_cols = [c for c in sp_ecog.columns if c.startswith('QID') and 
            sp_ecog[c].dtype in ['float64', 'int64']]

print(f"\nQID columns found: {len(qid_cols)}")
if qid_cols:
    print(f"Sample QIDs: {qid_cols[:5]}")
    
    # Check data completeness in baseline
    if len(baseline) > 0:
        has_any_qid = baseline[qid_cols].notna().any(axis=1)
        print(f"\nBaseline subjects with ANY QID data: {has_any_qid.sum():,}")
        
        # Calculate mean scores
        baseline['ECOG_inf_mean'] = baseline[qid_cols].mean(axis=1)
        
        # Check distribution
        print(f"\nECOG Informant score distribution:")
        print(baseline['ECOG_inf_mean'].describe())
        
        # Test different thresholds
        print(f"\nImpairment rates at different thresholds:")
        for threshold in [2.0, 2.5, 3.0, 3.5]:
            impaired = (baseline['ECOG_inf_mean'] > threshold).mean() * 100
            print(f"  > {threshold}: {impaired:.1f}%")
        
        # Compare with medical history
        med_path = DATA_DIR / 'BHR_MedicalHx.csv'
        if med_path.exists():
            med = pd.read_csv(med_path, low_memory=False)
            if 'TimepointCode' in med.columns:
                med = med[med['TimepointCode'] == 'm00']
            
            # Merge to compare
            merged = baseline[['SubjectCode', 'ECOG_inf_mean']].merge(
                med[['SubjectCode']], 
                on='SubjectCode', 
                how='inner'
            )
            
            if len(merged) > 0:
                print(f"\n✓ {len(merged):,} subjects have both informant ECOG and medical history")
            else:
                print(f"\n⚠️ No overlap between informant ECOG and medical history!")
                
                # Check subject ID format
                print(f"\nSample SP_ECOG subjects: {baseline['SubjectCode'].head(3).tolist()}")
                print(f"Sample MedHx subjects: {med['SubjectCode'].head(3).tolist()}")
else:
    print("No numeric QID columns found!")

# Also check ECOG self-report for comparison
ecog_self_path = DATA_DIR / 'BHR_EverydayCognition.csv'
if ecog_self_path.exists():
    ecog_self = pd.read_csv(ecog_self_path, low_memory=False)
    if 'Code' in ecog_self.columns:
        ecog_self = ecog_self.rename(columns={'Code': 'SubjectCode'})
    
    if 'TimepointCode' in ecog_self.columns:
        ecog_self_base = ecog_self[ecog_self['TimepointCode'] == 'm00']
    else:
        ecog_self_base = ecog_self
    
    # Check overlap
    overlap = set(baseline['SubjectCode']) & set(ecog_self_base['SubjectCode'])
    print(f"\n{len(overlap):,} subjects have BOTH self and informant ECOG")
