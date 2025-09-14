#!/usr/bin/env python3
"""
Debug data merge issue
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("DEBUGGING DATA MERGE ISSUE")
print("="*60)

# Load data
med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)

print(f"\n1. Raw data shapes:")
print(f"   Medical History: {med_hx.shape}")
print(f"   SP-ECOG: {sp_ecog.shape}")

# Check for duplicates
print(f"\n2. Checking for duplicates:")
print(f"   Med History unique subjects: {med_hx['SubjectCode'].nunique():,}")
print(f"   Med History total rows: {len(med_hx):,}")

if 'TimepointCode' in med_hx.columns:
    print(f"   Med History timepoints: {med_hx['TimepointCode'].value_counts().head()}")
    med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].copy()
    print(f"   Med History baseline rows: {len(med_baseline):,}")
    print(f"   Med History baseline unique subjects: {med_baseline['SubjectCode'].nunique():,}")
    
    # Check for duplicates at baseline
    dup_subjects = med_baseline['SubjectCode'].value_counts()
    dups = dup_subjects[dup_subjects > 1]
    if len(dups) > 0:
        print(f"\n   ⚠️ Found {len(dups)} subjects with duplicate baseline records!")
        print(f"   Example duplicates: {dups.head()}")
        
        # Example of duplicate
        example_dup = dups.index[0]
        dup_records = med_baseline[med_baseline['SubjectCode'] == example_dup]
        print(f"\n   Example duplicate subject {example_dup}:")
        print(f"   Number of records: {len(dup_records)}")

# SP-ECOG
sp_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
print(f"\n3. SP-ECOG baseline:")
print(f"   SP-ECOG baseline rows: {len(sp_baseline):,}")
print(f"   SP-ECOG baseline unique subjects: {sp_baseline['SubjectCode'].nunique():,}")

sp_dups = sp_baseline['SubjectCode'].value_counts()
sp_dups_multi = sp_dups[sp_dups > 1]
if len(sp_dups_multi) > 0:
    print(f"   ⚠️ Found {len(sp_dups_multi)} subjects with duplicate SP-ECOG baseline!")

# Test merge
print(f"\n4. Testing merge:")
if 'TimepointCode' in med_hx.columns:
    test_data = med_baseline[['SubjectCode']].copy()
else:
    test_data = med_hx[['SubjectCode']].copy()

print(f"   Starting with: {len(test_data):,} medical history records")

# Merge with SP-ECOG
test_merge = test_data.merge(
    sp_baseline[['SubjectCode']], 
    on='SubjectCode', 
    how='left',
    indicator=True
)

print(f"   After merge: {len(test_merge):,} records")
print(f"   Merge indicator:")
print(test_merge['_merge'].value_counts())

# Calculate expected size
n_med = med_baseline['SubjectCode'].nunique() if 'TimepointCode' in med_hx.columns else med_hx['SubjectCode'].nunique()
n_sp = sp_baseline['SubjectCode'].nunique()
overlap = len(set(med_baseline['SubjectCode'].unique()) & set(sp_baseline['SubjectCode'].unique())) if 'TimepointCode' in med_hx.columns else 0

print(f"\n5. Expected vs Actual:")
print(f"   Medical unique subjects: {n_med:,}")
print(f"   SP-ECOG unique subjects: {n_sp:,}") 
print(f"   Overlap: {overlap:,}")
print(f"   Expected merge size: ~{n_med:,}")
print(f"   Actual merge size: {len(test_merge):,}")

if len(test_merge) > n_med * 1.5:
    print(f"\n   ❌ CARTESIAN PRODUCT DETECTED!")
    print(f"   Merge created {len(test_merge)/n_med:.1f}x more rows than expected")
    
print("\nSOLUTION: Drop duplicates before merging!")
print("  med_baseline = med_baseline.drop_duplicates(subset=['SubjectCode'])")
print("  sp_baseline = sp_baseline.drop_duplicates(subset=['SubjectCode'])")

