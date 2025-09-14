#!/usr/bin/env python3
"""
Investigate SP-ECOG (informant) data and why it's not matching
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*70)
print("INVESTIGATING SP-ECOG INFORMANT DATA MATCHING ISSUE")
print("="*70)

# 1. Load MemTrax to get subject IDs
print("\n1. Loading MemTrax subjects...")
memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
memtrax_subjects = set(memtrax['SubjectCode'].unique())
print(f"   MemTrax subjects: {len(memtrax_subjects):,}")
print(f"   Sample IDs: {list(memtrax_subjects)[:5]}")

# 2. Load Medical History
print("\n2. Loading Medical History subjects...")
med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
if 'TimepointCode' in med_hx.columns:
    med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
med_subjects = set(med_hx['SubjectCode'].unique())
print(f"   Medical History subjects: {len(med_subjects):,}")
print(f"   Sample IDs: {list(med_subjects)[:5]}")

# 3. Check SP-ECOG file existence
print("\n3. Checking SP-ECOG file...")
sp_ecog_path = DATA_DIR / 'BHR_SP_ECog.csv'
if not sp_ecog_path.exists():
    print(f"   ❌ ERROR: {sp_ecog_path} not found!")
    
    # Check for alternative names
    print("\n   Searching for alternative ECOG files...")
    for file in DATA_DIR.glob("*ECOG*.csv"):
        print(f"   Found: {file.name}")
    for file in DATA_DIR.glob("*ECog*.csv"):
        print(f"   Found: {file.name}")
    for file in DATA_DIR.glob("*ecog*.csv"):
        print(f"   Found: {file.name}")
    exit(1)

print(f"   ✓ Found: {sp_ecog_path.name}")
sp_ecog = pd.read_csv(sp_ecog_path, low_memory=False)
print(f"   Shape: {sp_ecog.shape}")
print(f"   Columns: {list(sp_ecog.columns[:10])}")

# 4. Check subject identifier column
print("\n4. Checking subject identifier columns...")
subject_cols = ['SubjectCode', 'Code', 'Subject', 'ID', 'SubjectID', 
                'ParticipantID', 'Participant', 'StudyID']

found_col = None
for col in subject_cols:
    if col in sp_ecog.columns:
        found_col = col
        print(f"   Found subject column: '{col}'")
        break

if not found_col:
    print("   ❌ No standard subject identifier found!")
    print(f"   Available columns: {list(sp_ecog.columns)}")
    
    # Try to find any column with 'subject' or 'code' in name
    for col in sp_ecog.columns:
        if 'subject' in col.lower() or 'code' in col.lower() or 'id' in col.lower():
            print(f"   Potential ID column: '{col}'")
            print(f"   Sample values: {sp_ecog[col].dropna().head().tolist()}")
    exit(1)

# Rename to SubjectCode for consistency
if found_col != 'SubjectCode':
    sp_ecog = sp_ecog.rename(columns={found_col: 'SubjectCode'})
    print(f"   Renamed '{found_col}' to 'SubjectCode'")

# 5. Check SP-ECOG subjects
sp_subjects_raw = set(sp_ecog['SubjectCode'].dropna().unique())
print(f"\n5. SP-ECOG subjects (raw): {len(sp_subjects_raw):,}")
print(f"   Sample IDs: {list(sp_subjects_raw)[:5]}")

# 6. Check for timepoints
if 'TimepointCode' in sp_ecog.columns:
    print(f"\n6. SP-ECOG timepoints:")
    print(sp_ecog['TimepointCode'].value_counts().head())
    
    # Filter to baseline
    sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'm00']
    sp_subjects = set(sp_ecog_baseline['SubjectCode'].dropna().unique())
    print(f"\n   Baseline (m00) subjects: {len(sp_subjects):,}")
else:
    sp_subjects = sp_subjects_raw
    print("\n6. No timepoint column - using all records")

# 7. Check overlap
print("\n7. CHECKING OVERLAP:")
print("="*50)

# SP-ECOG ∩ MemTrax
overlap_memtrax = sp_subjects & memtrax_subjects
print(f"   SP-ECOG ∩ MemTrax: {len(overlap_memtrax):,} subjects")
if len(overlap_memtrax) == 0:
    print("   ⚠️ NO OVERLAP - checking ID format differences...")
    
    # Check format differences
    sp_sample = list(sp_subjects)[:5] if sp_subjects else []
    mem_sample = list(memtrax_subjects)[:5]
    
    print(f"\n   SP-ECOG ID format: {sp_sample}")
    print(f"   MemTrax ID format: {mem_sample}")
    
    # Check if IDs differ by prefix/suffix
    if sp_sample and mem_sample:
        sp_id = str(sp_sample[0])
        mem_id = str(mem_sample[0])
        
        print(f"\n   SP-ECOG ID length: {len(sp_id)}")
        print(f"   MemTrax ID length: {len(mem_id)}")
        
        # Check for numeric vs string differences
        try:
            sp_numeric = [int(str(s).replace('BHR', '').replace('bhr', '')) for s in sp_sample]
            print(f"   SP-ECOG as numbers: {sp_numeric}")
        except:
            pass
            
        try:
            mem_numeric = [int(str(s).replace('BHR', '').replace('bhr', '')) for s in mem_sample]
            print(f"   MemTrax as numbers: {mem_numeric}")
        except:
            pass

# SP-ECOG ∩ Medical History
overlap_med = sp_subjects & med_subjects
print(f"\n   SP-ECOG ∩ Medical History: {len(overlap_med):,} subjects")

# 8. Check data quality
print("\n8. SP-ECOG DATA QUALITY:")
print("="*50)

# Check for numeric QID columns
qid_cols = [c for c in sp_ecog.columns if c.startswith('QID')]
numeric_cols = sp_ecog.select_dtypes(include=[np.number]).columns.tolist()
numeric_qids = [c for c in qid_cols if c in numeric_cols]

print(f"   Total columns: {len(sp_ecog.columns)}")
print(f"   QID columns: {len(qid_cols)}")
print(f"   Numeric columns: {len(numeric_cols)}")
print(f"   Numeric QIDs: {len(numeric_qids)}")

if numeric_qids:
    print(f"\n   Sample numeric QIDs: {numeric_qids[:5]}")
    
    # Check non-null counts
    if 'TimepointCode' in sp_ecog.columns:
        data_check = sp_ecog[sp_ecog['TimepointCode'] == 'm00']
    else:
        data_check = sp_ecog
    
    non_null = data_check[numeric_qids].notna().sum().sum()
    total_cells = len(data_check) * len(numeric_qids)
    print(f"\n   Data completeness: {non_null:,}/{total_cells:,} ({100*non_null/total_cells:.1f}%)")

# 9. Recommendations
print("\n" + "="*70)
print("DIAGNOSIS & RECOMMENDATIONS:")
print("="*70)

if len(overlap_memtrax) == 0:
    print("""
❌ PROBLEM: Subject IDs don't match between SP-ECOG and MemTrax

LIKELY CAUSES:
1. Different ID formats (numeric vs string)
2. Missing prefixes/suffixes (e.g., 'BHR' prefix)
3. Different ID systems entirely
4. SP-ECOG might be for different cohort

SOLUTIONS TO TRY:
1. Standardize ID formats (strip prefixes, convert to same type)
2. Create mapping table if systematic difference
3. Check if other ID columns exist for linking
4. Use demographic matching as fallback
""")
else:
    print(f"""
✅ GOOD NEWS: {len(overlap_memtrax):,} subjects have SP-ECOG data!

This informant data could significantly improve MCI detection.
SP-ECOG is more reliable than self-report ECOG.

Next steps:
1. Extract SP-ECOG features for these subjects
2. Compare with self-report to identify discrepancies
3. Use as primary or validation labels
""")

print("\nFINAL SUMMARY:")
print(f"  • MemTrax subjects: {len(memtrax_subjects):,}")
print(f"  • SP-ECOG subjects: {len(sp_subjects):,}")
print(f"  • Overlap: {len(overlap_memtrax):,} ({100*len(overlap_memtrax)/len(memtrax_subjects):.1f}%)")

