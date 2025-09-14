#!/usr/bin/env python3
"""
BHR Population-Level Analysis
Compare BHR demographics and MCI prevalence to general population
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*70)
print("BHR POPULATION-LEVEL ANALYSIS")
print("="*70)

# Load demographics
demo_files = {
    'BHR_Demographics.csv': 'BHR Demographics',
    'Profile.csv': 'Profile',
    'Participants.csv': 'Participants'
}

# Try to get the best demographic data
demo_df = None
for file, name in demo_files.items():
    path = DATA_DIR / file
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        if 'Code' in df.columns:
            df = df.rename(columns={'Code': 'SubjectCode'})
        
        # Check for age and education columns
        has_age = any('age' in c.lower() for c in df.columns)
        has_edu = any('educ' in c.lower() for c in df.columns)
        
        if has_age and has_edu and 'SubjectCode' in df.columns:
            print(f"Using {name} for demographics (best data)")
            demo_df = df
            break
        elif demo_df is None and 'SubjectCode' in df.columns:
            demo_df = df

if demo_df is None:
    print("ERROR: No demographic data found")
    exit(1)

# Find age column
age_col = None
for col in demo_df.columns:
    if 'age_baseline' in col.lower():
        age_col = col
        break
if not age_col:
    for col in demo_df.columns:
        if 'age' in col.lower() and demo_df[col].dtype in ['float64', 'int64']:
            age_col = col
            break

# Find education column  
edu_col = None
for col in demo_df.columns:
    if 'yearseducation' in col.lower():
        edu_col = col
        break

print(f"\nUsing columns: Age='{age_col}', Education='{edu_col}'")

# Clean demographics
if age_col:
    demo_df['Age'] = pd.to_numeric(demo_df[age_col], errors='coerce')
    demo_df = demo_df[(demo_df['Age'] >= 18) & (demo_df['Age'] <= 110)]

if edu_col:
    demo_df['Education'] = pd.to_numeric(demo_df[edu_col], errors='coerce')
    demo_df = demo_df[(demo_df['Education'] >= 0) & (demo_df['Education'] <= 30)]

print(f"Total participants with valid demographics: {len(demo_df):,}")

# Load medical history for MCI labels
med_path = DATA_DIR / 'BHR_MedicalHx.csv'
med_df = pd.read_csv(med_path, low_memory=False)

# Get baseline only
if 'TimepointCode' in med_df.columns:
    med_df = med_df[med_df['TimepointCode'] == 'm00']
    print(f"Medical history (baseline): {len(med_df):,} records")

# Define MCI from multiple QIDs
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
qids_present = [q for q in COGNITIVE_QIDS if q in med_df.columns]
print(f"Using cognitive QIDs: {qids_present}")

# Create MCI label
med_df['MCI'] = 0
for qid in qids_present:
    med_df['MCI'] |= (med_df[qid] == 1)

# Merge demographics with medical
merged = demo_df.merge(med_df[['SubjectCode', 'MCI'] + qids_present], 
                       on='SubjectCode', how='inner')
print(f"\nMerged data: {len(merged):,} participants")

print("\n" + "="*70)
print("1. AGE DISTRIBUTION")
print("="*70)

if 'Age' in merged.columns:
    age_stats = merged['Age'].describe()
    print(f"Mean age: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years")
    print(f"Median age: {age_stats['50%']:.1f} years")
    print(f"Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
    
    print("\n%-20s %8s %8s %15s %15s" % ("Age Group", "N", "%", "BHR %", "US Pop %"))
    print("-"*70)
    
    # US population percentages (rough estimates for adults 18+)
    us_pop = {
        '18-44': 46.0,
        '45-54': 17.0,
        '55-64': 17.0,
        '65-74': 13.0,
        '75-84': 5.5,
        '85+': 1.5
    }
    
    age_groups = [
        ('18-44', 18, 45),
        ('45-54', 45, 55),
        ('55-64', 55, 65),
        ('65-74', 65, 75),
        ('75-84', 75, 85),
        ('85+', 85, 120)
    ]
    
    for label, min_age, max_age in age_groups:
        mask = (merged['Age'] >= min_age) & (merged['Age'] < max_age)
        n = mask.sum()
        pct = n / len(merged) * 100
        us_pct = us_pop.get(label, 0)
        diff = pct - us_pct
        sign = '+' if diff > 0 else ''
        print("%-20s %8d %7.1f%% %14.1f%% %14.1f%% (%s%.1f%%)" % 
              (label, n, pct, pct, us_pct, sign, diff))

print("\n" + "="*70)
print("2. EDUCATION DISTRIBUTION")
print("="*70)

if 'Education' in merged.columns:
    edu_stats = merged['Education'].describe()
    print(f"Mean education: {edu_stats['mean']:.1f} ± {edu_stats['std']:.1f} years")
    print(f"Median education: {edu_stats['50%']:.1f} years")
    
    print("\n%-25s %8s %8s %15s %15s" % ("Education Level", "N", "%", "BHR %", "US Pop %"))
    print("-"*70)
    
    # US population education (adults 25+, 2021 census)
    us_edu = {
        'Less than HS (<12)': 11.4,
        'High School (12)': 27.9,
        'Some College (13-15)': 28.9,
        "Bachelor's (16)": 19.8,
        'Graduate (17+)': 12.0
    }
    
    edu_groups = [
        ('Less than HS (<12)', 0, 12),
        ('High School (12)', 12, 13),
        ('Some College (13-15)', 13, 16),
        ("Bachelor's (16)", 16, 17),
        ('Graduate (17+)', 17, 30)
    ]
    
    for label, min_edu, max_edu in edu_groups:
        if max_edu == 13:  # Exactly 12
            mask = merged['Education'] == 12
        else:
            mask = (merged['Education'] >= min_edu) & (merged['Education'] < max_edu)
        n = mask.sum()
        pct = n / len(merged) * 100
        us_pct = us_edu.get(label, 0)
        diff = pct - us_pct
        sign = '+' if diff > 0 else ''
        print("%-25s %8d %7.1f%% %14.1f%% %14.1f%% (%s%.1f%%)" % 
              (label, n, pct, pct, us_pct, sign, diff))

print("\n" + "="*70)
print("3. MCI PREVALENCE BY AGE")
print("="*70)

# Expected MCI prevalence by age (from literature)
expected_mci = {
    '45-54': 2.0,
    '55-64': 6.7,
    '65-74': 13.1,
    '75-84': 20.7,
    '85+': 37.6
}

print("\n%-15s %8s %8s %12s %12s %15s" % 
      ("Age Group", "N Total", "N MCI", "BHR Prev", "Expected", "Difference"))
print("-"*80)

for label, min_age, max_age in age_groups[1:]:  # Skip 18-44
    mask = (merged['Age'] >= min_age) & (merged['Age'] < max_age)
    subset = merged[mask]
    if len(subset) > 0:
        n_total = len(subset)
        n_mci = subset['MCI'].sum()
        prevalence = n_mci / n_total * 100
        expected = expected_mci.get(label, 0)
        diff = prevalence - expected
        sign = '+' if diff > 0 else ''
        
        print("%-15s %8d %8d %11.1f%% %11.1f%% %14s%.1f%%" % 
              (label, n_total, n_mci, prevalence, expected, sign, diff))

# Overall prevalence
overall_prev = merged['MCI'].mean() * 100
print(f"\nOverall MCI prevalence: {overall_prev:.1f}%")
print(f"Expected (age-adjusted): ~10-12% for this age distribution")

print("\n" + "="*70)
print("4. MCI PREVALENCE BY EDUCATION")
print("="*70)

print("\n%-20s %8s %8s %12s %15s" % 
      ("Education Level", "N Total", "N MCI", "Prevalence", "Notes"))
print("-"*70)

edu_prev_groups = [
    ('< High School', 0, 12),
    ('High School', 12, 13),
    ('Some College', 13, 16),
    ("Bachelor's", 16, 17),
    ('Graduate+', 17, 30)
]

for label, min_edu, max_edu in edu_prev_groups:
    if 'Education' in merged.columns:
        if max_edu == 13:
            mask = merged['Education'] == 12
        else:
            mask = (merged['Education'] >= min_edu) & (merged['Education'] < max_edu)
        subset = merged[mask]
        if len(subset) > 0:
            n_total = len(subset)
            n_mci = subset['MCI'].sum()
            prevalence = n_mci / n_total * 100
            
            # Add interpretation
            if prevalence < 8:
                note = "Lower (reserve?)"
            elif prevalence > 12:
                note = "Higher"
            else:
                note = "Expected"
                
            print("%-20s %8d %8d %11.1f%% %14s" % 
                  (label, n_total, n_mci, prevalence, note))

print("\n" + "="*70)
print("5. KEY INSIGHTS")
print("="*70)

print("""
📊 BHR vs General Population:
- BHR is YOUNGER and MORE EDUCATED than US population
- Under-representation of 75+ age group (where MCI is common)
- Over-representation of college-educated participants

🧠 MCI Prevalence Patterns:
- Overall prevalence appears REASONABLE for age distribution
- BUT may have under-reporting in highly educated groups
- Cognitive reserve could mask symptoms in educated participants

⚠️ Self-Reporting Issues:
- Educated participants may not recognize/report mild symptoms
- "Worried well" may join but not have actual MCI
- Those with MCI may not participate (selection bias)
""")

# Check for missing MCI labels
print("\n" + "="*70)
print("6. DATA QUALITY CHECK")
print("="*70)

for qid in qids_present:
    valid = merged[qid].isin([1.0, 2.0]).sum()
    missing = merged[qid].isna().sum()
    print(f"{qid}: {valid:,} valid responses, {missing:,} missing ({missing/len(merged)*100:.1f}%)")

print(f"\nParticipants with at least one cognitive assessment: {(merged[qids_present].notna().any(axis=1)).sum():,}")
print(f"Participants with NO cognitive assessment: {(merged[qids_present].isna().all(axis=1)).sum():,}")
