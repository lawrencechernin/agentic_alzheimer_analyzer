#!/usr/bin/env python3
"""
BHR Population Analysis using the demographics enrichment module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from improvements.demographics_enrichment import enrich_demographics

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*70)
print("BHR POPULATION-LEVEL ANALYSIS V2")
print("="*70)

# Load medical history first to get subjects
med_path = DATA_DIR / 'BHR_MedicalHx.csv'
med_df = pd.read_csv(med_path, low_memory=False)
print(f"Medical history loaded: {len(med_df):,} records")

# Get baseline only
if 'TimepointCode' in med_df.columns:
    med_baseline = med_df[med_df['TimepointCode'] == 'm00'].copy()
    print(f"Baseline records: {len(med_baseline):,}")
else:
    med_baseline = med_df

# Create MCI composite
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
qids_present = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]
print(f"Cognitive QIDs found: {qids_present}")

# Create MCI label - count how many said yes (1) or no (2) to any cognitive question
med_baseline['MCI'] = 0
med_baseline['has_cognitive_data'] = False

for qid in qids_present:
    # QID value of 1 = Yes (has condition), 2 = No
    med_baseline['MCI'] |= (med_baseline[qid] == 1).fillna(False)
    med_baseline['has_cognitive_data'] |= med_baseline[qid].isin([1.0, 2.0])

# Keep only those with at least one cognitive assessment
med_with_cog = med_baseline[med_baseline['has_cognitive_data']].copy()
print(f"Subjects with cognitive assessment: {len(med_with_cog):,}")
print(f"MCI cases identified: {med_with_cog['MCI'].sum():,}")
print(f"MCI prevalence in assessed: {med_with_cog['MCI'].mean()*100:.1f}%")

# Create a base dataframe with just SubjectCode to enrich
base_df = med_with_cog[['SubjectCode', 'MCI']].copy()

# Use the demographics enrichment function
print("\nEnriching with demographics...")
enriched = enrich_demographics(DATA_DIR, base_df)

# Check what we got
print(f"Enriched data shape: {enriched.shape}")
demo_cols = [c for c in enriched.columns if c not in ['SubjectCode', 'MCI']]
print(f"Added columns: {demo_cols[:10]}...")

# Use the enriched data for analysis
merged = enriched

print(f"\nFinal dataset: {len(merged):,} participants")

print("\n" + "="*70)
print("1. AGE DISTRIBUTION")
print("="*70)

age_col = 'Age_Baseline' if 'Age_Baseline' in merged.columns else None
if not age_col and 'Age' in merged.columns:
    age_col = 'Age'

if age_col:
    age_data = pd.to_numeric(merged[age_col], errors='coerce')
    age_data = age_data[(age_data >= 18) & (age_data <= 110)]
    
    if len(age_data) > 0:
        print(f"N with valid age: {len(age_data):,}")
        print(f"Mean age: {age_data.mean():.1f} ± {age_data.std():.1f} years")
        print(f"Median age: {age_data.median():.1f} years")
        print(f"Range: {age_data.min():.0f} - {age_data.max():.0f} years")
        
        print("\n%-20s %8s %8s %15s" % ("Age Group", "N", "BHR %", "US Pop %"))
        print("-"*65)
        
        # US adult population distribution
        us_pop = {
            '18-44': 46.0,
            '45-54': 17.0,
            '55-64': 17.0,
            '65-74': 13.0,
            '75-84': 5.5,
            '85+': 1.5
        }
        
        # Calculate BHR distribution
        age_groups = [
            ('18-44', 18, 45),
            ('45-54', 45, 55),
            ('55-64', 55, 65),
            ('65-74', 65, 75),
            ('75-84', 75, 85),
            ('85+', 85, 120)
        ]
        
        for label, min_age, max_age in age_groups:
            mask = (age_data >= min_age) & (age_data < max_age)
            n = mask.sum()
            pct = n / len(age_data) * 100
            us_pct = us_pop[label]
            diff = pct - us_pct
            
            if abs(diff) > 5:
                status = "***" if abs(diff) > 10 else "**"
            else:
                status = ""
                
            print("%-20s %8d %7.1f%% %14.1f%% %s" % 
                  (label, n, pct, us_pct, status))
        
        # Summary stats
        pct_over_65 = (age_data >= 65).mean() * 100
        pct_over_75 = (age_data >= 75).mean() * 100
        print(f"\n65+ years: {pct_over_65:.1f}% (US: ~20%)")
        print(f"75+ years: {pct_over_75:.1f}% (US: ~7%)")
else:
    print("No age data available")

print("\n" + "="*70)
print("2. EDUCATION DISTRIBUTION")
print("="*70)

edu_col = 'YearsEducationUS_Converted' if 'YearsEducationUS_Converted' in merged.columns else None
if not edu_col and 'Education_Years' in merged.columns:
    edu_col = 'Education_Years'

if edu_col:
    edu_data = pd.to_numeric(merged[edu_col], errors='coerce')
    edu_data = edu_data[(edu_data >= 0) & (edu_data <= 30)]
    
    if len(edu_data) > 0:
        print(f"N with valid education: {len(edu_data):,}")
        print(f"Mean education: {edu_data.mean():.1f} ± {edu_data.std():.1f} years")
        print(f"Median education: {edu_data.median():.1f} years")
        
        print("\n%-25s %8s %8s %15s" % ("Education Level", "N", "BHR %", "US Pop %"))
        print("-"*70)
        
        # US education distribution
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
            if label == 'High School (12)':
                mask = edu_data == 12
            elif label == "Bachelor's (16)":
                mask = edu_data == 16
            else:
                mask = (edu_data >= min_edu) & (edu_data < max_edu)
                
            n = mask.sum()
            pct = n / len(edu_data) * 100
            us_pct = us_edu[label]
            diff = pct - us_pct
            
            if abs(diff) > 5:
                status = "***" if abs(diff) > 10 else "**"
            else:
                status = ""
                
            print("%-25s %8d %7.1f%% %14.1f%% %s" % 
                  (label, n, pct, us_pct, status))
        
        # Summary stats
        pct_college = (edu_data >= 16).mean() * 100
        pct_graduate = (edu_data >= 17).mean() * 100
        print(f"\nCollege+ (16+ years): {pct_college:.1f}% (US: ~32%)")
        print(f"Graduate (17+ years): {pct_graduate:.1f}% (US: ~12%)")
else:
    print("No education data available")

print("\n" + "="*70)
print("3. MCI PREVALENCE BY AGE")
print("="*70)

if age_col:
    print("\n%-15s %8s %8s %12s %12s" % 
          ("Age Group", "N Total", "N MCI", "BHR Prev", "Expected"))
    print("-"*65)
    
    # Expected prevalence from literature
    expected_mci = {
        '45-54': 3.0,
        '55-64': 6.7,
        '65-74': 13.1,
        '75-84': 20.7,
        '85+': 37.6
    }
    
    age_prevalence_groups = [
        ('45-54', 45, 55),
        ('55-64', 55, 65),
        ('65-74', 65, 75),
        ('75-84', 75, 85),
        ('85+', 85, 120)
    ]
    
    for label, min_age, max_age in age_prevalence_groups:
        age_vals = pd.to_numeric(merged[age_col], errors='coerce')
        mask = (age_vals >= min_age) & (age_vals < max_age)
        subset = merged[mask]
        
        if len(subset) > 10:
            n_total = len(subset)
            n_mci = subset['MCI'].sum()
            prevalence = n_mci / n_total * 100
            expected = expected_mci[label]
            
            diff = prevalence - expected
            if abs(diff) > 5:
                status = "***"
            elif abs(diff) > 2:
                status = "**"
            else:
                status = ""
                
            print("%-15s %8d %8d %11.1f%% %11.1f%% %s" % 
                  (label, n_total, n_mci, prevalence, expected, status))
    
    print(f"\nOverall prevalence: {merged['MCI'].mean()*100:.1f}%")

print("\n" + "="*70)
print("4. MCI PREVALENCE BY EDUCATION")
print("="*70)

if edu_col:
    print("\n%-20s %8s %8s %12s" % 
          ("Education Level", "N Total", "N MCI", "Prevalence"))
    print("-"*55)
    
    edu_prevalence_groups = [
        ('< High School', 0, 12),
        ('High School', 12, 13),
        ('Some College', 13, 16),
        ("Bachelor's", 16, 17),
        ('Graduate+', 17, 30)
    ]
    
    for label, min_edu, max_edu in edu_prevalence_groups:
        edu_vals = pd.to_numeric(merged[edu_col], errors='coerce')
        
        if label == 'High School':
            mask = edu_vals == 12
        elif label == "Bachelor's":
            mask = edu_vals == 16
        else:
            mask = (edu_vals >= min_edu) & (edu_vals < max_edu)
            
        subset = merged[mask]
        
        if len(subset) > 10:
            n_total = len(subset)
            n_mci = subset['MCI'].sum()
            prevalence = n_mci / n_total * 100
            
            print("%-20s %8d %8d %11.1f%%" % 
                  (label, n_total, n_mci, prevalence))

print("\n" + "="*70)
print("5. KEY INSIGHTS")
print("="*70)

print("""
Legend:
  ** = Moderate difference from US population (5-10%)
  *** = Large difference from US population (>10%)

Key Findings:
- Compare BHR age distribution to US population
- Compare BHR education levels to US population
- Check if MCI prevalence matches expected rates by age
- Look for cognitive reserve effects in educated groups
""")

# Check for selection bias indicators
if age_col and edu_col:
    age_vals = pd.to_numeric(merged[age_col], errors='coerce')
    edu_vals = pd.to_numeric(merged[edu_col], errors='coerce')
    
    # Check correlation between education and MCI
    edu_mci_corr = edu_vals.corr(merged['MCI'])
    age_mci_corr = age_vals.corr(merged['MCI'])
    
    print(f"\nCorrelations:")
    print(f"  Age vs MCI: {age_mci_corr:.3f} (expect positive)")
    print(f"  Education vs MCI: {edu_mci_corr:.3f} (expect negative)")
    
    # Check if highly educated have lower MCI rates
    high_edu = edu_vals >= 16
    low_edu = edu_vals < 12
    
    if high_edu.sum() > 10 and low_edu.sum() > 10:
        high_edu_mci = merged[high_edu]['MCI'].mean() * 100
        low_edu_mci = merged[low_edu]['MCI'].mean() * 100
        
        print(f"\nCognitive Reserve Effect:")
        print(f"  MCI in low education (<12y): {low_edu_mci:.1f}%")
        print(f"  MCI in high education (16+y): {high_edu_mci:.1f}%")
        print(f"  Difference: {low_edu_mci - high_edu_mci:.1f}%")
        
        if high_edu_mci < low_edu_mci:
            print("  → Evidence of cognitive reserve masking MCI in educated")
