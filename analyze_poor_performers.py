#!/usr/bin/env python3
"""
Analyze Poor MemTrax Performers
===============================

Find participants with consistently poor MemTrax results and investigate:
1. Their ECOG/SP-ECOG scores (informant reports)
2. Medical history patterns
3. Demographics and risk factors
4. Longitudinal patterns

Goal: Understand what drives poor MemTrax performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data paths
MEMTRAX_DIR = Path('../bhr/from_paul/processed/')
DATA_DIR = Path('../bhr/BHR-ALL-EXT_Mem_2022/')
OUTPUT_DIR = Path('bhr_memtrax_results')

def load_data():
    """Load all relevant datasets"""
    print("Loading datasets...")
    
    # MemTrax data
    memtrax = pd.read_csv(MEMTRAX_DIR / 'MemTraxRecalculated.csv')
    print(f"MemTrax: {len(memtrax)} records")
    
    # Demographics (use Profile.csv for basic demographics)
    demo = pd.read_csv(DATA_DIR / 'Profile.csv')
    demo = demo.rename(columns={'Code': 'SubjectCode'})  # Rename for consistency
    print(f"Demographics: {len(demo)} records")
    
    # Medical history
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    print(f"Medical History: {len(med_hx)} records")
    
    # ECOG (self-report)
    ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv')
    print(f"ECOG: {len(ecog)} records")
    
    # SP-ECOG (informant)
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv')
    print(f"SP-ECOG: {len(sp_ecog)} records")
    
    return memtrax, demo, med_hx, ecog, sp_ecog

def identify_poor_performers(memtrax, min_tests=3, poor_threshold=0.7):
    """
    Identify participants with consistently poor MemTrax performance
    
    Criteria:
    - At least min_tests test sessions
    - Average accuracy below poor_threshold
    - High RT (slow responses)
    """
    print(f"\nIdentifying poor performers...")
    print(f"Criteria: ≥{min_tests} tests, accuracy <{poor_threshold}")
    
    # Filter to quality data (no Status column in processed data)
    memtrax_clean = memtrax[
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ].copy()
    
    # Aggregate by subject
    subject_stats = memtrax_clean.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std', 'count'],
        'CorrectResponsesRT': ['mean', 'std'],
        'IncorrectPCT': 'mean',
        'TimepointCode': 'nunique'
    }).round(3)
    
    subject_stats.columns = ['_'.join(col) for col in subject_stats.columns]
    subject_stats = subject_stats.reset_index()
    
    # Filter for poor performers
    poor_performers = subject_stats[
        (subject_stats['CorrectPCT_count'] >= min_tests) &
        (subject_stats['CorrectPCT_mean'] < poor_threshold) &
        (subject_stats['CorrectResponsesRT_mean'] > 1.0)  # Slow responses
    ].copy()
    
    # Sort by worst performance
    poor_performers = poor_performers.sort_values('CorrectPCT_mean')
    
    print(f"Found {len(poor_performers)} poor performers")
    print(f"Accuracy range: {poor_performers['CorrectPCT_mean'].min():.3f} - {poor_performers['CorrectPCT_mean'].max():.3f}")
    print(f"RT range: {poor_performers['CorrectResponsesRT_mean'].min():.3f} - {poor_performers['CorrectResponsesRT_mean'].max():.3f}s")
    
    return poor_performers, subject_stats

def analyze_demographics(poor_performers, demo):
    """Analyze demographic patterns of poor performers"""
    print(f"\nAnalyzing demographics...")
    
    # Merge with demographics
    poor_demo = poor_performers.merge(demo, on='SubjectCode', how='left')
    
    print(f"Demographics available for {poor_demo['SubjectCode'].notna().sum()}/{len(poor_demo)} poor performers")
    
    if poor_demo['SubjectCode'].notna().sum() > 0:
        print(f"\nAge distribution:")
        if 'AgeRange' in poor_demo.columns:
            age_counts = poor_demo['AgeRange'].value_counts().sort_index()
            for age, count in age_counts.items():
                print(f"  {age}: {count} ({count/len(poor_demo)*100:.1f}%)")
        else:
            print("  Age data not available")
        
        print(f"\nEducation distribution:")
        if 'YearsEducationUS_Converted' in poor_demo.columns:
            edu_counts = poor_demo['YearsEducationUS_Converted'].value_counts().sort_index()
            for edu, count in edu_counts.items():
                if pd.notna(edu):
                    print(f"  {edu} years: {count} ({count/len(poor_demo)*100:.1f}%)")
        else:
            print("  Education data not available")
        
        print(f"\nGender distribution:")
        if 'Gender' in poor_demo.columns:
            gender_counts = poor_demo['Gender'].value_counts()
            for gender, count in gender_counts.items():
                gender_name = "Male" if gender == 1 else "Female" if gender == 0 else f"Unknown({gender})"
                print(f"  {gender_name}: {count} ({count/len(poor_demo)*100:.1f}%)")
        else:
            print("  Gender data not available")
    
    return poor_demo

def analyze_medical_history(poor_performers, med_hx):
    """Analyze medical conditions of poor performers"""
    print(f"\nAnalyzing medical history...")
    
    # Filter to baseline
    med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].copy()
    med_baseline = med_baseline.drop_duplicates(subset=['SubjectCode'])
    
    # Merge with poor performers
    poor_med = poor_performers.merge(med_baseline, on='SubjectCode', how='left')
    
    print(f"Medical history available for {poor_med['SubjectCode'].notna().sum()}/{len(poor_performers)} poor performers")
    
    # Key cognitive/neurological conditions to check
    cognitive_conditions = [
        'QID186',  # Memory problems
        'QID187',  # Confusion
        'QID188',  # Dementia
        'QID189',  # Alzheimer's
        'QID190',  # Stroke
        'QID191',  # Parkinson's
        'QID192',  # Depression
        'QID193',  # Anxiety
        'QID194',  # Sleep problems
        'QID195',  # Head injury
    ]
    
    available_conditions = [q for q in cognitive_conditions if q in poor_med.columns]
    
    print(f"\nMedical conditions in poor performers:")
    for qid in available_conditions:
        if qid in poor_med.columns:
            condition_count = (poor_med[qid] == 1).sum()
            if condition_count > 0:
                print(f"  {qid}: {condition_count}/{len(poor_med)} ({condition_count/len(poor_med)*100:.1f}%)")
    
    return poor_med

def analyze_ecog_scores(poor_performers, ecog, sp_ecog):
    """Analyze ECOG and SP-ECOG scores of poor performers"""
    print(f"\nAnalyzing ECOG scores...")
    
    # Filter to baseline
    ecog_baseline = ecog[ecog['TimepointCode'] == 'm00'].copy()
    sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
    
    # Merge ECOG
    poor_ecog = poor_performers.merge(ecog_baseline, on='SubjectCode', how='left')
    print(f"ECOG available for {poor_ecog['SubjectCode'].notna().sum()}/{len(poor_performers)} poor performers")
    
    # Merge SP-ECOG
    poor_sp_ecog = poor_performers.merge(sp_ecog_baseline, on='SubjectCode', how='left')
    print(f"SP-ECOG available for {poor_sp_ecog['SubjectCode'].notna().sum()}/{len(poor_performers)} poor performers")
    
    # Analyze ECOG domains if available
    ecog_domains = ['QID49', 'QID50', 'QID51', 'QID52', 'QID53', 'QID54']
    
    print(f"\nECOG domain scores (self-report):")
    for domain in ecog_domains:
        domain_cols = [col for col in poor_ecog.columns if col.startswith(f'{domain}-')]
        if domain_cols:
            # Calculate mean score for this domain
            domain_scores = poor_ecog[domain_cols].replace(8, np.nan).mean(axis=1)
            valid_scores = domain_scores.dropna()
            if len(valid_scores) > 0:
                print(f"  {domain}: {valid_scores.mean():.2f} ± {valid_scores.std():.2f} (n={len(valid_scores)})")
    
    print(f"\nSP-ECOG domain scores (informant):")
    for domain in ecog_domains:
        domain_cols = [col for col in poor_sp_ecog.columns if col.startswith(f'{domain}-')]
        if domain_cols:
            # Calculate mean score for this domain
            domain_scores = poor_sp_ecog[domain_cols].replace(8, np.nan).mean(axis=1)
            valid_scores = domain_scores.dropna()
            if len(valid_scores) > 0:
                print(f"  {domain}: {valid_scores.mean():.2f} ± {valid_scores.std():.2f} (n={len(valid_scores)})")
    
    return poor_ecog, poor_sp_ecog

def create_detailed_profiles(poor_performers, poor_demo, poor_med, poor_ecog, poor_sp_ecog):
    """Create detailed profiles of the worst performers"""
    print(f"\nCreating detailed profiles of worst performers...")
    
    # Get top 10 worst performers
    worst_10 = poor_performers.head(10).copy()
    
    profiles = []
    for _, subject in worst_10.iterrows():
        subj_code = subject['SubjectCode']
        
        profile = {
            'SubjectCode': subj_code,
            'MemTrax_Accuracy': subject['CorrectPCT_mean'],
            'MemTrax_RT': subject['CorrectResponsesRT_mean'],
            'N_Tests': subject['CorrectPCT_count'],
            'RT_Std': subject['CorrectResponsesRT_std']
        }
        
        # Add demographics
        demo_row = poor_demo[poor_demo['SubjectCode'] == subj_code]
        if not demo_row.empty:
            profile.update({
                'AgeRange': demo_row['AgeRange'].iloc[0] if pd.notna(demo_row['AgeRange'].iloc[0]) else None,
                'Education': demo_row['YearsEducationUS_Converted'].iloc[0] if pd.notna(demo_row['YearsEducationUS_Converted'].iloc[0]) else None,
                'Gender': demo_row['Gender'].iloc[0] if pd.notna(demo_row['Gender'].iloc[0]) else None
            })
        
        # Add medical conditions
        med_row = poor_med[poor_med['SubjectCode'] == subj_code]
        if not med_row.empty:
            conditions = []
            for col in med_row.columns:
                if col.startswith('QID') and med_row[col].iloc[0] == 1:
                    conditions.append(col)
            profile['Medical_Conditions'] = conditions
        
        # Add ECOG scores
        ecog_row = poor_ecog[poor_ecog['SubjectCode'] == subj_code]
        if not ecog_row.empty:
            ecog_domains = ['QID49', 'QID50', 'QID51', 'QID52', 'QID53', 'QID54']
            for domain in ecog_domains:
                domain_cols = [col for col in ecog_row.columns if col.startswith(f'{domain}-')]
                if domain_cols:
                    domain_scores = ecog_row[domain_cols].replace(8, np.nan).mean(axis=1)
                    if not domain_scores.isna().all():
                        profile[f'ECOG_{domain}'] = domain_scores.iloc[0]
        
        # Add SP-ECOG scores
        sp_ecog_row = poor_sp_ecog[poor_sp_ecog['SubjectCode'] == subj_code]
        if not sp_ecog_row.empty:
            for domain in ecog_domains:
                domain_cols = [col for col in sp_ecog_row.columns if col.startswith(f'{domain}-')]
                if domain_cols:
                    domain_scores = sp_ecog_row[domain_cols].replace(8, np.nan).mean(axis=1)
                    if not domain_scores.isna().all():
                        profile[f'SP_ECOG_{domain}'] = domain_scores.iloc[0]
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def main():
    """Main analysis function"""
    print("="*60)
    print("ANALYZING POOR MEMTRAX PERFORMERS")
    print("="*60)
    
    # Load data
    memtrax, demo, med_hx, ecog, sp_ecog = load_data()
    
    # Identify poor performers
    poor_performers, subject_stats = identify_poor_performers(memtrax)
    
    if len(poor_performers) == 0:
        print("No poor performers found with current criteria. Relaxing criteria...")
        poor_performers, subject_stats = identify_poor_performers(memtrax, min_tests=2, poor_threshold=0.8)
    
    if len(poor_performers) == 0:
        print("Still no poor performers found. Using bottom 5% of all subjects...")
        poor_performers = subject_stats.nsmallest(max(10, len(subject_stats) // 20), 'CorrectPCT_mean')
    
    # Analyze different aspects
    poor_demo = analyze_demographics(poor_performers, demo)
    poor_med = analyze_medical_history(poor_performers, med_hx)
    poor_ecog, poor_sp_ecog = analyze_ecog_scores(poor_performers, ecog, sp_ecog)
    
    # Create detailed profiles
    profiles = create_detailed_profiles(poor_performers, poor_demo, poor_med, poor_ecog, poor_sp_ecog)
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save poor performers summary
    poor_performers.to_csv(OUTPUT_DIR / 'poor_performers_summary.csv', index=False)
    print(f"\nSaved poor performers summary to: {OUTPUT_DIR / 'poor_performers_summary.csv'}")
    
    # Save detailed profiles
    profiles.to_csv(OUTPUT_DIR / 'poor_performers_profiles.csv', index=False)
    print(f"Saved detailed profiles to: {OUTPUT_DIR / 'poor_performers_profiles.csv'}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total poor performers: {len(poor_performers)}")
    print(f"Average accuracy: {poor_performers['CorrectPCT_mean'].mean():.3f}")
    print(f"Average RT: {poor_performers['CorrectResponsesRT_mean'].mean():.3f}s")
    print(f"Average tests per subject: {poor_performers['CorrectPCT_count'].mean():.1f}")
    
    print(f"\nTop 5 worst performers:")
    for i, (_, subject) in enumerate(profiles.head().iterrows(), 1):
        print(f"  {i}. {subject['SubjectCode']}: {subject['MemTrax_Accuracy']:.3f} accuracy, {subject['MemTrax_RT']:.3f}s RT")
        if pd.notna(subject.get('AgeRange')):
            print(f"     Age Range: {subject['AgeRange']}, Education: {subject.get('Education', 'N/A')} years")
        if subject.get('Medical_Conditions'):
            print(f"     Conditions: {', '.join(subject['Medical_Conditions'])}")

if __name__ == "__main__":
    main()
