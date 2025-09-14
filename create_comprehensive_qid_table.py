#!/usr/bin/env python3
"""
Comprehensive QID Table: All Medical History QIDs
================================================

This script creates a complete table of all QIDs in the MedicalHx file with:
- QID code
- QID description/label
- Average RT for subjects with condition (Yes)
- Standard deviation RT for subjects with condition (Yes)
- Average RT for subjects without condition (No)
- Standard deviation RT for subjects without condition (No)
- Number of subjects with condition
- Number of subjects without condition
- Prevalence
- Statistical significance (p-value)
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Data paths
MEMTRAX_DIR = Path('../bhr/from_paul/processed/')
DATA_DIR = Path('../bhr/BHR-ALL-EXT_Mem_2022/')
OUTPUT_DIR = Path('bhr_memtrax_results')

# Complete QID definitions from data dictionary
QID_DEFINITIONS = {
    # Cognitive/Neurological Conditions
    'QID1-1': 'Parkinson\'s Disease',
    'QID1-2': 'Movement Disorder',
    'QID1-3': 'Stroke',
    'QID1-4': 'Motor Neuron Disease',
    'QID1-5': 'Dementia',
    'QID1-6': 'Heart Disease',
    'QID1-7': 'High Blood Pressure',
    'QID1-8': 'High Cholesterol',
    'QID1-9': 'Diabetes',
    'QID1-10': 'Cancer',
    'QID1-12': 'Alzheimer\'s Disease',
    'QID1-13': 'Mild Cognitive Impairment (MCI)',
    'QID1-14': 'Traumatic Brain Injury',
    'QID1-15': 'Lung Disease',
    'QID1-16': 'Asthma',
    'QID1-17': 'Arthritis',
    'QID1-18': 'Concussion',
    'QID1-19': 'Epilepsy or Seizures',
    'QID1-20': 'Hearing Loss',
    'QID1-21': 'Multiple Sclerosis (MS)',
    'QID1-22': 'Frontotemporal Dementia (FTD)',
    'QID1-23': 'Lewy Body Disease (LBD)',
    'QID1-24': 'Essential Tremor',
    'QID1-25': 'Huntington\'s Disease',
    'QID1-26': 'Amyotrophic Lateral Sclerosis (ALS)',
    
    # Substance Use
    'QID9': 'Alcohol Abuse',
    'QID17': 'Drug Abuse',
    'QID20': 'Tobacco Smoking',
    
    # Other Medical
    'QID2': 'Chronic Pain',
    'QID4': 'Sleep Apnea',
    'QID6': 'Allergies',
    
    # Mental Health (Current)
    'QID28#1-1': 'Current Major Depressive Disorder',
    'QID28#1-3': 'Current Specific Phobia / Social Phobia',
    'QID28#1-4': 'Current Obsessive Compulsive Disorder',
    'QID28#1-5': 'Current Hoarding Disorder',
    'QID28#1-6': 'Current ADHD',
    'QID28#1-8': 'Current PTSD',
    'QID28#1-9': 'Current Generalized Anxiety Disorder',
    'QID28#1-10': 'Current Panic Disorder',
    'QID28#1-11': 'Current Bipolar Disorder',
    'QID28#1-12': 'Current Autism',
    'QID28#1-13': 'Current Schizophrenia',
    'QID28#1-14': 'Current Eating Disorder',
    'QID28#1-15': 'Current Psychosis',
    
    # Mental Health (Past)
    'QID28#2-1': 'Past Major Depressive Disorder',
    'QID28#2-3': 'Past Specific Phobia / Social Phobia',
    'QID28#2-4': 'Past Obsessive Compulsive Disorder',
    'QID28#2-5': 'Past Hoarding Disorder',
    'QID28#2-6': 'Past ADHD',
    'QID28#2-8': 'Past PTSD',
    'QID28#2-9': 'Past Generalized Anxiety Disorder',
    'QID28#2-10': 'Past Panic Disorder',
    'QID28#2-11': 'Past Bipolar Disorder',
    'QID28#2-12': 'Past Autism',
    'QID28#2-13': 'Past Schizophrenia',
    'QID28#2-14': 'Past Eating Disorder',
    'QID28#2-15': 'Past Psychosis',
    
    # MRI Safety
    'QID30-1': 'Cardiac Pacemaker/Defibrillator',
    'QID30-2': 'Surgical Metal/Foreign Objects',
    'QID30-3': 'Stents/Filter/Intravascular Coils',
    'QID30-4': 'Internal Pacing Wires',
    'QID30-5': 'Sternum Wires',
    'QID30-6': 'Claustrophobia',
    'QID31': 'Worked with Metal',
    'QID32': 'Previous MRI Scan',
    
    # Medications
    'QID34-1': 'Donepezil (Aricept)',
    'QID34-2': 'Rivastigmine (Exelon)',
    'QID34-3': 'Memantine (Namenda)',
    'QID34-4': 'Galantamine (Razadyne)',
    'QID34-5': 'Aducanumab (Aduhelm)',
    
    # Text fields (for completeness)
    'QID3-1': 'Pain Severity (1-10)',
    'QID7-1-TEXT': 'Allergy 1',
    'QID7-2-TEXT': 'Allergy 2',
    'QID7-3-TEXT': 'Allergy 3',
    'QID7-4-TEXT': 'Allergy 4',
    'QID7-5-TEXT': 'Allergy 5',
    'QID13-TEXT': 'Alcohol Abuse Duration (years)',
    'QID15-TEXT': 'Years Since Stopped Alcohol',
    'QID16-TEXT': 'Average Drinks Per Day',
    'QID18-TEXT': 'Drug Abuse Duration (years)',
    'QID19-TEXT': 'Years Since Stopped Drugs',
    'QID21-TEXT': 'Tobacco Smoking Duration (years)',
    'QID22-TEXT': 'Years Since Stopped Smoking',
    'QID23-TEXT': 'Average Cigarettes Per Day',
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def apply_ashford_filter(df, min_acc=0.60):
    """Apply Ashford quality criteria for cognitive data validity"""
    return df[(df['CorrectPCT'] >= min_acc) &
              (df['CorrectResponsesRT'].between(0.5, 2.5))].copy()

def load_and_prepare_data():
    """Load and prepare MemTrax and medical history data"""
    print("1. Loading MemTrax data...")
    memtrax = pd.read_csv(MEMTRAX_DIR / 'MemTraxRecalculated.csv')
    print(f"   Loaded {len(memtrax)} MemTrax records")
    
    # Apply quality filter
    print("2. Applying Ashford quality filter...")
    memtrax_q = apply_ashford_filter(memtrax)
    print(f"   After quality filter: {len(memtrax_q)} records")
    
    # Compute aggregates per subject
    print("3. Computing MemTrax aggregates...")
    memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std', 'min', 'max', 'count'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': ['mean', 'std'],
        'IncorrectResponsesRT': ['mean', 'std'],
        'CorrectRejectionsN': ['mean', 'std'],
        'IncorrectRejectionsN': ['mean', 'std']
    })
    memtrax_agg.columns = ['_'.join(col) for col in memtrax_agg.columns]
    memtrax_agg = memtrax_agg.reset_index()
    
    # Filter subjects with sufficient data
    memtrax_agg = memtrax_agg[memtrax_agg['CorrectPCT_count'] >= 3]
    print(f"   Subjects with sufficient MemTrax data: {len(memtrax_agg)}")
    
    # Load medical history
    print("4. Loading medical history...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    
    # Filter to baseline timepoint
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    print(f"   Medical history records: {len(med_hx)}")
    
    # Get all QID columns
    qid_columns = [col for col in med_hx.columns if col.startswith('QID')]
    print(f"   Found {len(qid_columns)} QID columns")
    
    # Merge data
    print("5. Merging data...")
    data = memtrax_agg.merge(med_hx[['SubjectCode'] + qid_columns], on='SubjectCode', how='inner')
    print(f"   Final dataset: {len(data)} subjects")
    
    return data, qid_columns

def analyze_all_qids(data, qid_columns):
    """Analyze all QIDs for RT statistics"""
    print("\n" + "="*120)
    print("COMPREHENSIVE QID ANALYSIS")
    print("="*120)
    
    results = []
    
    for qid in qid_columns:
        print(f"Processing {qid}...")
        
        # Get QID description
        description = QID_DEFINITIONS.get(qid, f"Unknown condition ({qid})")
        
        # Get subjects with and without this condition
        # Handle different coding schemes
        has_condition = None
        no_condition = None
        
        if qid in data.columns:
            # Check for different value coding schemes
            unique_values = data[qid].dropna().unique()
            
            if 1 in unique_values and 2 in unique_values:
                # Standard coding: 1=Yes, 2=No
                has_condition = data[data[qid] == 1]
                no_condition = data[data[qid] == 2]
            elif 1 in unique_values and 0 in unique_values:
                # Alternative coding: 1=Yes, 0=No
                has_condition = data[data[qid] == 1]
                no_condition = data[data[qid] == 0]
            elif True in unique_values and False in unique_values:
                # Boolean coding
                has_condition = data[data[qid] == True]
                no_condition = data[data[qid] == False]
            else:
                # Skip if no clear Yes/No coding
                print(f"   Skipping {qid} - unclear coding scheme: {unique_values}")
                continue
        
        if has_condition is None or len(has_condition) == 0:
            print(f"   No subjects with condition for {qid}")
            continue
        
        if len(no_condition) == 0:
            print(f"   No subjects without condition for {qid}")
            continue
        
        # RT statistics
        rt_with = has_condition['CorrectResponsesRT_mean']
        rt_without = no_condition['CorrectResponsesRT_mean']
        
        # Accuracy statistics
        acc_with = has_condition['CorrectPCT_mean']
        acc_without = no_condition['CorrectPCT_mean']
        
        # Statistical tests
        rt_p_value = np.nan
        rt_cohens_d = np.nan
        acc_p_value = np.nan
        acc_cohens_d = np.nan
        
        if len(rt_with) >= 3 and len(rt_without) >= 3:
            try:
                # RT statistical test
                rt_statistic, rt_p_value = stats.mannwhitneyu(rt_with, rt_without, alternative='two-sided')
                
                # RT effect size
                rt_pooled_std = np.sqrt(((len(rt_with) - 1) * rt_with.var() + 
                                       (len(rt_without) - 1) * rt_without.var()) / 
                                      (len(rt_with) + len(rt_without) - 2))
                rt_cohens_d = (rt_with.mean() - rt_without.mean()) / rt_pooled_std
                
            except Exception as e:
                print(f"   RT statistical test failed for {qid}: {e}")
        
        if len(acc_with) >= 3 and len(acc_without) >= 3:
            try:
                # Accuracy statistical test
                acc_statistic, acc_p_value = stats.mannwhitneyu(acc_with, acc_without, alternative='two-sided')
                
                # Accuracy effect size
                acc_pooled_std = np.sqrt(((len(acc_with) - 1) * acc_with.var() + 
                                        (len(acc_without) - 1) * acc_without.var()) / 
                                       (len(acc_with) + len(acc_without) - 2))
                acc_cohens_d = (acc_with.mean() - acc_without.mean()) / acc_pooled_std
                
            except Exception as e:
                print(f"   Accuracy statistical test failed for {qid}: {e}")
        
        # Store results
        result = {
            'QID': qid,
            'Description': description,
            'N_With': len(has_condition),
            'N_Without': len(no_condition),
            'Prevalence': len(has_condition) / (len(has_condition) + len(no_condition)),
            'RT_With_Mean': rt_with.mean(),
            'RT_With_Std': rt_with.std(),
            'RT_Without_Mean': rt_without.mean(),
            'RT_Without_Std': rt_without.std(),
            'RT_Difference': rt_with.mean() - rt_without.mean(),
            'RT_P_Value': rt_p_value,
            'RT_Cohens_D': rt_cohens_d,
            'Acc_With_Mean': acc_with.mean(),
            'Acc_With_Std': acc_with.std(),
            'Acc_Without_Mean': acc_without.mean(),
            'Acc_Without_Std': acc_without.std(),
            'Acc_Difference': acc_with.mean() - acc_without.mean(),
            'Acc_P_Value': acc_p_value,
            'Acc_Cohens_D': acc_cohens_d
        }
        
        results.append(result)
    
    return results

def create_comprehensive_table(results):
    """Create the comprehensive QID table"""
    print("\n" + "="*120)
    print("COMPREHENSIVE QID TABLE")
    print("="*120)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by prevalence (descending)
    df = df.sort_values('Prevalence', ascending=False)
    
    # Print table header
    print(f"{'QID':<12} {'Description':<40} {'N_Yes':<6} {'N_No':<6} {'Prev':<6} {'RT_Yes':<8} {'RT_Yes_Std':<8} {'RT_No':<8} {'RT_No_Std':<8} {'RT_Diff':<8} {'RT_P':<8}")
    print("-" * 120)
    
    # Print each row
    for _, row in df.iterrows():
        qid = row['QID']
        desc = row['Description'][:38] + ".." if len(row['Description']) > 40 else row['Description']
        n_yes = row['N_With']
        n_no = row['N_Without']
        prev = f"{row['Prevalence']:.1%}"
        rt_yes = f"{row['RT_With_Mean']:.4f}"
        rt_yes_std = f"{row['RT_With_Std']:.4f}"
        rt_no = f"{row['RT_Without_Mean']:.4f}"
        rt_no_std = f"{row['RT_Without_Std']:.4f}"
        rt_diff = f"{row['RT_Difference']:.4f}"
        rt_p = f"{row['RT_P_Value']:.4f}" if not pd.isna(row['RT_P_Value']) else "N/A"
        
        print(f"{qid:<12} {desc:<40} {n_yes:<6} {n_no:<6} {prev:<6} {rt_yes:<8} {rt_yes_std:<8} {rt_no:<8} {rt_no_std:<8} {rt_diff:<8} {rt_p:<8}")
    
    return df

def save_results(results_df):
    """Save results to CSV files"""
    print("\n6. Saving results...")
    
    # Save comprehensive table
    results_df.to_csv(OUTPUT_DIR / 'comprehensive_qid_table.csv', index=False)
    print(f"   Comprehensive table saved to: {OUTPUT_DIR / 'comprehensive_qid_table.csv'}")
    
    # Save summary statistics
    summary = {
        'total_qids_analyzed': len(results_df),
        'qids_with_conditions': len(results_df[results_df['N_With'] > 0]),
        'qids_significant_rt': len(results_df[results_df['RT_P_Value'] < 0.05]),
        'qids_significant_acc': len(results_df[results_df['Acc_P_Value'] < 0.05]),
        'analysis_date': pd.Timestamp.now().isoformat()
    }
    
    with open(OUTPUT_DIR / 'qid_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   Summary saved to: {OUTPUT_DIR / 'qid_analysis_summary.json'}")

def main():
    """Main function"""
    print("="*120)
    print("Comprehensive QID Table: All Medical History QIDs")
    print("="*120)
    print("Creating complete table with QID, label, RT statistics, and significance")
    print()
    
    # Load and prepare data
    data, qid_columns = load_and_prepare_data()
    
    # Analyze all QIDs
    results = analyze_all_qids(data, qid_columns)
    
    # Create comprehensive table
    results_df = create_comprehensive_table(results)
    
    # Save results
    save_results(results_df)
    
    print("\n" + "="*120)
    print("ANALYSIS COMPLETE")
    print("="*120)
    print(f"Analyzed {len(results_df)} QIDs")
    print(f"QIDs with conditions: {len(results_df[results_df['N_With'] > 0])}")
    print(f"QIDs with significant RT differences: {len(results_df[results_df['RT_P_Value'] < 0.05])}")
    print(f"QIDs with significant accuracy differences: {len(results_df[results_df['Acc_P_Value'] < 0.05])}")

if __name__ == "__main__":
    main()
