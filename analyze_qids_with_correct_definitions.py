#!/usr/bin/env python3
"""
Corrected QID Analysis with Proper Definitions
============================================

This script analyzes MemTrax RT and accuracy using the CORRECT QID definitions
from the data dictionary. Now we can see what each QID actually represents.

Key findings from data dictionary:
- QID1-5: Dementia (not MCI!)
- QID1-12: Alzheimer's Disease  
- QID1-13: Mild Cognitive Impairment (MCI)
- QID1-22: Frontotemporal Dementia (FTD)
- QID1-23: Lewy Body Disease (LBD)
- QID1-14: Traumatic Brain Injury
- QID1-18: Concussion
- QID1-19: Epilepsy or Seizures
- QID1-21: Multiple Sclerosis (MS)
- QID1-24: Essential Tremor
- QID1-25: Huntington's Disease
- QID1-26: Amyotrophic Lateral Sclerosis (ALS)

And many others for medical conditions, mental health, etc.
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Data paths
MEMTRAX_DIR = Path('../bhr/from_paul/processed/')
DATA_DIR = Path('../bhr/BHR-ALL-EXT_Mem_2022/')
OUTPUT_DIR = Path('bhr_memtrax_results')

# Correct QID definitions from data dictionary
QID_DEFINITIONS = {
    # Cognitive/Neurological Conditions
    'QID1-5': 'Dementia',
    'QID1-12': 'Alzheimer\'s Disease',
    'QID1-13': 'Mild Cognitive Impairment (MCI)',
    'QID1-22': 'Frontotemporal Dementia (FTD)',
    'QID1-23': 'Lewy Body Disease (LBD)',
    'QID1-14': 'Traumatic Brain Injury',
    'QID1-18': 'Concussion',
    'QID1-19': 'Epilepsy or Seizures',
    'QID1-21': 'Multiple Sclerosis (MS)',
    'QID1-24': 'Essential Tremor',
    'QID1-25': 'Huntington\'s Disease',
    'QID1-26': 'Amyotrophic Lateral Sclerosis (ALS)',
    
    # Movement Disorders
    'QID1-1': 'Parkinson\'s Disease',
    'QID1-2': 'Movement Disorder',
    
    # Other Neurological
    'QID1-3': 'Stroke',
    'QID1-4': 'Motor Neuron Disease',
    
    # Medical Conditions
    'QID1-6': 'Heart Disease',
    'QID1-7': 'High Blood Pressure',
    'QID1-8': 'High Cholesterol',
    'QID1-9': 'Diabetes',
    'QID1-10': 'Cancer',
    'QID1-15': 'Lung Disease',
    'QID1-16': 'Asthma',
    'QID1-17': 'Arthritis',
    'QID1-20': 'Hearing Loss',
    
    # Substance Use
    'QID9': 'Alcohol Abuse',
    'QID17': 'Drug Abuse',
    'QID20': 'Tobacco Smoking',
    
    # Other
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

def analyze_cognitive_conditions(data, qid_columns):
    """Focus on cognitive/neurological conditions with proper definitions"""
    print("\n" + "="*100)
    print("COGNITIVE CONDITIONS ANALYSIS (with correct definitions)")
    print("="*100)
    
    # Focus on cognitive/neurological QIDs
    cognitive_qids = [
        'QID1-5',   # Dementia
        'QID1-12',  # Alzheimer's Disease
        'QID1-13',  # MCI
        'QID1-22',  # FTD
        'QID1-23',  # LBD
        'QID1-14',  # TBI
        'QID1-18',  # Concussion
        'QID1-19',  # Epilepsy
        'QID1-21',  # MS
        'QID1-1',   # Parkinson's
        'QID1-2',   # Movement Disorder
        'QID1-3',   # Stroke
        'QID1-24',  # Essential Tremor
        'QID1-25',  # Huntington's
        'QID1-26',  # ALS
    ]
    
    results = []
    
    for qid in cognitive_qids:
        if qid not in data.columns:
            continue
            
        print(f"\n--- {qid}: {QID_DEFINITIONS.get(qid, 'Unknown')} ---")
        
        # Get subjects with and without this condition
        has_condition = data[data[qid] == 1]
        no_condition = data[data[qid] == 2]  # Note: 2 = No, 1 = Yes
        
        print(f"Subjects with condition: {len(has_condition)}")
        print(f"Subjects without condition: {len(no_condition)}")
        
        if len(has_condition) == 0:
            print("No subjects with this condition - skipping")
            continue
        
        # RT statistics
        rt_with = has_condition['CorrectResponsesRT_mean']
        rt_without = no_condition['CorrectResponsesRT_mean']
        
        # Accuracy statistics
        acc_with = has_condition['CorrectPCT_mean']
        acc_without = no_condition['CorrectPCT_mean']
        
        print(f"\nRT Statistics:")
        print(f"  With condition:    Mean={rt_with.mean():.4f}, Std={rt_with.std():.4f}")
        print(f"  Without condition: Mean={rt_without.mean():.4f}, Std={rt_without.std():.4f}")
        
        print(f"\nAccuracy Statistics:")
        print(f"  With condition:    Mean={acc_with.mean():.4f}, Std={acc_with.std():.4f}")
        print(f"  Without condition: Mean={acc_without.mean():.4f}, Std={acc_without.std():.4f}")
        
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
                
                print(f"\nRT Statistical Test:")
                print(f"  p-value: {rt_p_value:.6f}")
                print(f"  Effect size (Cohen's d): {rt_cohens_d:.4f}")
                
                if rt_p_value < 0.05:
                    print(f"  Significance: * (RT is significantly different)")
                else:
                    print(f"  Significance: ns")
                
            except Exception as e:
                print(f"  RT statistical test failed: {e}")
        
        if len(acc_with) >= 3 and len(acc_without) >= 3:
            try:
                # Accuracy statistical test
                acc_statistic, acc_p_value = stats.mannwhitneyu(acc_with, acc_without, alternative='two-sided')
                
                # Accuracy effect size
                acc_pooled_std = np.sqrt(((len(acc_with) - 1) * acc_with.var() + 
                                        (len(acc_without) - 1) * acc_without.var()) / 
                                       (len(acc_with) + len(acc_without) - 2))
                acc_cohens_d = (acc_with.mean() - acc_without.mean()) / acc_pooled_std
                
                print(f"\nAccuracy Statistical Test:")
                print(f"  p-value: {acc_p_value:.6f}")
                print(f"  Effect size (Cohen's d): {acc_cohens_d:.4f}")
                
                if acc_p_value < 0.05:
                    print(f"  Significance: * (Accuracy is significantly different)")
                else:
                    print(f"  Significance: ns")
                
            except Exception as e:
                print(f"  Accuracy statistical test failed: {e}")
        
        # Store results
        result = {
            'qid': qid,
            'description': QID_DEFINITIONS.get(qid, 'Unknown'),
            'n_with': len(has_condition),
            'n_without': len(no_condition),
            'prevalence': len(has_condition) / (len(has_condition) + len(no_condition)),
            'rt_with_mean': rt_with.mean(),
            'rt_with_std': rt_with.std(),
            'rt_without_mean': rt_without.mean(),
            'rt_without_std': rt_without.std(),
            'rt_difference': rt_with.mean() - rt_without.mean(),
            'rt_p_value': rt_p_value,
            'rt_cohens_d': rt_cohens_d,
            'acc_with_mean': acc_with.mean(),
            'acc_with_std': acc_with.std(),
            'acc_without_mean': acc_without.mean(),
            'acc_without_std': acc_without.std(),
            'acc_difference': acc_with.mean() - acc_without.mean(),
            'acc_p_value': acc_p_value,
            'acc_cohens_d': acc_cohens_d
        }
        
        results.append(result)
    
    return results

def create_cognitive_summary_table(results):
    """Create a summary table for cognitive conditions"""
    print("\n" + "="*100)
    print("COGNITIVE CONDITIONS SUMMARY")
    print("="*100)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by prevalence (descending)
    df = df.sort_values('prevalence', ascending=False)
    
    # Print summary table
    print(f"{'QID':<12} {'Condition':<35} {'N':<6} {'Prev':<6} {'RT Diff':<8} {'RT p-val':<10} {'Acc Diff':<8} {'Acc p-val':<10}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        qid = row['qid']
        desc = row['description'][:33] + ".." if len(row['description']) > 35 else row['description']
        n = row['n_with']
        prev = f"{row['prevalence']:.1%}"
        rt_diff = f"{row['rt_difference']:.4f}"
        rt_p = f"{row['rt_p_value']:.4f}" if not pd.isna(row['rt_p_value']) else "N/A"
        acc_diff = f"{row['acc_difference']:.4f}"
        acc_p = f"{row['acc_p_value']:.4f}" if not pd.isna(row['acc_p_value']) else "N/A"
        
        print(f"{qid:<12} {desc:<35} {n:<6} {prev:<6} {rt_diff:<8} {rt_p:<10} {acc_diff:<8} {acc_p:<10}")
    
    return df

def analyze_healthy_vs_impaired(data):
    """Analyze healthy subjects vs those with any cognitive impairment"""
    print("\n" + "="*100)
    print("HEALTHY vs COGNITIVE IMPAIRMENT ANALYSIS")
    print("="*100)
    
    # Define cognitive impairment QIDs
    cognitive_impairment_qids = [
        'QID1-5',   # Dementia
        'QID1-12',  # Alzheimer's Disease
        'QID1-13',  # MCI
        'QID1-22',  # FTD
        'QID1-23',  # LBD
    ]
    
    # Create cognitive impairment flag
    data['has_cognitive_impairment'] = False
    for qid in cognitive_impairment_qids:
        if qid in data.columns:
            data['has_cognitive_impairment'] |= (data[qid] == 1)
    
    # Analyze
    healthy = data[~data['has_cognitive_impairment']]
    impaired = data[data['has_cognitive_impairment']]
    
    print(f"Healthy subjects: {len(healthy)}")
    print(f"Cognitive impairment subjects: {len(impaired)}")
    
    if len(healthy) > 0 and len(impaired) > 0:
        print(f"\nRT Statistics:")
        print(f"  Healthy:    Mean={healthy['CorrectResponsesRT_mean'].mean():.4f}, Std={healthy['CorrectResponsesRT_mean'].std():.4f}")
        print(f"  Impaired:   Mean={impaired['CorrectResponsesRT_mean'].mean():.4f}, Std={impaired['CorrectResponsesRT_mean'].std():.4f}")
        
        print(f"\nAccuracy Statistics:")
        print(f"  Healthy:    Mean={healthy['CorrectPCT_mean'].mean():.4f}, Std={healthy['CorrectPCT_mean'].std():.4f}")
        print(f"  Impaired:   Mean={impaired['CorrectPCT_mean'].mean():.4f}, Std={impaired['CorrectPCT_mean'].std():.4f}")
        
        # Statistical test
        try:
            rt_statistic, rt_p_value = stats.mannwhitneyu(
                healthy['CorrectResponsesRT_mean'], 
                impaired['CorrectResponsesRT_mean'], 
                alternative='two-sided'
            )
            
            acc_statistic, acc_p_value = stats.mannwhitneyu(
                healthy['CorrectPCT_mean'], 
                impaired['CorrectPCT_mean'], 
                alternative='two-sided'
            )
            
            print(f"\nStatistical Tests:")
            print(f"  RT p-value: {rt_p_value:.6f}")
            print(f"  Accuracy p-value: {acc_p_value:.6f}")
            
            if rt_p_value < 0.05:
                print(f"  RT: Significantly different between healthy and impaired")
            else:
                print(f"  RT: Not significantly different")
                
            if acc_p_value < 0.05:
                print(f"  Accuracy: Significantly different between healthy and impaired")
            else:
                print(f"  Accuracy: Not significantly different")
                
        except Exception as e:
            print(f"Statistical test failed: {e}")

def main():
    """Main function"""
    print("="*100)
    print("Corrected QID Analysis with Proper Definitions")
    print("="*100)
    print("Now using the ACTUAL QID definitions from the data dictionary!")
    print()
    
    # Load and prepare data
    data, qid_columns = load_and_prepare_data()
    
    # Analyze cognitive conditions
    results = analyze_cognitive_conditions(data, qid_columns)
    
    # Create summary table
    results_df = create_cognitive_summary_table(results)
    
    # Analyze healthy vs impaired
    analyze_healthy_vs_impaired(data)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("Key insights:")
    print("- Now using correct QID definitions from data dictionary")
    print("- QID1-5 = Dementia (not MCI!)")
    print("- QID1-13 = MCI (the correct one)")
    print("- QID1-12 = Alzheimer's Disease")
    print("- This should give us much more meaningful results!")

if __name__ == "__main__":
    main()
