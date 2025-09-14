#!/usr/bin/env python3
"""
Analyze Mismatch Cases: MemTrax vs Medical Labels
================================================

Find cases where:
1. Poor MemTrax performance BUT no cognitive impairment medical labels
2. Good MemTrax performance BUT high cognitive impairment medical labels

This helps identify:
- Label quality issues
- Systematic biases
- Factors affecting AUC ceiling
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
    
    # Demographics
    demo = pd.read_csv(DATA_DIR / 'Profile.csv')
    demo = demo.rename(columns={'Code': 'SubjectCode'})
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

def create_cognitive_labels(med_hx):
    """Create comprehensive cognitive impairment labels from medical history"""
    print("\nCreating cognitive impairment labels...")
    
    # Filter to baseline
    med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].copy()
    med_baseline = med_baseline.drop_duplicates(subset=['SubjectCode'])
    
    # Define cognitive impairment QIDs
    cognitive_qids = [
        'QID1-13',  # Mild Cognitive Impairment
        'QID1-14',  # Alzheimer's Disease
        'QID1-15',  # Dementia
        'QID1-16',  # Parkinson's Disease
        'QID1-17',  # Other Dementia
        'QID1-18',  # Stroke (can cause cognitive impairment)
        'QID1-19',  # Epilepsy (can cause cognitive impairment)
        'QID1-20',  # Head Injury (can cause cognitive impairment)
    ]
    
    # Check which QIDs are available
    available_qids = [q for q in cognitive_qids if q in med_baseline.columns]
    print(f"Available cognitive QIDs: {available_qids}")
    
    # Create labels
    labels = med_baseline[['SubjectCode']].copy()
    
    # Individual conditions
    for qid in available_qids:
        labels[f'{qid}_present'] = (med_baseline[qid] == 1).astype(int)
    
    # Composite labels
    cognitive_cols = [f'{qid}_present' for qid in available_qids]
    labels['any_cognitive_condition'] = labels[cognitive_cols].max(axis=1)
    labels['cognitive_condition_count'] = labels[cognitive_cols].sum(axis=1)
    
    # Specific combinations
    if 'QID1-13_present' in labels.columns:  # MCI
        labels['mci_diagnosed'] = labels['QID1-13_present']
    else:
        labels['mci_diagnosed'] = 0
    
    if 'QID1-14_present' in labels.columns and 'QID1-15_present' in labels.columns:  # Alzheimer's or Dementia
        labels['alzheimers_or_dementia'] = (labels['QID1-14_present'] | labels['QID1-15_present']).astype(int)
    else:
        labels['alzheimers_or_dementia'] = 0
    
    # Neurological conditions that can affect cognition
    neuro_cols = []
    for qid in ['QID1-18', 'QID1-19', 'QID1-20']:  # Stroke, Epilepsy, Head Injury
        if f'{qid}_present' in labels.columns:
            neuro_cols.append(f'{qid}_present')
    
    if neuro_cols:
        labels['neurological_cognitive_risk'] = labels[neuro_cols].max(axis=1)
    else:
        labels['neurological_cognitive_risk'] = 0
    
    print(f"Label distribution:")
    print(f"  Any cognitive condition: {labels['any_cognitive_condition'].mean():.1%}")
    print(f"  MCI diagnosed: {labels['mci_diagnosed'].mean():.1%}")
    print(f"  Alzheimer's/Dementia: {labels['alzheimers_or_dementia'].mean():.1%}")
    print(f"  Neurological risk: {labels['neurological_cognitive_risk'].mean():.1%}")
    
    return labels

def create_memtrax_performance(memtrax):
    """Create MemTrax performance metrics"""
    print("\nCreating MemTrax performance metrics...")
    
    # Filter to quality data
    memtrax_clean = memtrax[
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ].copy()
    
    # Aggregate by subject
    subject_performance = memtrax_clean.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std', 'count'],
        'CorrectResponsesRT': ['mean', 'std'],
        'IncorrectPCT': 'mean',
        'TimepointCode': 'nunique'
    }).round(3)
    
    subject_performance.columns = ['_'.join(col) for col in subject_performance.columns]
    subject_performance = subject_performance.reset_index()
    
    # Filter for subjects with sufficient tests
    subject_performance = subject_performance[subject_performance['CorrectPCT_count'] >= 3]
    
    print(f"Subjects with ≥3 tests: {len(subject_performance)}")
    print(f"Accuracy range: {subject_performance['CorrectPCT_mean'].min():.3f} - {subject_performance['CorrectPCT_mean'].max():.3f}")
    print(f"RT range: {subject_performance['CorrectResponsesRT_mean'].min():.3f} - {subject_performance['CorrectResponsesRT_mean'].max():.3f}s")
    
    return subject_performance

def find_mismatch_cases(memtrax_perf, cognitive_labels, demo):
    """Find mismatch cases between MemTrax and cognitive labels"""
    print("\nFinding mismatch cases...")
    
    # Merge data
    data = memtrax_perf.merge(cognitive_labels, on='SubjectCode', how='inner')
    data = data.merge(demo, on='SubjectCode', how='left')
    
    print(f"Subjects with both MemTrax and cognitive labels: {len(data)}")
    
    # Define thresholds
    poor_memtrax_threshold = 0.75  # Bottom 25% of accuracy
    good_memtrax_threshold = 0.90  # Top 25% of accuracy
    high_rt_threshold = 1.2  # Slow RT
    
    # Calculate thresholds based on data
    poor_threshold = data['CorrectPCT_mean'].quantile(0.25)
    good_threshold = data['CorrectPCT_mean'].quantile(0.75)
    
    print(f"Poor MemTrax threshold (25th percentile): {poor_threshold:.3f}")
    print(f"Good MemTrax threshold (75th percentile): {good_threshold:.3f}")
    
    # Case 1: Poor MemTrax but NO cognitive impairment labels
    case1 = data[
        (data['CorrectPCT_mean'] <= poor_threshold) &
        (data['any_cognitive_condition'] == 0) &
        (data['CorrectResponsesRT_mean'] >= high_rt_threshold)  # Also slow
    ].copy()
    
    # Case 2: Good MemTrax but HIGH cognitive impairment labels
    case2 = data[
        (data['CorrectPCT_mean'] >= good_threshold) &
        (data['any_cognitive_condition'] == 1) &
        (data['CorrectResponsesRT_mean'] <= 1.0)  # Also fast
    ].copy()
    
    # Case 3: Poor MemTrax with cognitive labels (expected)
    case3 = data[
        (data['CorrectPCT_mean'] <= poor_threshold) &
        (data['any_cognitive_condition'] == 1)
    ].copy()
    
    # Case 4: Good MemTrax without cognitive labels (expected)
    case4 = data[
        (data['CorrectPCT_mean'] >= good_threshold) &
        (data['any_cognitive_condition'] == 0)
    ].copy()
    
    print(f"\nMismatch Cases Found:")
    print(f"  Case 1 - Poor MemTrax, NO cognitive labels: {len(case1)}")
    print(f"  Case 2 - Good MemTrax, HIGH cognitive labels: {len(case2)}")
    print(f"  Case 3 - Poor MemTrax, WITH cognitive labels: {len(case3)} (expected)")
    print(f"  Case 4 - Good MemTrax, NO cognitive labels: {len(case4)} (expected)")
    
    return case1, case2, case3, case4, data

def analyze_mismatch_case1(case1, ecog, sp_ecog, med_hx):
    """Analyze Case 1: Poor MemTrax but no cognitive labels"""
    print(f"\n" + "="*60)
    print("CASE 1 ANALYSIS: Poor MemTrax, NO Cognitive Labels")
    print("="*60)
    
    if len(case1) == 0:
        print("No cases found!")
        return
    
    print(f"Found {len(case1)} subjects with poor MemTrax but no cognitive impairment labels")
    
    # Show top cases
    print(f"\nTop 10 worst performers without cognitive labels:")
    top_cases = case1.nsmallest(10, 'CorrectPCT_mean')
    
    for i, (_, subject) in enumerate(top_cases.iterrows(), 1):
        print(f"\n{i}. {subject['SubjectCode']}")
        print(f"   MemTrax: {subject['CorrectPCT_mean']:.3f} accuracy, {subject['CorrectResponsesRT_mean']:.3f}s RT")
        print(f"   Tests: {subject['CorrectPCT_count']}, RT Std: {subject['CorrectResponsesRT_std']:.3f}")
        
        # Demographics
        if pd.notna(subject.get('AgeRange')):
            print(f"   Demographics: Age {subject['AgeRange']}, Education {subject.get('Education', 'N/A')} years")
        
        # Check ECOG scores
        ecog_data = ecog[ecog['SubjectCode'] == subject['SubjectCode']]
        if not ecog_data.empty:
            ecog_baseline = ecog_data[ecog_data['TimepointCode'] == 'm00']
            if not ecog_baseline.empty:
                # Check memory domain (QID49)
                memory_cols = [col for col in ecog_baseline.columns if col.startswith('QID49-')]
                if memory_cols:
                    memory_scores = ecog_baseline[memory_cols].replace(8, np.nan).mean(axis=1)
                    if not memory_scores.isna().all():
                        print(f"   ECOG Memory: {memory_scores.iloc[0]:.2f}")
                
                # Check language domain (QID50)
                language_cols = [col for col in ecog_baseline.columns if col.startswith('QID50-')]
                if language_cols:
                    language_scores = ecog_baseline[language_cols].replace(8, np.nan).mean(axis=1)
                    if not language_scores.isna().all():
                        print(f"   ECOG Language: {language_scores.iloc[0]:.2f}")
    
    # Analyze patterns
    print(f"\nPatterns in Case 1:")
    print(f"  Average accuracy: {case1['CorrectPCT_mean'].mean():.3f}")
    print(f"  Average RT: {case1['CorrectResponsesRT_mean'].mean():.3f}s")
    print(f"  Average RT std: {case1['CorrectResponsesRT_std'].mean():.3f}")
    
    if 'Education' in case1.columns:
        edu_counts = case1['Education'].value_counts()
        print(f"  Education distribution: {edu_counts.to_dict()}")
    
    return top_cases

def analyze_mismatch_case2(case2, ecog, sp_ecog, med_hx):
    """Analyze Case 2: Good MemTrax but high cognitive labels"""
    print(f"\n" + "="*60)
    print("CASE 2 ANALYSIS: Good MemTrax, HIGH Cognitive Labels")
    print("="*60)
    
    if len(case2) == 0:
        print("No cases found!")
        return
    
    print(f"Found {len(case2)} subjects with good MemTrax but cognitive impairment labels")
    
    # Show cases
    print(f"\nAll cases with good MemTrax but cognitive labels:")
    
    for i, (_, subject) in enumerate(case2.iterrows(), 1):
        print(f"\n{i}. {subject['SubjectCode']}")
        print(f"   MemTrax: {subject['CorrectPCT_mean']:.3f} accuracy, {subject['CorrectResponsesRT_mean']:.3f}s RT")
        print(f"   Cognitive conditions: {subject['cognitive_condition_count']}")
        
        # Show which conditions
        condition_cols = [col for col in subject.index if col.endswith('_present') and subject[col] == 1]
        if condition_cols:
            print(f"   Conditions: {', '.join(condition_cols)}")
        
        # Demographics
        if pd.notna(subject.get('AgeRange')):
            print(f"   Demographics: Age {subject['AgeRange']}, Education {subject.get('Education', 'N/A')} years")
        
        # Check ECOG scores
        ecog_data = ecog[ecog['SubjectCode'] == subject['SubjectCode']]
        if not ecog_data.empty:
            ecog_baseline = ecog_data[ecog_data['TimepointCode'] == 'm00']
            if not ecog_baseline.empty:
                # Check memory domain (QID49)
                memory_cols = [col for col in ecog_baseline.columns if col.startswith('QID49-')]
                if memory_cols:
                    memory_scores = ecog_baseline[memory_cols].replace(8, np.nan).mean(axis=1)
                    if not memory_scores.isna().all():
                        print(f"   ECOG Memory: {memory_scores.iloc[0]:.2f}")
    
    # Analyze patterns
    print(f"\nPatterns in Case 2:")
    print(f"  Average accuracy: {case2['CorrectPCT_mean'].mean():.3f}")
    print(f"  Average RT: {case2['CorrectResponsesRT_mean'].mean():.3f}s")
    print(f"  Average condition count: {case2['cognitive_condition_count'].mean():.1f}")
    
    # Check which conditions are most common
    condition_cols = [col for col in case2.columns if col.endswith('_present')]
    condition_counts = case2[condition_cols].sum().sort_values(ascending=False)
    print(f"  Most common conditions:")
    for condition, count in condition_counts.head(5).items():
        if count > 0:
            print(f"    {condition}: {count}/{len(case2)} ({count/len(case2)*100:.1f}%)")
    
    return case2

def main():
    """Main analysis function"""
    print("="*60)
    print("ANALYZING MISMATCH CASES: MemTrax vs Medical Labels")
    print("="*60)
    
    # Load data
    memtrax, demo, med_hx, ecog, sp_ecog = load_data()
    
    # Create labels and performance metrics
    cognitive_labels = create_cognitive_labels(med_hx)
    memtrax_perf = create_memtrax_performance(memtrax)
    
    # Find mismatch cases
    case1, case2, case3, case4, all_data = find_mismatch_cases(memtrax_perf, cognitive_labels, demo)
    
    # Analyze each case
    case1_details = analyze_mismatch_case1(case1, ecog, sp_ecog, med_hx)
    case2_details = analyze_mismatch_case2(case2, ecog, sp_ecog, med_hx)
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if len(case1) > 0:
        case1.to_csv(OUTPUT_DIR / 'mismatch_case1_poor_memtrax_no_cognitive.csv', index=False)
        print(f"\nSaved Case 1 to: {OUTPUT_DIR / 'mismatch_case1_poor_memtrax_no_cognitive.csv'}")
    
    if len(case2) > 0:
        case2.to_csv(OUTPUT_DIR / 'mismatch_case2_good_memtrax_high_cognitive.csv', index=False)
        print(f"Saved Case 2 to: {OUTPUT_DIR / 'mismatch_case2_good_memtrax_high_cognitive.csv'}")
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total subjects analyzed: {len(all_data)}")
    print(f"Case 1 (Poor MemTrax, No Cognitive): {len(case1)} ({len(case1)/len(all_data)*100:.1f}%)")
    print(f"Case 2 (Good MemTrax, High Cognitive): {len(case2)} ({len(case2)/len(all_data)*100:.1f}%)")
    print(f"Case 3 (Poor MemTrax, With Cognitive): {len(case3)} ({len(case3)/len(all_data)*100:.1f}%)")
    print(f"Case 4 (Good MemTrax, No Cognitive): {len(case4)} ({len(case4)/len(all_data)*100:.1f}%)")
    
    # Calculate agreement
    expected_cases = len(case3) + len(case4)
    mismatch_cases = len(case1) + len(case2)
    agreement_rate = expected_cases / (expected_cases + mismatch_cases) * 100
    
    print(f"\nAgreement Rate: {agreement_rate:.1f}%")
    print(f"Mismatch Rate: {100-agreement_rate:.1f}%")
    
    if len(case1) > 0:
        print(f"\nPotential issues:")
        print(f"  - {len(case1)} subjects with poor MemTrax but no cognitive labels")
        print(f"    → Possible: Undiagnosed cognitive impairment, test issues, other factors")
    
    if len(case2) > 0:
        print(f"  - {len(case2)} subjects with good MemTrax but cognitive labels")
        print(f"    → Possible: Cognitive reserve, early stage, test insensitivity, label issues")

if __name__ == "__main__":
    main()

