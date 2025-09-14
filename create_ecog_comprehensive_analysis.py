#!/usr/bin/env python3
"""
Comprehensive ECOG Analysis: Self-Reported and Informant
========================================================

This script creates comprehensive tables for both:
1. Self-reported ECOG (BHR_EverydayCognition.csv) - QID53-* and QID54-*
2. Informant ECOG (BHR_SP_ECog.csv) - QID48, QID49-*, QID50-*

Each table will show:
- QID code and description
- Average RT and std dev for each response level
- Sample sizes for each response
- Statistical significance
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def apply_ashford_filter(df, min_acc=0.60):
    """Apply Ashford quality criteria for cognitive data validity"""
    return df[(df['CorrectPCT'] >= min_acc) &
              (df['CorrectResponsesRT'].between(0.5, 2.5))].copy()

def load_memtrax_data():
    """Load and prepare MemTrax data"""
    print("1. Loading MemTrax data...")
    memtrax = pd.read_csv(MEMTRAX_DIR / 'MemTraxRecalculated.csv')
    print(f"   Loaded {len(memtrax)} MemTrax records")
    
    # Apply quality filter
    memtrax_q = apply_ashford_filter(memtrax)
    print(f"   After quality filter: {len(memtrax_q)} records")
    
    # Compute aggregates per subject
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
    
    return memtrax_agg

def load_ecog_data():
    """Load both self-reported and informant ECOG data"""
    print("2. Loading ECOG data...")
    
    # Self-reported ECOG
    self_ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv')
    if 'TimepointCode' in self_ecog.columns:
        self_ecog = self_ecog[self_ecog['TimepointCode'] == 'm00']
    self_ecog = self_ecog.drop_duplicates(subset=['SubjectCode'])
    print(f"   Self-reported ECOG records: {len(self_ecog)}")
    
    # Informant ECOG
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv')
    if 'TimepointCode' in sp_ecog.columns:
        sp_ecog = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00']
    sp_ecog = sp_ecog.drop_duplicates(subset=['SubjectCode'])
    print(f"   Informant ECOG records: {len(sp_ecog)}")
    
    return self_ecog, sp_ecog

def get_ecog_qid_definitions():
    """Get ECOG QID definitions"""
    # Self-reported ECOG QIDs (QID53-* and QID54-*)
    self_ecog_definitions = {
        'QID53-1': 'Executive Function: Organization - Keeping living and work space organized',
        'QID53-2': 'Executive Function: Planning - Planning activities and events',
        'QID53-3': 'Executive Function: Problem Solving - Solving problems and making decisions',
        'QID53-4': 'Executive Function: Multitasking - Doing multiple things at once',
        'QID53-5': 'Executive Function: Attention - Paying attention to details',
        'QID53-6': 'Executive Function: Working Memory - Keeping track of information',
        'QID54-1': 'Memory: Recent Events - Remembering recent conversations',
        'QID54-2': 'Memory: Names - Remembering people\'s names',
        'QID54-3': 'Memory: Appointments - Remembering appointments and meetings',
        'QID54-4': 'Memory: Where Things Are - Remembering where you put things',
    }
    
    # Informant ECOG QIDs (QID48, QID49-*, QID50-*)
    sp_ecog_definitions = {
        'QID48': 'Overall Cognitive Change - Compared to 10 years ago, overall cognitive function',
        'QID49-1': 'Memory: Shopping Items - Remembering a few shopping items without a list',
        'QID49-2': 'Memory: Recent Events - Remembering recent events and conversations',
        'QID49-3': 'Memory: Conversations - Following conversations and discussions',
        'QID49-4': 'Memory: Object Placement - Remembering where objects are placed',
        'QID49-5': 'Memory: Repeating Stories - Avoiding repeating the same stories',
        'QID49-6': 'Memory: Current Date - Remembering the current date and time',
        'QID49-7': 'Memory: Told Someone - Remembering what you told someone',
        'QID49-8': 'Memory: Appointments - Remembering appointments and commitments',
        'QID50-1': 'Executive Function: Organization - Keeping things organized',
        'QID50-2': 'Executive Function: Planning - Planning and organizing activities',
        'QID50-3': 'Executive Function: Problem Solving - Solving everyday problems',
        'QID50-4': 'Executive Function: Multitasking - Doing multiple things at once',
        'QID50-5': 'Executive Function: Attention - Paying attention to details',
        'QID50-6': 'Executive Function: Working Memory - Keeping track of information',
    }
    
    return self_ecog_definitions, sp_ecog_definitions

def analyze_ecog_qids(ecog_data, memtrax_data, qid_definitions, ecog_type):
    """Analyze ECOG QIDs for RT statistics by response level"""
    print(f"\n3. Analyzing {ecog_type} ECOG QIDs...")
    
    # Get QID columns
    qid_columns = [col for col in ecog_data.columns if col.startswith('QID')]
    print(f"   Found {len(qid_columns)} QID columns")
    
    # Merge with MemTrax data
    data = ecog_data.merge(memtrax_data, on='SubjectCode', how='inner')
    print(f"   After merging with MemTrax: {len(data)} subjects")
    
    results = []
    
    for qid in qid_columns:
        if qid not in data.columns:
            continue
            
        print(f"   Processing {qid}...")
        
        # Get QID description
        description = qid_definitions.get(qid, f"Unknown ECOG item ({qid})")
        
        # Get unique response values (excluding NaN)
        unique_values = data[qid].dropna().unique()
        unique_values = sorted([v for v in unique_values if pd.notna(v)])
        
        if len(unique_values) == 0:
            print(f"     No valid responses for {qid}")
            continue
        
        # Analyze each response level
        for response_value in unique_values:
            # Get subjects with this response
            has_response = data[data[qid] == response_value]
            no_response = data[data[qid] != response_value]
            
            if len(has_response) == 0 or len(no_response) == 0:
                continue
            
            # RT statistics
            rt_with = has_response['CorrectResponsesRT_mean']
            rt_without = no_response['CorrectResponsesRT_mean']
            
            # Accuracy statistics
            acc_with = has_response['CorrectPCT_mean']
            acc_without = no_response['CorrectPCT_mean']
            
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
                    print(f"     RT statistical test failed for {qid}={response_value}: {e}")
            
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
                    print(f"     Accuracy statistical test failed for {qid}={response_value}: {e}")
            
            # Store results
            result = {
                'ECOG_Type': ecog_type,
                'QID': qid,
                'Description': description,
                'Response_Value': response_value,
                'N_With': len(has_response),
                'N_Without': len(no_response),
                'Prevalence': len(has_response) / (len(has_response) + len(no_response)),
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

def create_ecog_tables(results):
    """Create comprehensive ECOG tables"""
    print("\n4. Creating ECOG tables...")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Separate by ECOG type
    self_ecog_df = df[df['ECOG_Type'] == 'Self-Reported'].copy()
    sp_ecog_df = df[df['ECOG_Type'] == 'Informant'].copy()
    
    # Sort by QID and Response_Value
    self_ecog_df = self_ecog_df.sort_values(['QID', 'Response_Value'])
    sp_ecog_df = sp_ecog_df.sort_values(['QID', 'Response_Value'])
    
    # Print Self-Reported ECOG table
    print("\n" + "="*120)
    print("SELF-REPORTED ECOG ANALYSIS")
    print("="*120)
    print(f"{'QID':<12} {'Response':<8} {'N_Yes':<6} {'N_No':<6} {'Prev':<6} {'RT_Yes':<8} {'RT_Yes_Std':<8} {'RT_No':<8} {'RT_No_Std':<8} {'RT_Diff':<8} {'RT_P':<8}")
    print("-" * 120)
    
    for _, row in self_ecog_df.iterrows():
        qid = row['QID']
        resp = str(row['Response_Value'])
        desc = row['Description'][:30] + ".." if len(row['Description']) > 32 else row['Description']
        n_yes = row['N_With']
        n_no = row['N_Without']
        prev = f"{row['Prevalence']:.1%}"
        rt_yes = f"{row['RT_With_Mean']:.4f}"
        rt_yes_std = f"{row['RT_With_Std']:.4f}"
        rt_no = f"{row['RT_Without_Mean']:.4f}"
        rt_no_std = f"{row['RT_Without_Std']:.4f}"
        rt_diff = f"{row['RT_Difference']:.4f}"
        rt_p = f"{row['RT_P_Value']:.4f}" if not pd.isna(row['RT_P_Value']) else "N/A"
        
        print(f"{qid:<12} {resp:<8} {n_yes:<6} {n_no:<6} {prev:<6} {rt_yes:<8} {rt_yes_std:<8} {rt_no:<8} {rt_no_std:<8} {rt_diff:<8} {rt_p:<8}")
    
    # Print Informant ECOG table
    print("\n" + "="*120)
    print("INFORMANT ECOG ANALYSIS")
    print("="*120)
    print(f"{'QID':<12} {'Response':<8} {'N_Yes':<6} {'N_No':<6} {'Prev':<6} {'RT_Yes':<8} {'RT_Yes_Std':<8} {'RT_No':<8} {'RT_No_Std':<8} {'RT_Diff':<8} {'RT_P':<8}")
    print("-" * 120)
    
    for _, row in sp_ecog_df.iterrows():
        qid = row['QID']
        resp = str(row['Response_Value'])
        desc = row['Description'][:30] + ".." if len(row['Description']) > 32 else row['Description']
        n_yes = row['N_With']
        n_no = row['N_Without']
        prev = f"{row['Prevalence']:.1%}"
        rt_yes = f"{row['RT_With_Mean']:.4f}"
        rt_yes_std = f"{row['RT_With_Std']:.4f}"
        rt_no = f"{row['RT_Without_Mean']:.4f}"
        rt_no_std = f"{row['RT_Without_Std']:.4f}"
        rt_diff = f"{row['RT_Difference']:.4f}"
        rt_p = f"{row['RT_P_Value']:.4f}" if not pd.isna(row['RT_P_Value']) else "N/A"
        
        print(f"{qid:<12} {resp:<8} {n_yes:<6} {n_no:<6} {prev:<6} {rt_yes:<8} {rt_yes_std:<8} {rt_no:<8} {rt_no_std:<8} {rt_diff:<8} {rt_p:<8}")
    
    return df, self_ecog_df, sp_ecog_df

def save_ecog_results(df, self_ecog_df, sp_ecog_df):
    """Save ECOG results to CSV files"""
    print("\n5. Saving ECOG results...")
    
    # Save comprehensive table
    df.to_csv(OUTPUT_DIR / 'comprehensive_ecog_analysis.csv', index=False)
    print(f"   Comprehensive ECOG table saved to: {OUTPUT_DIR / 'comprehensive_ecog_analysis.csv'}")
    
    # Save separate tables
    self_ecog_df.to_csv(OUTPUT_DIR / 'self_reported_ecog_analysis.csv', index=False)
    sp_ecog_df.to_csv(OUTPUT_DIR / 'informant_ecog_analysis.csv', index=False)
    print(f"   Self-reported ECOG table saved to: {OUTPUT_DIR / 'self_reported_ecog_analysis.csv'}")
    print(f"   Informant ECOG table saved to: {OUTPUT_DIR / 'informant_ecog_analysis.csv'}")
    
    # Save summary
    summary = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'total_ecog_items': len(df),
        'self_reported_items': len(self_ecog_df),
        'informant_items': len(sp_ecog_df),
        'qids_analyzed': df['QID'].nunique(),
        'response_levels_analyzed': len(df),
        'significant_rt_differences': len(df[df['RT_P_Value'] < 0.05]),
        'significant_acc_differences': len(df[df['Acc_P_Value'] < 0.05])
    }
    
    with open(OUTPUT_DIR / 'ecog_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   Summary saved to: {OUTPUT_DIR / 'ecog_analysis_summary.json'}")

def main():
    """Main function"""
    print("="*120)
    print("Comprehensive ECOG Analysis: Self-Reported and Informant")
    print("="*120)
    print("Analyzing ECOG QIDs and their relationship to MemTrax performance")
    print()
    
    # Load data
    memtrax_data = load_memtrax_data()
    self_ecog, sp_ecog = load_ecog_data()
    self_ecog_definitions, sp_ecog_definitions = get_ecog_qid_definitions()
    
    # Analyze self-reported ECOG
    self_results = analyze_ecog_qids(self_ecog, memtrax_data, self_ecog_definitions, 'Self-Reported')
    
    # Analyze informant ECOG
    sp_results = analyze_ecog_qids(sp_ecog, memtrax_data, sp_ecog_definitions, 'Informant')
    
    # Combine results
    all_results = self_results + sp_results
    
    # Create tables
    df, self_ecog_df, sp_ecog_df = create_ecog_tables(all_results)
    
    # Save results
    save_ecog_results(df, self_ecog_df, sp_ecog_df)
    
    print("\n" + "="*120)
    print("ECOG ANALYSIS COMPLETE")
    print("="*120)
    print(f"Analyzed {len(all_results)} ECOG response levels")
    print(f"Self-reported ECOG items: {len(self_ecog_df)}")
    print(f"Informant ECOG items: {len(sp_ecog_df)}")
    print(f"QIDs analyzed: {df['QID'].nunique()}")
    print(f"Significant RT differences: {len(df[df['RT_P_Value'] < 0.05])}")
    print(f"Significant accuracy differences: {len(df[df['Acc_P_Value'] < 0.05])}")

if __name__ == "__main__":
    main()
