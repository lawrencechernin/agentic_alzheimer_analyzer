#!/usr/bin/env python3
"""
Comprehensive QID Analysis: RT and Accuracy
==========================================

This script analyzes average RT and accuracy for ALL QIDs in the dataset,
showing their labels and statistical significance.

We'll examine:
- Average RT and accuracy for each QID
- QID descriptions from data dictionary
- Statistical significance of differences
- Effect sizes for each QID
- Data completeness and prevalence
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def apply_ashford_filter(df, min_acc=0.60):
    """Apply Ashford quality criteria for cognitive data validity"""
    return df[(df['CorrectPCT'] >= min_acc) &
              (df['CorrectResponsesRT'].between(0.5, 2.5))].copy()

def load_data_dictionary():
    """Load the data dictionary to get QID descriptions"""
    try:
        # Try to load from config directory first
        dd_path = Path('config/data_dictionary.json')
        if dd_path.exists():
            with open(dd_path, 'r') as f:
                return json.load(f)
        
        # Try to load from BHR data directory
        dd_path = DATA_DIR / 'data_dictionary.json'
        if dd_path.exists():
            with open(dd_path, 'r') as f:
                return json.load(f)
        
        # Try to find any JSON file with QID information
        for json_file in DATA_DIR.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and any('QID' in str(key) for key in data.keys()):
                        return data
            except:
                continue
        
        print("   No data dictionary found - will use QID codes only")
        return {}
        
    except Exception as e:
        print(f"   Error loading data dictionary: {e}")
        return {}

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
    memtrax_agg = memtrax_agg[memtrax_agg['CorrectPCT_count'] >= 3]  # At least 3 valid measurements
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

def analyze_qid_performance(data, qid_columns, data_dict):
    """Analyze RT and accuracy for each QID"""
    print("\n" + "="*100)
    print("QID PERFORMANCE ANALYSIS")
    print("="*100)
    
    results = []
    
    for qid in qid_columns:
        print(f"\n--- {qid} ---")
        
        # Get QID description
        description = data_dict.get(qid, f"Unknown condition ({qid})")
        print(f"Description: {description}")
        
        # Get subjects with and without this condition
        has_condition = data[data[qid] == 1]
        no_condition = data[data[qid] == 0]
        
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
                
            except Exception as e:
                print(f"  Accuracy statistical test failed: {e}")
        
        # Store results
        result = {
            'qid': qid,
            'description': description,
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

def create_summary_table(results):
    """Create a summary table of all QID results"""
    print("\n" + "="*100)
    print("SUMMARY TABLE: ALL QIDs")
    print("="*100)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by prevalence (descending)
    df = df.sort_values('prevalence', ascending=False)
    
    # Print summary table
    print(f"{'QID':<12} {'Description':<40} {'N':<6} {'Prev':<6} {'RT Diff':<8} {'RT p-val':<10} {'Acc Diff':<8} {'Acc p-val':<10}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        qid = row['qid']
        desc = row['description'][:38] + ".." if len(row['description']) > 40 else row['description']
        n = row['n_with']
        prev = f"{row['prevalence']:.1%}"
        rt_diff = f"{row['rt_difference']:.4f}"
        rt_p = f"{row['rt_p_value']:.4f}" if not pd.isna(row['rt_p_value']) else "N/A"
        acc_diff = f"{row['acc_difference']:.4f}"
        acc_p = f"{row['acc_p_value']:.4f}" if not pd.isna(row['acc_p_value']) else "N/A"
        
        print(f"{qid:<12} {desc:<40} {n:<6} {prev:<6} {rt_diff:<8} {rt_p:<10} {acc_diff:<8} {acc_p:<10}")
    
    return df

def create_visualizations(results_df):
    """Create visualization plots"""
    print("\n6. Creating visualizations...")
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('QID Performance Analysis: RT and Accuracy', fontsize=16, fontweight='bold')
    
    # Plot 1: RT difference by prevalence
    ax1 = axes[0, 0]
    scatter = ax1.scatter(results_df['prevalence'], results_df['rt_difference'], 
                         c=results_df['rt_p_value'], s=100, alpha=0.7, cmap='viridis')
    ax1.set_xlabel('Prevalence')
    ax1.set_ylabel('RT Difference (With - Without)')
    ax1.set_title('RT Difference vs Prevalence')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='RT p-value')
    
    # Plot 2: Accuracy difference by prevalence
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(results_df['prevalence'], results_df['acc_difference'], 
                          c=results_df['acc_p_value'], s=100, alpha=0.7, cmap='viridis')
    ax2.set_xlabel('Prevalence')
    ax2.set_ylabel('Accuracy Difference (With - Without)')
    ax2.set_title('Accuracy Difference vs Prevalence')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Accuracy p-value')
    
    # Plot 3: RT vs Accuracy differences
    ax3 = axes[1, 0]
    ax3.scatter(results_df['rt_difference'], results_df['acc_difference'], 
               c=results_df['prevalence'], s=100, alpha=0.7, cmap='plasma')
    ax3.set_xlabel('RT Difference (With - Without)')
    ax3.set_ylabel('Accuracy Difference (With - Without)')
    ax3.set_title('RT vs Accuracy Differences')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(ax3.collections[0], ax=ax3, label='Prevalence')
    
    # Plot 4: Effect sizes
    ax4 = axes[1, 1]
    # Filter out NaN values
    rt_effects = results_df['rt_cohens_d'].dropna()
    acc_effects = results_df['acc_cohens_d'].dropna()
    
    if len(rt_effects) > 0 and len(acc_effects) > 0:
        ax4.scatter(rt_effects, acc_effects, alpha=0.7, s=100)
        ax4.set_xlabel('RT Effect Size (Cohen\'s d)')
        ax4.set_ylabel('Accuracy Effect Size (Cohen\'s d)')
        ax4.set_title('RT vs Accuracy Effect Sizes')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'qid_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   Plots saved to: {OUTPUT_DIR / 'qid_performance_analysis.png'}")

def save_results(results_df, data_dict):
    """Save analysis results"""
    print("\n7. Saving results...")
    
    # Save detailed results
    results_df.to_csv(OUTPUT_DIR / 'qid_performance_analysis.csv', index=False)
    print(f"   Detailed results saved to: {OUTPUT_DIR / 'qid_performance_analysis.csv'}")
    
    # Save summary
    summary = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'total_qids': len(results_df),
        'qids_with_conditions': len(results_df[results_df['n_with'] > 0]),
        'qids_significant_rt': len(results_df[results_df['rt_p_value'] < 0.05]),
        'qids_significant_acc': len(results_df[results_df['acc_p_value'] < 0.05]),
        'data_dictionary': data_dict,
        'results': results_df.to_dict('records')
    }
    
    with open(OUTPUT_DIR / 'qid_performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   Summary saved to: {OUTPUT_DIR / 'qid_performance_summary.json'}")

def main():
    """Main function"""
    print("="*100)
    print("Comprehensive QID Analysis: RT and Accuracy")
    print("="*100)
    print("Analyzing all QIDs in the dataset")
    print()
    
    # Load data dictionary
    print("0. Loading data dictionary...")
    data_dict = load_data_dictionary()
    
    # Load and prepare data
    data, qid_columns = load_and_prepare_data()
    
    # Analyze each QID
    results = analyze_qid_performance(data, qid_columns, data_dict)
    
    # Create summary table
    results_df = create_summary_table(results)
    
    # Create visualizations
    create_visualizations(results_df)
    
    # Save results
    save_results(results_df, data_dict)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"Analyzed {len(results_df)} QIDs")
    print(f"QIDs with conditions: {len(results_df[results_df['n_with'] > 0])}")
    print(f"QIDs with significant RT differences: {len(results_df[results_df['rt_p_value'] < 0.05])}")
    print(f"QIDs with significant accuracy differences: {len(results_df[results_df['acc_p_value'] < 0.05])}")

if __name__ == "__main__":
    main()
