#!/usr/bin/env python3
"""
Analyze MemTrax RT by Cognitive Medical Conditions
=================================================

This script analyzes the relationship between MemTrax RT and specific cognitive
medical conditions (QIDs), ignoring accuracy due to ceiling effects.

We'll examine:
- Average RT for each cognitive QID condition
- RT distribution for subjects with no cognitive conditions
- Statistical significance of RT differences
- Effect sizes for each condition
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

# Cognitive QIDs from the best script
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# QID descriptions for better interpretation
QID_DESCRIPTIONS = {
    'QID1-5': 'Mild Cognitive Impairment (MCI)',
    'QID1-12': 'Alzheimer\'s Disease',
    'QID1-13': 'Dementia',
    'QID1-22': 'Parkinson\'s Disease',
    'QID1-23': 'Other Neurological Condition'
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
    
    # Compute RT aggregates per subject
    print("3. Computing RT aggregates...")
    rt_agg = memtrax_q.groupby('SubjectCode').agg({
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max', 'count']
    })
    rt_agg.columns = ['rt_mean', 'rt_std', 'rt_min', 'rt_max', 'rt_count']
    rt_agg = rt_agg.reset_index()
    
    # Filter subjects with sufficient data
    rt_agg = rt_agg[rt_agg['rt_count'] >= 3]  # At least 3 valid RT measurements
    print(f"   Subjects with sufficient RT data: {len(rt_agg)}")
    
    # Load medical history
    print("4. Loading medical history...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    
    # Filter to baseline timepoint
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    print(f"   Medical history records: {len(med_hx)}")
    
    # Merge RT data with medical history
    print("5. Merging RT and medical data...")
    data = rt_agg.merge(med_hx[['SubjectCode'] + COGNITIVE_QIDS], on='SubjectCode', how='inner')
    print(f"   Final dataset: {len(data)} subjects")
    
    return data

def analyze_rt_by_conditions(data):
    """Analyze RT by each cognitive condition"""
    print("\n" + "="*80)
    print("RT ANALYSIS BY COGNITIVE CONDITIONS")
    print("="*80)
    
    results = {}
    
    # Analyze each QID
    for qid in COGNITIVE_QIDS:
        if qid not in data.columns:
            print(f"   {qid} not found in data")
            continue
            
        print(f"\n--- {qid}: {QID_DESCRIPTIONS.get(qid, 'Unknown')} ---")
        
        # Get subjects with and without this condition
        has_condition = data[data[qid] == 1]
        no_condition = data[data[qid] == 0]
        
        print(f"   Subjects with condition: {len(has_condition)}")
        print(f"   Subjects without condition: {len(no_condition)}")
        
        if len(has_condition) == 0:
            print("   No subjects with this condition - skipping")
            continue
            
        # RT statistics
        rt_with = has_condition['rt_mean']
        rt_without = no_condition['rt_mean']
        
        print(f"\n   RT Statistics:")
        print(f"   With condition:    Mean={rt_with.mean():.4f}, Std={rt_with.std():.4f}, N={len(rt_with)}")
        print(f"   Without condition: Mean={rt_without.mean():.4f}, Std={rt_without.std():.4f}, N={len(rt_without)}")
        
        # Statistical test
        if len(rt_with) >= 3 and len(rt_without) >= 3:
            try:
                # Mann-Whitney U test (non-parametric, handles non-normal distributions)
                statistic, p_value = stats.mannwhitneyu(rt_with, rt_without, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(rt_with) - 1) * rt_with.var() + 
                                    (len(rt_without) - 1) * rt_without.var()) / 
                                   (len(rt_with) + len(rt_without) - 2))
                cohens_d = (rt_with.mean() - rt_without.mean()) / pooled_std
                
                print(f"\n   Statistical Test (Mann-Whitney U):")
                print(f"   p-value: {p_value:.6f}")
                print(f"   Effect size (Cohen's d): {cohens_d:.4f}")
                
                # Interpretation
                if p_value < 0.001:
                    sig_level = "***"
                elif p_value < 0.01:
                    sig_level = "**"
                elif p_value < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"
                
                print(f"   Significance: {sig_level}")
                
                # Effect size interpretation
                if abs(cohens_d) >= 0.8:
                    effect_size = "Large"
                elif abs(cohens_d) >= 0.5:
                    effect_size = "Medium"
                elif abs(cohens_d) >= 0.2:
                    effect_size = "Small"
                else:
                    effect_size = "Negligible"
                
                print(f"   Effect size: {effect_size}")
                
                # Store results
                results[qid] = {
                    'description': QID_DESCRIPTIONS.get(qid, 'Unknown'),
                    'n_with': len(rt_with),
                    'n_without': len(rt_without),
                    'rt_with_mean': rt_with.mean(),
                    'rt_with_std': rt_with.std(),
                    'rt_without_mean': rt_without.mean(),
                    'rt_without_std': rt_without.std(),
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significance': sig_level,
                    'effect_size': effect_size
                }
                
            except Exception as e:
                print(f"   Statistical test failed: {e}")
                results[qid] = {
                    'description': QID_DESCRIPTIONS.get(qid, 'Unknown'),
                    'n_with': len(rt_with),
                    'n_without': len(rt_without),
                    'rt_with_mean': rt_with.mean(),
                    'rt_with_std': rt_with.std(),
                    'rt_without_mean': rt_without.mean(),
                    'rt_without_std': rt_without.std(),
                    'p_value': np.nan,
                    'cohens_d': np.nan,
                    'significance': 'Error',
                    'effect_size': 'Error'
                }
        else:
            print("   Insufficient data for statistical test")
            results[qid] = {
                'description': QID_DESCRIPTIONS.get(qid, 'Unknown'),
                'n_with': len(rt_with),
                'n_without': len(rt_without),
                'rt_with_mean': rt_with.mean(),
                'rt_with_std': rt_with.std(),
                'rt_without_mean': rt_without.mean(),
                'rt_without_std': rt_without.std(),
                'p_value': np.nan,
                'cohens_d': np.nan,
                'significance': 'Insufficient data',
                'effect_size': 'Insufficient data'
            }
    
    return results

def analyze_no_cognitive_conditions(data):
    """Analyze RT for subjects with no cognitive conditions"""
    print("\n" + "="*80)
    print("RT ANALYSIS: NO COGNITIVE CONDITIONS")
    print("="*80)
    
    # Create a column indicating if subject has any cognitive condition
    data['has_any_cognitive'] = data[COGNITIVE_QIDS].sum(axis=1) > 0
    
    no_cognitive = data[~data['has_any_cognitive']]
    has_cognitive = data[data['has_any_cognitive']]
    
    print(f"Subjects with no cognitive conditions: {len(no_cognitive)}")
    print(f"Subjects with any cognitive condition: {len(has_cognitive)}")
    
    if len(no_cognitive) > 0:
        print(f"\nRT Statistics for No Cognitive Conditions:")
        print(f"Mean RT: {no_cognitive['rt_mean'].mean():.4f}")
        print(f"Std RT:  {no_cognitive['rt_mean'].std():.4f}")
        print(f"Min RT:  {no_cognitive['rt_mean'].min():.4f}")
        print(f"Max RT:  {no_cognitive['rt_mean'].max():.4f}")
        print(f"Median RT: {no_cognitive['rt_mean'].median():.4f}")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\nRT Percentiles:")
        for p in percentiles:
            value = np.percentile(no_cognitive['rt_mean'], p)
            print(f"  {p}th percentile: {value:.4f}")
    
    if len(has_cognitive) > 0:
        print(f"\nRT Statistics for Any Cognitive Condition:")
        print(f"Mean RT: {has_cognitive['rt_mean'].mean():.4f}")
        print(f"Std RT:  {has_cognitive['rt_mean'].std():.4f}")
        print(f"Min RT:  {has_cognitive['rt_mean'].min():.4f}")
        print(f"Max RT:  {has_cognitive['rt_mean'].max():.4f}")
        print(f"Median RT: {has_cognitive['rt_mean'].median():.4f}")
        
        # Statistical comparison
        if len(no_cognitive) >= 3 and len(has_cognitive) >= 3:
            try:
                statistic, p_value = stats.mannwhitneyu(no_cognitive['rt_mean'], 
                                                      has_cognitive['rt_mean'], 
                                                      alternative='two-sided')
                
                # Effect size
                pooled_std = np.sqrt(((len(no_cognitive) - 1) * no_cognitive['rt_mean'].var() + 
                                    (len(has_cognitive) - 1) * has_cognitive['rt_mean'].var()) / 
                                   (len(no_cognitive) + len(has_cognitive) - 2))
                cohens_d = (no_cognitive['rt_mean'].mean() - has_cognitive['rt_mean'].mean()) / pooled_std
                
                print(f"\nStatistical Comparison (No Cognitive vs Any Cognitive):")
                print(f"p-value: {p_value:.6f}")
                print(f"Effect size (Cohen's d): {cohens_d:.4f}")
                
                if p_value < 0.001:
                    print("Significance: ***")
                elif p_value < 0.01:
                    print("Significance: **")
                elif p_value < 0.05:
                    print("Significance: *")
                else:
                    print("Significance: ns")
                    
            except Exception as e:
                print(f"Statistical comparison failed: {e}")
    
    return no_cognitive, has_cognitive

def create_rt_distribution_plots(data, results, no_cognitive, has_cognitive):
    """Create visualization plots for RT distributions"""
    print("\n6. Creating RT distribution plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MemTrax RT Analysis by Cognitive Conditions', fontsize=16, fontweight='bold')
    
    # Plot 1: RT distribution by condition status
    ax1 = axes[0, 0]
    ax1.hist(no_cognitive['rt_mean'], bins=50, alpha=0.7, label='No Cognitive Conditions', density=True)
    ax1.hist(has_cognitive['rt_mean'], bins=50, alpha=0.7, label='Any Cognitive Condition', density=True)
    ax1.set_xlabel('Mean RT (seconds)')
    ax1.set_ylabel('Density')
    ax1.set_title('RT Distribution: No Cognitive vs Any Cognitive')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot for each QID
    ax2 = axes[0, 1]
    qid_data = []
    qid_labels = []
    
    for qid in COGNITIVE_QIDS:
        if qid in data.columns:
            has_condition = data[data[qid] == 1]['rt_mean']
            if len(has_condition) > 0:
                qid_data.append(has_condition)
                qid_labels.append(f"{qid}\n({QID_DESCRIPTIONS.get(qid, 'Unknown')})")
    
    if qid_data:
        ax2.boxplot(qid_data, labels=qid_labels)
        ax2.set_ylabel('Mean RT (seconds)')
        ax2.set_title('RT Distribution by Cognitive Condition')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: RT vs condition count
    ax3 = axes[1, 0]
    condition_counts = data[COGNITIVE_QIDS].sum(axis=1)
    ax3.scatter(condition_counts, data['rt_mean'], alpha=0.6, s=20)
    ax3.set_xlabel('Number of Cognitive Conditions')
    ax3.set_ylabel('Mean RT (seconds)')
    ax3.set_title('RT vs Number of Cognitive Conditions')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    if len(condition_counts.unique()) > 1:
        z = np.polyfit(condition_counts, data['rt_mean'], 1)
        p = np.poly1d(z)
        ax3.plot(condition_counts, p(condition_counts), "r--", alpha=0.8)
    
    # Plot 4: RT percentiles by condition status
    ax4 = axes[1, 1]
    percentiles = [10, 25, 50, 75, 90, 95]
    
    no_cog_percentiles = [np.percentile(no_cognitive['rt_mean'], p) for p in percentiles]
    has_cog_percentiles = [np.percentile(has_cognitive['rt_mean'], p) for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    
    ax4.bar(x - width/2, no_cog_percentiles, width, label='No Cognitive', alpha=0.8)
    ax4.bar(x + width/2, has_cog_percentiles, width, label='Any Cognitive', alpha=0.8)
    
    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('RT (seconds)')
    ax4.set_title('RT Percentiles by Condition Status')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{p}th' for p in percentiles])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rt_by_cognitive_conditions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   Plots saved to: {OUTPUT_DIR / 'rt_by_cognitive_conditions.png'}")

def save_results(results, no_cognitive, has_cognitive):
    """Save analysis results to files"""
    print("\n7. Saving results...")
    
    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(OUTPUT_DIR / 'rt_by_cognitive_conditions_detailed.csv')
    print(f"   Detailed results saved to: {OUTPUT_DIR / 'rt_by_cognitive_conditions_detailed.csv'}")
    
    # Save summary
    summary = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'total_subjects': len(no_cognitive) + len(has_cognitive),
        'subjects_no_cognitive': len(no_cognitive),
        'subjects_any_cognitive': len(has_cognitive),
        'rt_no_cognitive_mean': float(no_cognitive['rt_mean'].mean()) if len(no_cognitive) > 0 else None,
        'rt_any_cognitive_mean': float(has_cognitive['rt_mean'].mean()) if len(has_cognitive) > 0 else None,
        'qid_results': results
    }
    
    with open(OUTPUT_DIR / 'rt_by_cognitive_conditions_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   Summary saved to: {OUTPUT_DIR / 'rt_by_cognitive_conditions_summary.json'}")

def main():
    """Main function"""
    print("="*80)
    print("MemTrax RT Analysis by Cognitive Medical Conditions")
    print("="*80)
    print("Analyzing RT patterns for each cognitive QID condition")
    print("Ignoring accuracy due to ceiling effects")
    print()
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Analyze RT by each condition
    results = analyze_rt_by_conditions(data)
    
    # Analyze subjects with no cognitive conditions
    no_cognitive, has_cognitive = analyze_no_cognitive_conditions(data)
    
    # Create visualizations
    create_rt_distribution_plots(data, results, no_cognitive, has_cognitive)
    
    # Save results
    save_results(results, no_cognitive, has_cognitive)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Key findings:")
    print(f"- Subjects with no cognitive conditions: {len(no_cognitive)}")
    print(f"- Subjects with any cognitive condition: {len(has_cognitive)}")
    if len(no_cognitive) > 0 and len(has_cognitive) > 0:
        rt_diff = has_cognitive['rt_mean'].mean() - no_cognitive['rt_mean'].mean()
        print(f"- RT difference (cognitive - no cognitive): {rt_diff:.4f} seconds")
        print(f"- RT increase for cognitive conditions: {(rt_diff/no_cognitive['rt_mean'].mean()*100):.1f}%")

if __name__ == "__main__":
    main()
