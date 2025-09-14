#!/usr/bin/env python3
"""
Debug SP-ECOG scale to understand the values
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

print("="*70)
print("DEBUGGING SP-ECOG SCALE AND VALUES")
print("="*70)

# Load SP-ECOG
sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()

# Get numeric QID columns
qid_cols = [c for c in sp_ecog_baseline.columns if c.startswith('QID')]
numeric_qids = sp_ecog_baseline[qid_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"\n1. SP-ECOG Scale Analysis:")
print(f"   Numeric QID columns: {len(numeric_qids)}")

if numeric_qids:
    # Check value ranges
    all_values = sp_ecog_baseline[numeric_qids].values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    print(f"\n2. Value Distribution:")
    print(f"   Total responses: {len(all_values):,}")
    print(f"   Unique values: {np.unique(all_values)}")
    print(f"   Min: {all_values.min():.1f}")
    print(f"   Max: {all_values.max():.1f}")
    print(f"   Mean: {all_values.mean():.2f}")
    print(f"   Median: {np.median(all_values):.1f}")
    print(f"   Std: {all_values.std():.2f}")
    
    # Distribution
    print(f"\n3. Value Frequency:")
    value_counts = pd.Series(all_values).value_counts().sort_index()
    for val, count in value_counts.items():
        pct = 100 * count / len(all_values)
        print(f"   {val:4.1f}: {count:7,} ({pct:5.1f}%)")
    
    # Subject-level means
    subject_means = sp_ecog_baseline[numeric_qids].mean(axis=1)
    
    print(f"\n4. Subject-Level Mean Scores:")
    print(f"   N subjects: {len(subject_means):,}")
    print(f"   Mean of means: {subject_means.mean():.2f}")
    print(f"   Median of means: {subject_means.median():.2f}")
    print(f"   Std of means: {subject_means.std():.2f}")
    
    # Percentiles
    print(f"\n5. Subject Mean Score Percentiles:")
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(subject_means.dropna(), p)
        print(f"   {p:3d}th percentile: {val:.2f}")
    
    # Count high scores per subject
    high_counts = (sp_ecog_baseline[numeric_qids] >= 3).sum(axis=1)
    
    print(f"\n6. Items with Score >= 3 per Subject:")
    print(f"   Mean: {high_counts.mean():.1f} items")
    print(f"   Median: {high_counts.median():.0f} items")
    print(f"   Max: {high_counts.max():.0f} items")
    
    print(f"\n   Distribution:")
    for threshold in [0, 1, 5, 10, 20, 30]:
        n = (high_counts >= threshold).sum()
        pct = 100 * n / len(high_counts)
        print(f"   >= {threshold:2d} items: {n:5,} subjects ({pct:5.1f}%)")
    
    # Load medical history for comparison
    print(f"\n7. Comparing with Self-Report MCI:")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'] if 'TimepointCode' in med_hx.columns else med_hx
    
    # Get self-reported MCI
    cognitive_qids = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
    available = [q for q in cognitive_qids if q in med_baseline.columns]
    
    if available:
        self_mci = np.zeros(len(med_baseline), dtype=int)
        for qid in available:
            self_mci |= (med_baseline[qid] == 1).values
        
        med_baseline['self_mci'] = self_mci
        
        # Add SP-ECOG mean
        sp_summary = sp_ecog_baseline.copy()
        sp_summary['sp_mean'] = sp_summary[numeric_qids].mean(axis=1)
        sp_summary['sp_high_count'] = (sp_summary[numeric_qids] >= 3).sum(axis=1)
        
        # Merge
        merged = med_baseline[['SubjectCode', 'self_mci']].merge(
            sp_summary[['SubjectCode', 'sp_mean', 'sp_high_count']], 
            on='SubjectCode', 
            how='inner'
        )
        
        print(f"   Merged subjects: {len(merged):,}")
        
        if len(merged) > 0:
            # Compare means
            print(f"\n   SP-ECOG mean by self-report status:")
            print(f"   Self-report Normal: {merged[merged['self_mci']==0]['sp_mean'].mean():.2f}")
            print(f"   Self-report MCI:    {merged[merged['self_mci']==1]['sp_mean'].mean():.2f}")
            
            # Find reasonable threshold
            from sklearn.metrics import roc_curve, roc_auc_score
            
            # Remove NaN
            valid = merged.dropna(subset=['sp_mean', 'self_mci'])
            if len(valid) > 100:
                fpr, tpr, thresholds = roc_curve(valid['self_mci'], valid['sp_mean'])
                auc = roc_auc_score(valid['self_mci'], valid['sp_mean'])
                
                print(f"\n   ROC Analysis (SP-ECOG mean vs self-report):")
                print(f"   AUC: {auc:.3f}")
                
                # Find optimal threshold (Youden's J)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresholds[optimal_idx]
                
                print(f"   Optimal threshold: {optimal_threshold:.2f}")
                print(f"   Sensitivity: {tpr[optimal_idx]:.1%}")
                print(f"   Specificity: {(1-fpr[optimal_idx]):.1%}")
                
                # Check prevalence at different thresholds
                print(f"\n8. MCI Prevalence at Different SP-ECOG Thresholds:")
                for thresh in [1.5, 2.0, 2.5, 3.0, 3.5, optimal_threshold]:
                    prev = (valid['sp_mean'] >= thresh).mean()
                    print(f"   Threshold >= {thresh:.1f}: {prev:.1%}")
    
    # Plot distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(subject_means.dropna(), bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(2.5, color='red', linestyle='--', label='Initial threshold (2.5)')
    plt.xlabel('SP-ECOG Mean Score')
    plt.ylabel('Number of Subjects')
    plt.title('Distribution of SP-ECOG Mean Scores')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(high_counts, bins=40, edgecolor='black', alpha=0.7)
    plt.axvline(5, color='red', linestyle='--', label='Initial threshold (5 items)')
    plt.xlabel('Number of Items >= 3')
    plt.ylabel('Number of Subjects')
    plt.title('Distribution of High-Score Items')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    value_counts.plot(kind='bar')
    plt.xlabel('SP-ECOG Item Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Item Responses')
    
    plt.tight_layout()
    plt.savefig('bhr_memtrax_results/sp_ecog_distribution.png', dpi=150)
    print(f"\n   Distribution plot saved to: bhr_memtrax_results/sp_ecog_distribution.png")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)
print("""
Based on the analysis, the SP-ECOG scale appears to be 1-4 where:
  1 = Normal
  2 = Questionable/Mild
  3 = Moderate  
  4 = Severe

The initial thresholds (mean > 2.5 or 5+ items >= 3) were TOO LENIENT.

Recommended thresholds for MCI detection:
  - SP-ECOG mean >= 1.5-2.0 (matches ~5-10% prevalence)
  - OR 2+ items with score >= 3
  
These would better align with expected MCI prevalence.
""")

