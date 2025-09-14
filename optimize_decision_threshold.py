#!/usr/bin/env python3
"""
Decision Threshold Optimization for MCI Detection
==================================================
Demonstrates how changing the probability threshold affects performance
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report,
    roc_auc_score, f1_score
)

# Quick setup for demonstration
np.random.seed(42)
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def demonstrate_threshold_impact():
    """
    Show how decision threshold affects classification
    """
    
    print("\n" + "="*70)
    print("UNDERSTANDING DECISION THRESHOLDS IN MCI DETECTION")
    print("="*70)
    
    print("\n📚 WHAT IS A DECISION THRESHOLD?")
    print("-" * 40)
    print("""
When a model predicts MCI, it actually outputs a PROBABILITY (0 to 1):
- 0.0 = Definitely normal
- 0.5 = Uncertain  
- 1.0 = Definitely MCI

The DECISION THRESHOLD determines when we classify as MCI:
- Default: probability >= 0.5 → Predict MCI
- But 0.5 is arbitrary! We can choose any threshold.
    """)
    
    print("\n🎯 WHY DOES THIS MATTER FOR MCI?")
    print("-" * 40)
    print("""
MCI detection has asymmetric costs:
- FALSE NEGATIVE (missing MCI): Patient doesn't get early intervention
- FALSE POSITIVE (false alarm): Unnecessary worry, but follow-up clarifies

Therefore, we might prefer HIGH SENSITIVITY (catch all MCI cases)
even if it means more false positives.
    """)
    
    # Load and prepare data (simplified version)
    print("\n1. Loading BHR data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Quick quality filter
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Simple features
    features = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std'],
        'CorrectResponsesRT': ['mean', 'std'],
        'IncorrectPCT': 'mean',
        'IncorrectResponsesRT': 'mean'
    })
    features.columns = ['_'.join(col) for col in features.columns]
    features = features.reset_index()
    
    # Labels
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    cognitive_qids = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
    available_qids = [q for q in cognitive_qids if q in med_hx.columns]
    
    impairment = np.zeros(len(med_hx), dtype=int)
    for qid in available_qids:
        impairment |= (med_hx[qid] == 1).values
    
    labels = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'mci': impairment
    })
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    X = data.drop(['SubjectCode', 'mci'], axis=1)
    y = data['mci']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Dataset: {len(data)} subjects")
    print(f"  MCI prevalence: {y.mean():.1%}")
    print(f"  Test set: {len(X_test)} subjects")
    
    # Train model
    print("\n2. Training model...")
    model = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n3. COMPARING DIFFERENT THRESHOLDS")
    print("="*70)
    
    # Test different thresholds
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n📊 Performance at Different Thresholds:\n")
    print(f"{'Threshold':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10} {'Predicted MCI%':<15}")
    print("-"*60)
    
    best_f1 = 0
    best_threshold = 0.5
    results = []
    
    for threshold in thresholds_to_test:
        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        predicted_mci_pct = y_pred.mean()
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'predicted_pct': predicted_mci_pct
        })
        
        print(f"{threshold:<10.1f} {sensitivity:<12.1%} {specificity:<12.1%} {f1:<10.3f} {predicted_mci_pct:<15.1%}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    
    # Show specific examples
    default_results = next(r for r in results if r['threshold'] == 0.5)
    optimal_results = next(r for r in results if r['threshold'] == best_threshold)
    sensitive_results = next(r for r in results if r['threshold'] == 0.3)
    
    print(f"""
1. DEFAULT (threshold = 0.5):
   - Sensitivity: {default_results['sensitivity']:.1%}
   - Means we MISS {100-default_results['sensitivity']*100:.0f}% of MCI cases!
   
2. OPTIMIZED FOR F1 (threshold = {best_threshold}):
   - Sensitivity: {optimal_results['sensitivity']:.1%}
   - Better balance of precision/recall
   
3. HIGH SENSITIVITY (threshold = 0.3):
   - Sensitivity: {sensitive_results['sensitivity']:.1%}
   - Catches almost all MCI cases
   - But predicts {sensitive_results['predicted_pct']:.1%} have MCI (vs true {y_test.mean():.1%})
    """)
    
    # Find optimal threshold for specific sensitivity target
    print("\n4. FINDING OPTIMAL THRESHOLD FOR CLINICAL GOALS")
    print("-"*50)
    
    target_sensitivity = 0.80  # Want to catch 80% of MCI cases
    
    # Use ROC curve to find threshold
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    
    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    optimal_threshold = thresholds_roc[idx]
    achieved_sensitivity = tpr[idx]
    achieved_specificity = 1 - fpr[idx]
    
    print(f"\nTo achieve {target_sensitivity:.0%} sensitivity:")
    print(f"  → Use threshold: {optimal_threshold:.3f}")
    print(f"  → Actual sensitivity: {achieved_sensitivity:.1%}")
    print(f"  → Specificity: {achieved_specificity:.1%}")
    
    # Apply this threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    # Show confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
    
    print(f"\nConfusion Matrix at threshold {optimal_threshold:.3f}:")
    print(f"  True Negatives:  {tn:5d} (correctly identified as normal)")
    print(f"  False Positives: {fp:5d} (false alarms)")
    print(f"  False Negatives: {fn:5d} (missed MCI cases)")
    print(f"  True Positives:  {tp:5d} (correctly identified MCI)")
    
    # Clinical interpretation
    print("\n5. CLINICAL INTERPRETATION")
    print("-"*50)
    
    n_mci = tp + fn
    n_normal = tn + fp
    
    print(f"""
In our test set of {len(y_test)} patients:
- {n_mci} actually have MCI
- {n_normal} are normal

With DEFAULT threshold (0.5):
- We correctly identify {int(default_results['sensitivity'] * n_mci)}/{n_mci} MCI cases
- We miss {int((1-default_results['sensitivity']) * n_mci)} MCI patients!

With OPTIMIZED threshold ({optimal_threshold:.3f}):
- We correctly identify {tp}/{n_mci} MCI cases
- We miss only {fn} MCI patients
- But we have {fp} false alarms (worth it for early detection!)
    """)
    
    # Visualization
    print("\n6. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Probability distribution
    ax1 = axes[0, 0]
    ax1.hist(y_proba[y_test==0], bins=30, alpha=0.5, label='Normal', color='blue')
    ax1.hist(y_proba[y_test==1], bins=30, alpha=0.5, label='MCI', color='red')
    ax1.axvline(0.5, color='black', linestyle='--', label='Default (0.5)')
    ax1.axvline(optimal_threshold, color='green', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Predicted Probabilities')
    ax1.legend()
    
    # 2. Threshold vs Metrics
    ax2 = axes[0, 1]
    threshold_range = np.linspace(0.1, 0.9, 50)
    sensitivities = []
    specificities = []
    f1_scores = []
    
    for t in threshold_range:
        y_p = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_p).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_p)
        sensitivities.append(sens)
        specificities.append(spec)
        f1_scores.append(f1)
    
    ax2.plot(threshold_range, sensitivities, label='Sensitivity', color='red')
    ax2.plot(threshold_range, specificities, label='Specificity', color='blue')
    ax2.plot(threshold_range, f1_scores, label='F1-Score', color='green')
    ax2.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Performance')
    ax2.set_title('Performance vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. ROC Curve with thresholds
    ax3 = axes[1, 0]
    ax3.plot(fpr, tpr, color='blue', label=f'ROC (AUC={roc_auc_score(y_test, y_proba):.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Mark specific thresholds
    for t in [0.3, 0.5, optimal_threshold, 0.7]:
        idx = np.argmin(np.abs(thresholds_roc - t))
        ax3.plot(fpr[idx], tpr[idx], 'o', markersize=8)
        ax3.annotate(f'{t:.2f}', (fpr[idx], tpr[idx]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate (Sensitivity)')
    ax3.set_title('ROC Curve with Threshold Points')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    ax4 = axes[1, 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    ax4.plot(recall, precision, color='green')
    ax4.axhline(y_test.mean(), color='red', linestyle='--', 
                label=f'Baseline (prevalence={y_test.mean():.1%})')
    ax4.set_xlabel('Recall (Sensitivity)')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to: bhr_memtrax_results/threshold_optimization.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: DECISION THRESHOLD OPTIMIZATION")
    print("="*70)
    print(f"""
The AUC stays at {roc_auc_score(y_test, y_proba):.3f} regardless of threshold.
But the PRACTICAL PERFORMANCE changes dramatically:

DEFAULT (0.5):
  → Sensitivity: {default_results['sensitivity']:.1%}
  → Misses many MCI cases

OPTIMIZED ({optimal_threshold:.3f}):
  → Sensitivity: {achieved_sensitivity:.1%}
  → Better for clinical screening

KEY INSIGHT:
- Don't use default 0.5 threshold blindly!
- Choose threshold based on clinical goals
- For MCI screening, prefer high sensitivity
- This is "free" performance improvement
    """)
    
    return optimal_threshold, achieved_sensitivity


if __name__ == '__main__':
    optimal_threshold, sensitivity = demonstrate_threshold_impact()
    print(f"\n✅ Recommended threshold for MCI screening: {optimal_threshold:.3f}")
    print(f"   (Achieves {sensitivity:.1%} sensitivity)\n")

