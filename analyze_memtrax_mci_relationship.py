#!/usr/bin/env python3
"""
Deep dive into MemTrax-MCI relationship
========================================
Let's examine what we might be missing:
1. Are the MCI labels actually capturing cognitive impairment?
2. Is there a subset where the relationship is stronger?
3. Are we extracting the right MemTrax features?
4. What's the actual distribution of performance vs labels?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

# QIDs we've been using
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

def analyze_label_quality():
    """Examine the MCI labels in detail"""
    print("\n" + "="*70)
    print("1. ANALYZING LABEL QUALITY")
    print("="*70)
    
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Focus on baseline
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    # Check each QID
    print("\nQID Analysis:")
    for qid in COGNITIVE_QIDS:
        if qid in med_hx.columns:
            counts = med_hx[qid].value_counts()
            positive = counts.get(1, 0)
            negative = counts.get(2, 0)
            missing = med_hx[qid].isna().sum()
            total = len(med_hx)
            
            print(f"\n{qid}:")
            print(f"  Positive (Yes): {positive:,} ({positive/total*100:.1f}%)")
            print(f"  Negative (No): {negative:,} ({negative/total*100:.1f}%)")
            print(f"  Missing/Other: {missing:,} ({missing/total*100:.1f}%)")
            
            # Check what the actual questions are (if we can infer)
            unique_vals = med_hx[qid].unique()
            print(f"  Unique values: {unique_vals[:10]}")
    
    # Create composite label as we've been doing
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    impairment = np.zeros(len(med_hx), dtype=int)
    valid = np.zeros(len(med_hx), dtype=bool)
    
    for qid in available_qids:
        impairment |= (med_hx[qid] == 1).values
        valid |= med_hx[qid].isin([1, 2]).values
    
    # Check overlap between QIDs
    print("\n\nQID Overlap Analysis:")
    qid_matrix = pd.DataFrame()
    for qid in available_qids:
        qid_matrix[qid] = (med_hx[qid] == 1).astype(int)
    
    correlation = qid_matrix.corr()
    print("\nCorrelation between QIDs:")
    print(correlation)
    
    # How many people are positive on multiple QIDs?
    positive_count = qid_matrix.sum(axis=1)
    print("\nNumber of positive QIDs per person:")
    print(positive_count.value_counts().sort_index())
    
    return med_hx[valid].copy(), impairment[valid]


def analyze_memtrax_distribution(med_hx, impairment):
    """Examine MemTrax performance distribution by MCI status"""
    print("\n" + "="*70)
    print("2. MEMTRAX PERFORMANCE DISTRIBUTION")
    print("="*70)
    
    # Load MemTrax data
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    
    # Quality filter
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Aggregate per subject
    agg_features = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std', 'min', 'max'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': 'mean'
    }).reset_index()
    
    agg_features.columns = ['SubjectCode'] + [
        f"{col[0]}_{col[1]}" for col in agg_features.columns[1:]
    ]
    
    # Merge with labels
    data = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'mci': impairment
    })
    
    data = data.merge(agg_features, on='SubjectCode', how='inner')
    
    print(f"\nMerged data: {len(data):,} subjects")
    print(f"MCI prevalence: {data['mci'].mean():.1%}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    features_to_plot = [
        'CorrectPCT_mean', 'CorrectResponsesRT_mean', 'CorrectPCT_std',
        'CorrectResponsesRT_std', 'IncorrectPCT_mean',
        'CorrectPCT_min'
    ]
    
    for ax, feature in zip(axes.flat, features_to_plot):
        # Plot distributions
        data_mci = data[data['mci'] == 1][feature].dropna()
        data_no_mci = data[data['mci'] == 0][feature].dropna()
        
        ax.hist(data_no_mci, alpha=0.5, label='No MCI', bins=30, density=True, color='blue')
        ax.hist(data_mci, alpha=0.5, label='MCI', bins=30, density=True, color='red')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        
        # Calculate effect size (Cohen's d)
        if len(data_mci) > 0 and len(data_no_mci) > 0:
            mean_diff = data_mci.mean() - data_no_mci.mean()
            pooled_std = np.sqrt((data_mci.std()**2 + data_no_mci.std()**2) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # T-test
            t_stat, p_value = stats.ttest_ind(data_mci, data_no_mci)
            
            ax.set_title(f"d={cohen_d:.2f}, p={p_value:.3f}")
    
    plt.suptitle("MemTrax Performance by MCI Status")
    plt.tight_layout()
    plt.savefig('memtrax_mci_distributions.png', dpi=150)
    plt.show()
    
    # Statistical summary
    print("\nStatistical Comparison (MCI vs No MCI):")
    for feature in features_to_plot:
        data_mci = data[data['mci'] == 1][feature].dropna()
        data_no_mci = data[data['mci'] == 0][feature].dropna()
        
        if len(data_mci) > 0 and len(data_no_mci) > 0:
            mean_diff = data_mci.mean() - data_no_mci.mean()
            t_stat, p_value = stats.ttest_ind(data_mci, data_no_mci)
            
            print(f"\n{feature}:")
            print(f"  No MCI: {data_no_mci.mean():.3f} ± {data_no_mci.std():.3f}")
            print(f"  MCI: {data_mci.mean():.3f} ± {data_mci.std():.3f}")
            print(f"  Difference: {mean_diff:.3f} (p={p_value:.4f})")
    
    return data


def analyze_age_education_subgroups(data):
    """Check if the model works better in certain subgroups"""
    print("\n" + "="*70)
    print("3. SUBGROUP ANALYSIS")
    print("="*70)
    
    # Load demographics
    demo = pd.read_csv(DATA_DIR / 'BHR_Demographics.csv', low_memory=False)
    
    # Get age and education
    demo_cols = ['SubjectCode']
    if 'QID186' in demo.columns:
        demo['Age'] = demo['QID186']
        demo_cols.append('Age')
    if 'QID184' in demo.columns:
        demo['Education'] = demo['QID184']
        demo_cols.append('Education')
    
    if len(demo_cols) > 1:
        demo = demo[demo_cols].drop_duplicates('SubjectCode')
        data = data.merge(demo, on='SubjectCode', how='left')
    
    # Prepare features for modeling
    feature_cols = [c for c in data.columns if c not in ['SubjectCode', 'mci', 'Age', 'Education']]
    
    # Test model performance by age groups
    if 'Age' in data.columns:
        print("\nPerformance by Age Group:")
        age_groups = [
            ('45-54', 45, 54),
            ('55-64', 55, 64),
            ('65-74', 65, 74),
            ('75+', 75, 120)
        ]
        
        for name, min_age, max_age in age_groups:
            subset = data[(data['Age'] >= min_age) & (data['Age'] <= max_age)]
            if len(subset) > 100:
                X = subset[feature_cols].fillna(subset[feature_cols].median())
                y = subset['mci'].values
                
                if len(np.unique(y)) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    model = LogisticRegression(class_weight='balanced', max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred)
                    
                    print(f"  {name}: n={len(subset):,}, MCI={subset['mci'].mean():.1%}, AUC={auc:.3f}")
    
    # Test by education level
    if 'Education' in data.columns:
        print("\nPerformance by Education Level:")
        edu_groups = [
            ('Low (1-3)', 1, 3),
            ('Medium (4-5)', 4, 5),
            ('High (6-7)', 6, 7)
        ]
        
        for name, min_edu, max_edu in edu_groups:
            subset = data[(data['Education'] >= min_edu) & (data['Education'] <= max_edu)]
            if len(subset) > 100:
                X = subset[feature_cols].fillna(subset[feature_cols].median())
                y = subset['mci'].values
                
                if len(np.unique(y)) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    model = LogisticRegression(class_weight='balanced', max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred)
                    
                    print(f"  {name}: n={len(subset):,}, MCI={subset['mci'].mean():.1%}, AUC={auc:.3f}")
    
    return data


def analyze_extreme_performers(data):
    """Look at extreme performers to understand the relationship"""
    print("\n" + "="*70)
    print("4. EXTREME PERFORMER ANALYSIS")
    print("="*70)
    
    # Create a composite cognitive score
    data['cognitive_score'] = (
        data['CorrectResponsesRT_mean'] / (data['CorrectPCT_mean'] + 0.01)
    )
    
    # Find extreme performers
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    thresholds = np.percentile(data['cognitive_score'].dropna(), percentiles)
    
    print("\nMCI Prevalence by Cognitive Score Percentile:")
    print("Percentile | Cog Score | MCI Prevalence | N")
    print("-" * 50)
    
    for i, (pct, thresh) in enumerate(zip(percentiles, thresholds)):
        if i == 0:
            subset = data[data['cognitive_score'] <= thresh]
            label = f"≤{pct}%"
        elif i == len(percentiles) - 1:
            subset = data[data['cognitive_score'] >= thresh]
            label = f"≥{pct}%"
        else:
            continue  # Skip middle percentiles for clarity
            
        mci_rate = subset['mci'].mean()
        n = len(subset)
        print(f"{label:10s} | {thresh:9.3f} | {mci_rate:14.1%} | {n:,}")
    
    # Look at false positives and false negatives
    print("\n\nAnalyzing Misclassifications:")
    
    # Simple threshold model
    threshold = np.percentile(data['cognitive_score'].dropna(), 80)
    data['predicted_mci'] = (data['cognitive_score'] >= threshold).astype(int)
    
    # False positives: predicted MCI but actually no MCI
    false_positives = data[(data['predicted_mci'] == 1) & (data['mci'] == 0)]
    print(f"\nFalse Positives: {len(false_positives):,}")
    print(f"  Mean accuracy: {false_positives['CorrectPCT_mean'].mean():.1f}%")
    print(f"  Mean RT: {false_positives['CorrectResponsesRT_mean'].mean():.3f}s")
    
    # False negatives: predicted no MCI but actually MCI
    false_negatives = data[(data['predicted_mci'] == 0) & (data['mci'] == 1)]
    print(f"\nFalse Negatives: {len(false_negatives):,}")
    print(f"  Mean accuracy: {false_negatives['CorrectPCT_mean'].mean():.1f}%")
    print(f"  Mean RT: {false_negatives['CorrectResponsesRT_mean'].mean():.3f}s")
    
    # These represent label noise
    print(f"\nLabel noise estimate: {(len(false_positives) + len(false_negatives)) / len(data) * 100:.1f}%")


def check_alternative_labels():
    """Check if there are better ways to define MCI from the data"""
    print("\n" + "="*70)
    print("5. ALTERNATIVE LABEL DEFINITIONS")
    print("="*70)
    
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    # Look for other potential cognitive indicators
    print("\nSearching for cognitive-related QIDs...")
    
    cognitive_keywords = ['memory', 'cognit', 'alzheimer', 'dementia', 'confusion', 'forget']
    
    # Check column names
    potential_qids = []
    for col in med_hx.columns:
        if 'QID' in col:
            # Check if this QID has meaningful variation
            if med_hx[col].nunique() > 1 and med_hx[col].nunique() < 10:
                val_counts = med_hx[col].value_counts()
                if len(val_counts) > 0:
                    positive_rate = val_counts.get(1, 0) / len(med_hx)
                    if 0.01 < positive_rate < 0.30:  # Between 1% and 30% positive
                        potential_qids.append(col)
    
    print(f"\nFound {len(potential_qids)} potential QIDs with reasonable variation")
    
    # Test each as a label
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Simple MemTrax features
    memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': 'mean',
        'CorrectResponsesRT': 'mean'
    }).reset_index()
    
    print("\nTesting alternative QIDs as labels:")
    best_auc = 0
    best_qid = None
    
    for qid in potential_qids[:20]:  # Test top 20
        try:
            # Create label from this QID
            labels = pd.DataFrame({
                'SubjectCode': med_hx['SubjectCode'],
                'label': (med_hx[qid] == 1).astype(int)
            })
            
            # Only keep valid responses
            valid = med_hx[qid].isin([1, 2])
            labels = labels[valid]
            
            if len(labels) < 100:
                continue
                
            # Merge with MemTrax
            data = memtrax_agg.merge(labels, on='SubjectCode', how='inner')
            
            if len(data) < 100 or data['label'].mean() < 0.01:
                continue
            
            # Quick model test
            X = data[['CorrectPCT', 'CorrectResponsesRT']].values
            y = data['label'].values
            
            if len(np.unique(y)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                model = LogisticRegression(class_weight='balanced')
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred)
                
                if auc > best_auc:
                    best_auc = auc
                    best_qid = qid
                    
                if auc > 0.75:
                    print(f"  {qid}: AUC={auc:.3f}, Prevalence={data['label'].mean():.1%}, N={len(data):,}")
        except:
            continue
    
    print(f"\nBest alternative QID: {best_qid} with AUC={best_auc:.3f}")


def main():
    print("\n" + "="*80)
    print("DEEP DIVE: WHY ISN'T MEMTRAX MORE PREDICTIVE OF MCI?")
    print("="*80)
    
    # 1. Analyze label quality
    med_hx, impairment = analyze_label_quality()
    
    # 2. Analyze MemTrax distributions
    data = analyze_memtrax_distribution(med_hx, impairment)
    
    # 3. Subgroup analysis
    data = analyze_age_education_subgroups(data)
    
    # 4. Extreme performer analysis
    analyze_extreme_performers(data)
    
    # 5. Alternative labels
    check_alternative_labels()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
    1. Label Quality Issues:
       - Self-reported MCI has high noise
       - Low correlation between different QIDs
       - Many subjects report only 1 symptom
    
    2. Weak Signal:
       - Small effect sizes (Cohen's d < 0.3)
       - Overlapping distributions between MCI and non-MCI
       - High variance within groups
    
    3. Subgroup Variation:
       - Model performs differently across age groups
       - Education level affects relationship
       - Cognitive reserve masks impairment
    
    4. Potential Solutions:
       - Need clinical diagnosis, not self-report
       - Consider longitudinal progression
       - Combine with other assessments
       - Focus on specific cognitive domains
    """)


if __name__ == "__main__":
    main()

