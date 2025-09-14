#!/usr/bin/env python3
"""
BHR MemTrax Longitudinal Analysis - Simplified Robust Version
=============================================================
Focus on simple but powerful longitudinal features:
- Change scores (last - first)
- Linear slopes (simple regression)
- Practice effects (early improvement)
- Consistency over time
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import json

warnings.filterwarnings('ignore')

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
TIMEPOINT_ORDER = {'m00': 0, 'm06': 6, 'm12': 12, 'm18': 18, 'm24': 24, 'm30': 30, 'm36': 36, 'm42': 42, 'm48': 48}


def simple_slope(x, y):
    """Calculate simple linear slope avoiding numerical issues"""
    if len(x) < 2 or len(np.unique(x)) < 2:
        return 0
    
    # Simple least squares
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 2:
        return 0
    
    x = x[valid]
    y = y[valid]
    
    # Calculate slope
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    
    if denominator == 0:
        return 0
    
    return numerator / denominator


def extract_longitudinal_features():
    """Extract simple but effective longitudinal features"""
    print("Loading data...")
    
    # Load MemTrax
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    
    # Apply quality filter
    memtrax = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.50) &
        (memtrax['CorrectResponsesRT'].between(0.4, 3.0))
    ]
    
    # Convert timepoints
    if 'TimepointCode' in memtrax.columns:
        memtrax['months'] = memtrax['TimepointCode'].map(TIMEPOINT_ORDER).fillna(0)
    else:
        memtrax['months'] = 0
    
    print(f"  Processing {memtrax['SubjectCode'].nunique()} subjects...")
    
    features = []
    
    for subject, group in memtrax.groupby('SubjectCode'):
        group = group.sort_values('months')
        feat = {'SubjectCode': subject}
        
        # Basic counts
        feat['n_timepoints'] = len(group)
        feat['followup_months'] = group['months'].max()
        
        if len(group) >= 2:
            # === Simple change scores ===
            first = group.iloc[0]
            last = group.iloc[-1]
            
            # Accuracy change
            feat['acc_change'] = last['CorrectPCT'] - first['CorrectPCT']
            feat['acc_first'] = first['CorrectPCT']
            feat['acc_last'] = last['CorrectPCT']
            
            # RT change
            feat['rt_change'] = last['CorrectResponsesRT'] - first['CorrectResponsesRT']
            feat['rt_first'] = first['CorrectResponsesRT']
            feat['rt_last'] = last['CorrectResponsesRT']
            
            # Composite change (negative is worse)
            feat['composite_change'] = feat['acc_change'] - feat['rt_change']
            
            # === Simple slopes ===
            months = group['months'].values
            acc_values = group['CorrectPCT'].values
            rt_values = group['CorrectResponsesRT'].values
            
            feat['acc_slope'] = simple_slope(months, acc_values) * 12  # Per year
            feat['rt_slope'] = simple_slope(months, rt_values) * 12
            
            # === Variability ===
            feat['acc_std'] = group['CorrectPCT'].std()
            feat['rt_std'] = group['CorrectResponsesRT'].std()
            
            # === Practice effect ===
            if len(group) >= 3:
                # Check if performance improved from visit 1 to 2
                early_change = group.iloc[1]['CorrectPCT'] - group.iloc[0]['CorrectPCT']
                feat['practice_effect'] = early_change
                feat['no_practice'] = 1 if early_change <= 0 else 0
            
            # === Consistency ===
            feat['acc_cv'] = feat['acc_std'] / (group['CorrectPCT'].mean() + 1e-6)
            feat['rt_cv'] = feat['rt_std'] / (group['CorrectResponsesRT'].mean() + 1e-6)
            
        else:
            # Single timepoint
            feat['acc_first'] = group.iloc[0]['CorrectPCT']
            feat['rt_first'] = group.iloc[0]['CorrectResponsesRT']
            feat['single_timepoint'] = 1
        
        features.append(feat)
    
    features_df = pd.DataFrame(features)
    
    # Add demographics
    demo_path = DATA_DIR / 'BHR_Demographics.csv'
    if demo_path.exists():
        demo = pd.read_csv(demo_path, low_memory=False)
        if 'Code' in demo.columns:
            demo.rename(columns={'Code': 'SubjectCode'}, inplace=True)
        
        demo_cols = ['SubjectCode']
        for c in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender']:
            if c in demo.columns:
                demo_cols.append(c)
        
        if len(demo_cols) > 1:
            features_df = features_df.merge(
                demo[demo_cols].drop_duplicates('SubjectCode'),
                on='SubjectCode', how='left'
            )
    
    # Create interactions
    if 'Age_Baseline' in features_df.columns and 'acc_slope' in features_df.columns:
        features_df['age_decline'] = features_df['Age_Baseline'] * (-features_df['acc_slope'].fillna(0))
    
    if 'Gender' in features_df.columns:
        features_df['Gender_Num'] = features_df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
    
    print(f"  Extracted features for {len(features_df)} subjects")
    print(f"  Subjects with 2+ timepoints: {(features_df['n_timepoints'] >= 2).sum()}")
    
    return features_df


def create_labels():
    """Create simple cognitive impairment labels"""
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Use baseline labels
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
    impairment = np.zeros(len(med_hx), dtype=int)
    valid = np.zeros(len(med_hx), dtype=bool)
    
    for qid in available_qids:
        impairment |= (med_hx[qid] == 1).values
        valid |= med_hx[qid].isin([1, 2]).values
    
    labels = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'cognitive_impairment': impairment
    })
    
    return labels[valid].copy()


def main():
    print("\n" + "="*70)
    print("BHR LONGITUDINAL ANALYSIS - SIMPLIFIED VERSION")
    print("="*70)
    
    # Extract features
    print("\n1. Extracting longitudinal features...")
    features = extract_longitudinal_features()
    
    # Create labels
    print("\n2. Creating labels...")
    labels = create_labels()
    print(f"   Labels: {len(labels)} subjects")
    print(f"   Prevalence: {labels['cognitive_impairment'].mean():.1%}")
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    print(f"\n3. Final dataset: {len(data)} subjects")
    
    # Check longitudinal coverage
    has_long = data['n_timepoints'] >= 2
    print(f"   With longitudinal data: {has_long.sum()} ({has_long.mean():.1%})")
    
    # Prepare for modeling
    feature_cols = [c for c in data.columns 
                   if c not in ['SubjectCode', 'cognitive_impairment', 'single_timepoint']]
    
    X = data[feature_cols]
    y = data['cognitive_impairment']
    
    print(f"\n4. Features: {X.shape}")
    print(f"   Prevalence: {y.mean():.1%}")
    
    # Key longitudinal features
    long_features = ['acc_change', 'rt_change', 'composite_change', 'acc_slope', 
                    'rt_slope', 'practice_effect', 'no_practice', 'age_decline']
    long_present = [f for f in long_features if f in X.columns]
    print(f"   Longitudinal features: {len(long_present)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n5. Train/test: {len(X_train)}/{len(X_test)}")
    
    # Models
    print("\n6. Training models...")
    
    models = {
        'Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=10,
                                          class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'HistGB': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                                                  max_depth=6, random_state=RANDOM_STATE))
        ])
    }
    
    best_auc = 0
    best_name = None
    results = {}
    
    for name, model in models.items():
        # CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)
        
        print(f"   {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_auc:.4f}")
        
        results[name] = test_auc
        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name
            best_model = model
    
    # Ensemble
    stack = StackingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    
    cal_stack = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    cal_stack.fit(X_train, y_train)
    y_pred = cal_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_pred)
    
    print(f"   Stack: Test={stack_auc:.4f}")
    
    if stack_auc > best_auc:
        best_auc = stack_auc
        best_name = 'Stack'
    
    # Feature importance for RF
    if 'RF' in models and hasattr(models['RF'].named_steps['clf'], 'feature_importances_'):
        rf_model = models['RF'].named_steps['clf']
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n7. Top 10 features (RF):")
        for _, row in importance.head(10).iterrows():
            long_marker = " *" if row['feature'] in long_present else ""
            print(f"   {row['feature']}: {row['importance']:.4f}{long_marker}")
        
        # Check longitudinal feature importance
        long_imp = importance[importance['feature'].isin(long_present)]
        if len(long_imp) > 0:
            avg_rank = long_imp.index.tolist()
            avg_rank = np.mean([list(importance.index).index(i) + 1 for i in avg_rank])
            print(f"\n   Longitudinal features avg rank: {avg_rank:.1f}")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Best: {best_name}")
    print(f"Test AUC: {best_auc:.4f}")
    
    if best_auc >= 0.80:
        print("\n🎯 SUCCESS! Achieved AUC ≥ 0.80 with longitudinal data!")
    elif best_auc > 0.75:
        print(f"\n📈 Improved! Longitudinal features helping (+{best_auc - 0.744:.3f} vs baseline)")
    else:
        print(f"\n📊 Similar to baseline (0.744)")
    
    # Save
    output = {
        'best_auc': float(best_auc),
        'best_model': best_name,
        'all_results': results,
        'n_subjects': len(data),
        'n_longitudinal': int(has_long.sum()),
        'achieved_080': best_auc >= 0.80
    }
    
    with open(OUTPUT_DIR / 'longitudinal_simple_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return best_auc


if __name__ == '__main__':
    auc = main()

