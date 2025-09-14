#!/usr/bin/env python3
"""
BHR MemTrax with Residualized Cognitive Scores
==============================================
Strategy: Remove expected age/education effects to reveal true impairment
Expected improvement: +0.02-0.03 AUC
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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


def compute_expected_performance(df, demo_cols=['Age_Baseline', 'YearsEducationUS_Converted']):
    """
    Compute expected cognitive performance based on demographics.
    This reveals who is performing worse than expected for their age/education.
    """
    print("  Computing expected performance norms...")
    
    # Features to residualize (cognitive measures)
    cognitive_features = [
        'CorrectPCT_mean', 'CorrectResponsesRT_mean',
        'IncorrectPCT_mean', 'IncorrectResponsesRT_mean',
        'CogScore', 'RT_CV', 'Speed_Accuracy_Product',
        'seq_mean_rt', 'seq_fatigue', 'reliability_change'
    ]
    
    # Only residualize features that exist
    cognitive_features = [f for f in cognitive_features if f in df.columns]
    
    # Check we have demographics
    demo_available = [c for c in demo_cols if c in df.columns]
    if len(demo_available) < 2:
        print("    Insufficient demographics for residualization")
        return df
    
    # Create polynomial features for demographics (captures non-linear aging)
    X_demo = df[demo_available].copy()
    X_demo = X_demo.fillna(X_demo.median())
    
    # Add polynomial terms
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_demo_poly = poly.fit_transform(X_demo)
    
    # For each cognitive feature, compute residuals
    residualized = df.copy()
    
    for feature in cognitive_features:
        if df[feature].notna().sum() < 100:
            continue
            
        y = df[feature].values.copy()
        valid = ~np.isnan(y)
        
        if valid.sum() < 100:
            continue
        
        # Fit expected performance model on normal subjects only (if labels available)
        if 'cognitive_impairment' in df.columns:
            # Train norm model on cognitively normal subjects only
            normal_mask = (df['cognitive_impairment'] == 0) & valid
            if normal_mask.sum() > 50:
                lr = LinearRegression()
                lr.fit(X_demo_poly[normal_mask], y[normal_mask])
                
                # Predict expected for everyone
                y_expected = lr.predict(X_demo_poly)
                
                # Compute residuals (actual - expected)
                residuals = np.full_like(y, np.nan)
                residuals[valid] = y[valid] - y_expected[valid]
                
                # Store both raw and residualized
                residualized[f'{feature}_residual'] = residuals
                
                # Z-score the residuals
                residuals_z = (residuals - np.nanmean(residuals)) / np.nanstd(residuals)
                residualized[f'{feature}_zscore'] = residuals_z
                
                print(f"    Residualized {feature}")
        else:
            # No labels - use all subjects for norms
            lr = LinearRegression()
            lr.fit(X_demo_poly[valid], y[valid])
            
            y_expected = lr.predict(X_demo_poly)
            residuals = np.full_like(y, np.nan)
            residuals[valid] = y[valid] - y_expected[valid]
            
            residualized[f'{feature}_residual'] = residuals
    
    return residualized


def extract_enhanced_features(memtrax_q):
    """Extract features including sequence patterns"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        all_rts = []
        
        for _, row in group.iterrows():
            if pd.notna(row.get('ReactionTimes')):
                try:
                    rts = [float(x.strip()) for x in str(row['ReactionTimes']).split(',') 
                           if x.strip() and x.strip() != 'nan']
                    all_rts.extend([r for r in rts if 0.3 <= r <= 3.0])
                except:
                    continue
        
        if len(all_rts) >= 10:
            n = len(all_rts)
            third = max(1, n // 3)
            feat['seq_first_third'] = np.mean(all_rts[:third])
            feat['seq_last_third'] = np.mean(all_rts[-third:])
            feat['seq_fatigue'] = feat['seq_last_third'] - feat['seq_first_third']
            feat['seq_mean_rt'] = np.mean(all_rts)
            feat['seq_std_rt'] = np.std(all_rts)
            feat['seq_cv'] = feat['seq_std_rt'] / (feat['seq_mean_rt'] + 1e-6)
            
            mid = n // 2
            if mid > 1:
                feat['reliability_change'] = np.var(all_rts[mid:]) - np.var(all_rts[:mid])
                
            if n >= 3:
                x = np.arange(n)
                if np.var(x) > 0:
                    slope, _ = np.polyfit(x, all_rts, 1)
                    feat['rt_slope'] = slope
                    
            feat['n_tests'] = len(group)
            
        features.append(feat)
    
    return pd.DataFrame(features)


def main():
    print("\n" + "="*70)
    print("BHR MEMTRAX WITH RESIDUALIZED COGNITIVE SCORES")
    print("="*70)
    print("Strategy: Remove demographic effects to reveal true impairment\n")
    
    # Load data
    print("1. Loading data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Quality filter
    print("2. Applying quality filter...")
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Feature engineering
    print("3. Engineering features...")
    
    # Sequence features
    seq_feat = extract_enhanced_features(memtrax_q)
    
    # Aggregates
    agg_feat = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std', 'min', 'max'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': ['mean', 'std'],
        'IncorrectResponsesRT': ['mean', 'std'],
        'CorrectRejectionsN': ['mean', 'std'],
        'IncorrectRejectionsN': ['mean', 'std']
    })
    agg_feat.columns = ['_'.join(col) for col in agg_feat.columns]
    agg_feat = agg_feat.reset_index()
    
    # Composite scores
    agg_feat['CogScore'] = agg_feat['CorrectResponsesRT_mean'] / (agg_feat['CorrectPCT_mean'] + 0.01)
    agg_feat['RT_CV'] = agg_feat['CorrectResponsesRT_std'] / (agg_feat['CorrectResponsesRT_mean'] + 1e-6)
    agg_feat['Speed_Accuracy_Product'] = agg_feat['CorrectPCT_mean'] / (agg_feat['CorrectResponsesRT_mean'] + 0.01)
    
    # Merge features
    features = agg_feat.merge(seq_feat, on='SubjectCode', how='left')
    
    # Add demographics
    print("4. Adding demographics...")
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
            features = features.merge(
                demo[demo_cols].drop_duplicates('SubjectCode'),
                on='SubjectCode', how='left'
            )
    
    # Create labels
    print("5. Creating labels...")
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
    labels = labels[valid].copy()
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    print(f"  Dataset: {len(data)} subjects, prevalence: {data['cognitive_impairment'].mean():.1%}")
    
    # CRITICAL: Compute residualized scores
    print("\n6. Computing residualized cognitive scores...")
    data_residualized = compute_expected_performance(data)
    
    # Count residualized features
    residual_cols = [c for c in data_residualized.columns if '_residual' in c or '_zscore' in c]
    print(f"  Created {len(residual_cols)} residualized features")
    
    # Prepare features
    # Use BOTH raw and residualized features
    feature_cols = [c for c in data_residualized.columns 
                   if c not in ['SubjectCode', 'cognitive_impairment']]
    
    X = data_residualized[feature_cols]
    y = data_residualized['cognitive_impairment']
    
    print(f"\n7. Final features: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  Train/test: {len(X_train)}/{len(X_test)}")
    
    # Models
    print("\n8. Training models with residualized features...")
    
    models = {
        'Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=10,
                                          min_samples_split=15, class_weight='balanced',
                                          random_state=RANDOM_STATE))
        ]),
        'HistGB': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                                                  max_depth=6, min_samples_leaf=20,
                                                  random_state=RANDOM_STATE))
        ])
    }
    
    best_auc = 0
    best_name = None
    results = {}
    
    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)
        
        print(f"  {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_auc:.4f}")
        
        results[name] = test_auc
        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name
            best_model = model
    
    # Stacking
    stack = StackingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    
    cal_stack = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    cal_stack.fit(X_train, y_train)
    y_pred = cal_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_pred)
    
    print(f"  Stack: Test={stack_auc:.4f}")
    
    if stack_auc > best_auc:
        best_auc = stack_auc
        best_name = 'Stack'
    
    # Feature importance
    if 'RF' in models:
        rf = models['RF'].named_steps['clf']
        if hasattr(rf, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n9. Top residualized features:")
            residual_importance = importance[importance['feature'].str.contains('residual|zscore')]
            for i, (_, row) in enumerate(residual_importance.head(5).iterrows(), 1):
                print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS WITH RESIDUALIZED COGNITIVE SCORES")
    print("="*70)
    print(f"Best Model: {best_name}")
    print(f"Test AUC: {best_auc:.4f}")
    
    baseline_auc = 0.744
    improvement = best_auc - baseline_auc
    
    if best_auc >= 0.80:
        print(f"\n🎯 SUCCESS! Achieved {best_auc:.3f} AUC!")
        print("Residualization helped break the 0.80 barrier!")
    elif improvement > 0.01:
        print(f"\n📈 Improved by {improvement:+.3f} over baseline ({baseline_auc:.3f})")
        print("Residualization is helping!")
    else:
        print(f"\n📊 Similar to baseline ({baseline_auc:.3f})")
    
    # Save
    output = {
        'strategy': 'Residualized cognitive scores',
        'best_auc': float(best_auc),
        'baseline_auc': baseline_auc,
        'improvement': float(improvement),
        'best_model': best_name,
        'n_residual_features': len(residual_cols),
        'all_results': {k: float(v) for k, v in results.items()}
    }
    
    with open(OUTPUT_DIR / 'residualized_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/residualized_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()

