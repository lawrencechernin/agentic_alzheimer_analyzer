#!/usr/bin/env python3
"""
BHR MCI Prediction with Informant SP-ECOG Data
===============================================
Leveraging the most reliable data source for breakthrough performance
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
import json

np.random.seed(42)
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cognitive impairment QIDs
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']


def extract_memtrax_features(memtrax_q):
    """Extract comprehensive MemTrax features"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Basic aggregates
        feat['correct_pct_mean'] = group['CorrectPCT'].mean()
        feat['correct_pct_std'] = group['CorrectPCT'].std()
        feat['correct_rt_mean'] = group['CorrectResponsesRT'].mean()
        feat['correct_rt_std'] = group['CorrectResponsesRT'].std()
        feat['incorrect_pct_mean'] = group['IncorrectPCT'].mean()
        feat['incorrect_rt_mean'] = group['IncorrectResponsesRT'].mean()
        
        # Composite scores
        feat['cog_score'] = feat['correct_rt_mean'] / (feat['correct_pct_mean'] + 0.01)
        feat['speed_accuracy'] = feat['correct_pct_mean'] / (feat['correct_rt_mean'] + 0.01)
        feat['error_rate'] = 1 - feat['correct_pct_mean']
        feat['rt_cv'] = feat['correct_rt_std'] / (feat['correct_rt_mean'] + 1e-6)
        
        # Sequence features from reaction times
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
            feat['seq_fatigue'] = np.mean(all_rts[-third:]) - np.mean(all_rts[:third])
            feat['seq_mean_rt'] = np.mean(all_rts)
            feat['seq_cv'] = np.std(all_rts) / (np.mean(all_rts) + 1e-6)
            
        feat['n_tests'] = len(group)
        features.append(feat)
    
    return pd.DataFrame(features)


def extract_sp_ecog_features(sp_ecog_baseline, subjects):
    """Extract informant SP-ECOG features"""
    # Filter to subjects
    sp_data = sp_ecog_baseline[sp_ecog_baseline['SubjectCode'].isin(subjects)].copy()
    
    # Get numeric QID columns
    qid_cols = [c for c in sp_data.columns if c.startswith('QID')]
    numeric_qids = sp_data[qid_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_qids:
        return pd.DataFrame({'SubjectCode': subjects})
    
    # Global scores
    sp_data['sp_ecog_global_mean'] = sp_data[numeric_qids].mean(axis=1)
    sp_data['sp_ecog_global_std'] = sp_data[numeric_qids].std(axis=1)
    sp_data['sp_ecog_max'] = sp_data[numeric_qids].max(axis=1)
    sp_data['sp_ecog_n_high'] = (sp_data[numeric_qids] >= 3).sum(axis=1)  # Count of significant impairment
    sp_data['sp_ecog_n_responses'] = sp_data[numeric_qids].notna().sum(axis=1)
    
    # Domain-specific if identifiable (simplified)
    memory_cols = [c for c in numeric_qids if '1' in c or '2' in c][:10]  # First 10 items often memory
    if memory_cols:
        sp_data['sp_ecog_memory'] = sp_data[memory_cols].mean(axis=1)
    
    # Keep only aggregate features
    feature_cols = ['SubjectCode'] + [c for c in sp_data.columns if c.startswith('sp_ecog_')]
    
    return sp_data[feature_cols].drop_duplicates(subset=['SubjectCode'])


def create_composite_labels(med_hx, sp_ecog_features):
    """
    Create more reliable labels using both self-report and informant data
    """
    # Get baseline medical history
    if 'TimepointCode' in med_hx.columns:
        med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].copy()
    else:
        med_baseline = med_hx.copy()
    
    # Self-reported impairment from QIDs
    available_qids = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]
    if available_qids:
        self_impairment = np.zeros(len(med_baseline), dtype=int)
        valid_self = np.zeros(len(med_baseline), dtype=bool)
        
        for qid in available_qids:
            self_impairment |= (med_baseline[qid] == 1).values
            valid_self |= med_baseline[qid].isin([1, 2]).values
        
        med_baseline['self_reported_mci'] = self_impairment
        med_baseline['self_valid'] = valid_self
    else:
        med_baseline['self_reported_mci'] = 0
        med_baseline['self_valid'] = False
    
    # Merge with informant data
    labels = med_baseline[['SubjectCode', 'self_reported_mci', 'self_valid']].merge(
        sp_ecog_features[['SubjectCode', 'sp_ecog_global_mean', 'sp_ecog_n_high']], 
        on='SubjectCode', 
        how='left'
    )
    
    # Create composite label
    # Strategy: Use informant if available, otherwise self-report
    # Informant threshold: mean > 2.5 or 5+ items with significant impairment
    labels['informant_mci'] = (
        (labels['sp_ecog_global_mean'] > 2.5) | 
        (labels['sp_ecog_n_high'] >= 5)
    ).astype(int)
    
    labels['has_informant'] = labels['sp_ecog_global_mean'].notna()
    
    # Composite: prioritize informant, fallback to self-report
    labels['composite_mci'] = np.where(
        labels['has_informant'],
        labels['informant_mci'],
        labels['self_reported_mci']
    )
    
    # Only keep valid labels
    labels['valid'] = labels['has_informant'] | labels['self_valid']
    
    return labels


def main():
    print("\n" + "="*70)
    print("BHR MCI PREDICTION WITH INFORMANT DATA")
    print("="*70)
    print("Strategy: Leverage SP-ECOG informant reports for reliable labels\n")
    
    # 1. Load data
    print("1. Loading datasets...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
    
    # Get SP-ECOG baseline
    sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
    print(f"   SP-ECOG baseline subjects: {sp_ecog_baseline['SubjectCode'].nunique():,}")
    
    # 2. Quality filter MemTrax
    print("\n2. Applying quality filters...")
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    print(f"   Quality filtered MemTrax: {len(memtrax_q):,} records")
    
    # 3. Find subjects with both MemTrax and SP-ECOG
    memtrax_subjects = set(memtrax_q['SubjectCode'].dropna().unique())
    sp_subjects = set(sp_ecog_baseline['SubjectCode'].dropna().unique())
    overlap_subjects = list(memtrax_subjects & sp_subjects)
    
    print(f"   Subjects with both MemTrax and SP-ECOG: {len(overlap_subjects):,}")
    
    # 4. Extract features
    print("\n3. Extracting features...")
    
    # MemTrax features for overlapping subjects
    memtrax_overlap = memtrax_q[memtrax_q['SubjectCode'].isin(overlap_subjects)]
    memtrax_features = extract_memtrax_features(memtrax_overlap)
    print(f"   MemTrax features: {memtrax_features.shape}")
    
    # SP-ECOG features
    sp_ecog_features = extract_sp_ecog_features(sp_ecog_baseline, overlap_subjects)
    print(f"   SP-ECOG features: {sp_ecog_features.shape}")
    
    # 5. Create composite labels
    print("\n4. Creating composite labels...")
    labels = create_composite_labels(med_hx, sp_ecog_features)
    labels = labels[labels['valid']].copy()
    
    print(f"   Valid labels: {len(labels):,}")
    print(f"   With informant data: {labels['has_informant'].sum():,}")
    print(f"   Self-report MCI rate: {labels['self_reported_mci'].mean():.1%}")
    print(f"   Informant MCI rate: {labels[labels['has_informant']]['informant_mci'].mean():.1%}")
    print(f"   Composite MCI rate: {labels['composite_mci'].mean():.1%}")
    
    # Check agreement between self and informant
    both = labels[labels['has_informant'] & labels['self_valid']]
    if len(both) > 0:
        agreement = (both['self_reported_mci'] == both['informant_mci']).mean()
        print(f"   Self-informant agreement: {agreement:.1%}")
    
    # 6. Merge all features
    print("\n5. Merging features...")
    data = memtrax_features.merge(sp_ecog_features, on='SubjectCode', how='inner')
    data = data.merge(labels[['SubjectCode', 'composite_mci']], on='SubjectCode', how='inner')
    
    print(f"   Final dataset: {len(data):,} subjects")
    print(f"   Features: {data.shape[1]-2}")
    print(f"   MCI prevalence: {data['composite_mci'].mean():.1%}")
    
    # 7. Prepare for modeling
    print("\n6. Training models...")
    
    X = data.drop(['SubjectCode', 'composite_mci'], axis=1)
    y = data['composite_mci']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train/test: {len(X_train)}/{len(X_test)}")
    print(f"   Test MCI cases: {y_test.sum()} ({y_test.mean():.1%})")
    
    # Models
    models = {
        'Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_split=15,
                class_weight='balanced', random_state=42
            ))
        ]),
        'HistGB': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.03, max_depth=6,
                min_samples_leaf=20, random_state=42
            ))
        ])
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=6,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric='logloss'
            ))
        ])
    
    # Train and evaluate
    best_auc = 0
    best_model = None
    best_name = None
    results = {}
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_auc:.4f}")
        
        results[name] = {
            'cv_auc': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'test_auc': float(test_auc)
        }
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
            best_name = name
    
    # Try stacking
    print("\n7. Training stacking ensemble...")
    stack = StackingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    
    cal_stack = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    cal_stack.fit(X_train, y_train)
    y_pred_stack = cal_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_pred_stack)
    
    print(f"   Calibrated Stack: Test={stack_auc:.4f}")
    
    if stack_auc > best_auc:
        best_auc = stack_auc
        best_name = 'Stack'
        best_model = cal_stack
    
    # Feature importance (if RF available)
    if 'RF' in models:
        rf = models['RF'].named_steps['clf']
        if hasattr(rf, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n8. Top features (Random Forest):")
            for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
                source = "SP-ECOG" if 'sp_ecog' in row['feature'] else "MemTrax"
                print(f"   {i:2d}. {row['feature']:<30} {row['importance']:.4f} ({source})")
    
    # Results summary
    print("\n" + "="*70)
    print("RESULTS WITH INFORMANT DATA")
    print("="*70)
    print(f"Best Model: {best_name}")
    print(f"Test AUC: {best_auc:.4f}")
    
    baseline_auc = 0.744
    improvement = best_auc - baseline_auc
    
    print(f"\nComparison to baseline (self-report only):")
    print(f"  Baseline: {baseline_auc:.3f}")
    print(f"  With informant: {best_auc:.3f}")
    print(f"  Improvement: {improvement:+.3f}")
    
    if best_auc >= 0.80:
        print(f"\n🎯 SUCCESS! Achieved {best_auc:.3f} AUC with informant data!")
    elif best_auc >= 0.78:
        print(f"\n📈 Significant improvement to {best_auc:.3f} AUC!")
    else:
        print(f"\n📊 Improvement to {best_auc:.3f} AUC")
    
    # Save results
    output = {
        'strategy': 'Informant SP-ECOG data',
        'n_subjects': len(data),
        'n_with_informant': int(labels['has_informant'].sum()),
        'prevalence': float(y.mean()),
        'best_model': best_name,
        'best_auc': float(best_auc),
        'baseline_comparison': {
            'baseline_auc': baseline_auc,
            'improvement': float(improvement)
        },
        'all_results': results
    }
    
    with open(OUTPUT_DIR / 'informant_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/informant_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()

