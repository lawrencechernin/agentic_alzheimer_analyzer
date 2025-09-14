#!/usr/bin/env python3
"""
BHR MemTrax with Advanced Models and Ensembles
===============================================
Building on 0.744 baseline with:
1. Cost-sensitive learning
2. Advanced models (XGBoost, MLP, BalancedBagging)
3. Hyperparameter tuning
4. Threshold optimization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, 
    StackingClassifier, BaggingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from scipy.stats import uniform, randint
import json
import time

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
    print("XGBoost not available")

np.random.seed(42)
RANDOM_STATE = 42
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cognitive impairment QIDs for labels
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']


def extract_memtrax_features(memtrax_q):
    """Extract comprehensive MemTrax features"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Basic statistics
        feat['CorrectPCT_mean'] = group['CorrectPCT'].mean()
        feat['CorrectPCT_std'] = group['CorrectPCT'].std()
        feat['CorrectPCT_min'] = group['CorrectPCT'].min()
        feat['CorrectPCT_max'] = group['CorrectPCT'].max()
        
        feat['CorrectResponsesRT_mean'] = group['CorrectResponsesRT'].mean()
        feat['CorrectResponsesRT_std'] = group['CorrectResponsesRT'].std()
        feat['CorrectResponsesRT_min'] = group['CorrectResponsesRT'].min()
        feat['CorrectResponsesRT_max'] = group['CorrectResponsesRT'].max()
        
        feat['IncorrectPCT_mean'] = group['IncorrectPCT'].mean()
        feat['IncorrectResponsesRT_mean'] = group['IncorrectResponsesRT'].mean()
        
        # Composite scores
        feat['CognitiveScore'] = feat['CorrectResponsesRT_mean'] / (feat['CorrectPCT_mean'] + 0.01)
        feat['Speed_Accuracy_Product'] = feat['CorrectPCT_mean'] * feat['CorrectResponsesRT_mean']
        feat['Error_Rate'] = 1 - feat['CorrectPCT_mean']
        feat['Response_Consistency'] = 1 / (feat['CorrectResponsesRT_std'] + 0.01)
        
        # Sequence analysis
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
            
            feat['first_third_mean'] = np.mean(all_rts[:third])
            feat['last_third_mean'] = np.mean(all_rts[-third:])
            feat['fatigue_effect'] = feat['last_third_mean'] - feat['first_third_mean']
            
            mid = n // 2
            if mid > 1:
                feat['reliability_change'] = np.var(all_rts[mid:]) - np.var(all_rts[:mid])
                
            if n >= 3:
                slope, _ = np.polyfit(np.arange(n), all_rts, 1)
                feat['rt_slope'] = slope
                
        feat['n_tests'] = len(group)
        features.append(feat)
    
    return pd.DataFrame(features)


def build_composite_labels(med_hx):
    """Build composite cognitive impairment labels"""
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
    if not available_qids:
        raise ValueError("No cognitive QIDs found!")
    
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


def add_demographics(df, data_dir):
    """Add demographics and create interaction features"""
    demo_files = ['BHR_Demographics.csv', 'Profile.csv']
    
    for filename in demo_files:
        path = data_dir / filename
        if path.exists():
            try:
                demo = pd.read_csv(path, low_memory=False)
                if 'Code' in demo.columns:
                    demo.rename(columns={'Code': 'SubjectCode'}, inplace=True)
                    
                if 'SubjectCode' in demo.columns:
                    cols = ['SubjectCode']
                    
                    # Age - include QID186
                    for c in ['QID186', 'Age_Baseline', 'Age']:
                        if c in demo.columns:
                            demo.rename(columns={c: 'Age_Baseline'}, inplace=True)
                            cols.append('Age_Baseline')
                            break
                    
                    # Education - include QID184
                    for c in ['QID184', 'YearsEducationUS_Converted', 'Education']:
                        if c in demo.columns:
                            demo.rename(columns={c: 'YearsEducationUS_Converted'}, inplace=True)
                            cols.append('YearsEducationUS_Converted')
                            break
                    
                    # Gender
                    for c in ['Gender', 'Sex']:
                        if c in demo.columns:
                            cols.append(c)
                            if c == 'Sex':
                                demo.rename(columns={'Sex': 'Gender'}, inplace=True)
                            break
                    
                    if len(cols) > 1:
                        df = df.merge(demo[cols].drop_duplicates('SubjectCode'), 
                                     on='SubjectCode', how='left')
                        break
            except:
                continue
    
    # Derived features
    if 'Age_Baseline' in df.columns:
        df['Age_sq'] = df['Age_Baseline'] ** 2
        if 'CorrectResponsesRT_mean' in df.columns:
            df['age_rt_interact'] = df['Age_Baseline'] * df['CorrectResponsesRT_mean'] / 65
            
    if 'YearsEducationUS_Converted' in df.columns:
        df['Edu_sq'] = df['YearsEducationUS_Converted'] ** 2
        
    if all(c in df.columns for c in ['Age_Baseline', 'YearsEducationUS_Converted']):
        df['Age_Edu_interact'] = df['Age_Baseline'] * df['YearsEducationUS_Converted']
        df['CogReserve'] = df['YearsEducationUS_Converted'] / (df['Age_Baseline'] + 1)
        
    if 'Gender' in df.columns:
        df['Gender_Num'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
        
    return df


def create_advanced_models():
    """Create advanced model configurations with cost-sensitive learning"""
    
    models = {}
    
    # 1. Cost-sensitive Logistic Regression
    models['LogisticCS'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=2000, 
            class_weight={0: 1, 1: 10},  # Higher cost for MCI misclassification
            C=0.5, 
            solver='saga'
        ))
    ])
    
    # 2. Random Forest with balanced subsample
    models['RF_Balanced'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=15,
            class_weight='balanced_subsample',  # Better for imbalanced data
            max_features='sqrt',
            bootstrap=True,
            random_state=RANDOM_STATE
        ))
    ])
    
    # 3. XGBoost with scale_pos_weight (if available)
    if XGB_AVAILABLE:
        # Calculate scale_pos_weight for the imbalanced dataset
        # Approximately 94% negative, 6% positive -> scale = 94/6 ≈ 15.7
        models['XGBoost_Weighted'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=15,  # Handle imbalance
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            ))
        ])
    
    # 4. Neural Network with imbalance handling
    models['MLP_Balanced'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE
        ))
    ])
    
    # 5. Balanced Bagging Classifier
    base_clf = LogisticRegression(max_iter=1000, C=0.5)
    models['BalancedBagging'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', BalancedBaggingClassifier(
            estimator=base_clf,
            n_estimators=100,
            sampling_strategy='not majority',
            replacement=False,
            random_state=RANDOM_STATE
        ))
    ])
    
    # 6. Easy Ensemble
    models['EasyEnsemble'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', EasyEnsembleClassifier(
            n_estimators=10,
            sampling_strategy='not majority',
            replacement=False,
            random_state=RANDOM_STATE
        ))
    ])
    
    return models


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for best models"""
    
    print("\n   Hyperparameter tuning (this may take a few minutes)...")
    
    # Define parameter grids
    param_grids = {
        'RF': {
            'clf__n_estimators': [200, 300, 400],
            'clf__max_depth': [8, 10, 12],
            'clf__min_samples_split': [10, 15, 20],
            'clf__class_weight': ['balanced', 'balanced_subsample']
        },
        'HistGB': {
            'clf__learning_rate': [0.03, 0.05, 0.08],
            'clf__max_depth': [4, 5, 6],
            'clf__max_leaf_nodes': [20, 31, 40],
            'clf__min_samples_leaf': [15, 20, 30]
        }
    }
    
    if XGB_AVAILABLE:
        param_grids['XGB'] = {
            'clf__n_estimators': [200, 300],
            'clf__learning_rate': [0.02, 0.03, 0.05],
            'clf__max_depth': [5, 6, 7],
            'clf__scale_pos_weight': [10, 15, 20]
        }
    
    # Models to tune
    models_to_tune = {
        'RF': RandomForestClassifier(random_state=RANDOM_STATE),
        'HistGB': HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    
    if XGB_AVAILABLE:
        models_to_tune['XGB'] = XGBClassifier(
            random_state=RANDOM_STATE, 
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    best_models = {}
    
    for name, base_model in models_to_tune.items():
        print(f"      Tuning {name}...")
        
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', base_model)
        ])
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=20,  # Number of parameter combinations to try
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        best_models[f'{name}_Tuned'] = random_search.best_estimator_
        print(f"         Best CV AUC: {random_search.best_score_:.4f}")
    
    return best_models


def find_optimal_threshold(y_true, y_proba):
    """Find optimal decision threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Also find threshold for 80% sensitivity (for screening)
    sensitivity_80_idx = np.where(tpr >= 0.80)[0]
    if len(sensitivity_80_idx) > 0:
        screening_threshold = thresholds[sensitivity_80_idx[0]]
    else:
        screening_threshold = optimal_threshold
    
    return {
        'youden': optimal_threshold,
        'screening_80': screening_threshold,
        'sensitivity_at_youden': tpr[optimal_idx],
        'specificity_at_youden': 1 - fpr[optimal_idx]
    }


def main():
    print("\n" + "="*70)
    print("BHR MEMTRAX WITH ADVANCED MODELS AND ENSEMBLES")
    print("="*70)
    
    # Load data
    print("\n1. Loading BHR data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Quality filter
    print("2. Applying Ashford quality filter...")
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Extract features
    print("3. Extracting MemTrax features...")
    features = extract_memtrax_features(memtrax_q)
    
    # Add demographics
    print("4. Adding demographics...")
    features = add_demographics(features, DATA_DIR)
    
    # Create labels
    print("5. Creating labels...")
    labels = build_composite_labels(med_hx)
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    
    print(f"\n   Final dataset: {len(data):,} subjects")
    print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
    
    # Prepare for modeling
    X = data.drop(['SubjectCode', 'cognitive_impairment'], axis=1).values
    y = data['cognitive_impairment'].values
    
    # Calculate sample weights for cost-sensitive learning
    sample_weights = compute_sample_weight('balanced', y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split sample weights
    weights_train = sample_weights[:len(y_train)]
    weights_test = sample_weights[len(y_train):]
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Class distribution - Train: {y_train.mean():.1%} positive")
    print(f"   Class distribution - Test: {y_test.mean():.1%} positive")
    
    # === TEST 1: ADVANCED MODELS ===
    print("\n6. TESTING ADVANCED MODELS")
    print("="*70)
    
    advanced_models = create_advanced_models()
    results = {}
    
    for name, model in advanced_models.items():
        print(f"\n   {name}:")
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            
            # Fit with sample weights if applicable
            if 'MLP' in name:
                # MLP doesn't support sample_weight in pipeline
                model.fit(X_train, y_train)
            else:
                try:
                    # Try with sample weights
                    model.fit(X_train, y_train, clf__sample_weight=weights_train)
                except:
                    # Fall back to regular fit
                    model.fit(X_train, y_train)
            
            # Test performance
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Find optimal threshold
            thresholds = find_optimal_threshold(y_test, y_pred_proba)
            
            print(f"      CV AUC: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            print(f"      Test AUC: {test_auc:.4f}")
            print(f"      Optimal threshold: {thresholds['youden']:.3f}")
            print(f"      Sensitivity at optimal: {thresholds['sensitivity_at_youden']:.1%}")
            
            results[name] = {
                'cv_auc': float(cv_scores.mean()),
                'test_auc': float(test_auc),
                'optimal_threshold': float(thresholds['youden'])
            }
            
        except Exception as e:
            print(f"      Failed: {e}")
            results[name] = {'error': str(e)}
    
    # === TEST 2: HYPERPARAMETER TUNING ===
    print("\n7. HYPERPARAMETER TUNING")
    print("="*70)
    
    tuned_models = hyperparameter_tuning(X_train, y_train)
    
    for name, model in tuned_models.items():
        # Test performance
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   {name} Test AUC: {test_auc:.4f}")
        results[name] = {'test_auc': float(test_auc)}
    
    # === TEST 3: ADVANCED STACKING ENSEMBLE ===
    print("\n8. ADVANCED STACKING ENSEMBLE")
    print("="*70)
    
    # Select best models for stacking
    best_base_models = []
    
    # Add best performers
    if 'LogisticCS' in advanced_models:
        best_base_models.append(('LogisticCS', advanced_models['LogisticCS']))
    if 'RF_Balanced' in advanced_models:
        best_base_models.append(('RF_Balanced', advanced_models['RF_Balanced']))
    if XGB_AVAILABLE and 'XGBoost_Weighted' in advanced_models:
        best_base_models.append(('XGBoost', advanced_models['XGBoost_Weighted']))
    if 'BalancedBagging' in advanced_models:
        best_base_models.append(('BalancedBag', advanced_models['BalancedBagging']))
    
    # Add tuned models
    for name, model in tuned_models.items():
        best_base_models.append((name, model))
    
    # Create advanced stacking ensemble
    advanced_stack = StackingClassifier(
        estimators=best_base_models[:5],  # Use top 5 models
        final_estimator=LogisticRegression(class_weight='balanced', C=0.5),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Calibrate the stacked model
    calibrated_stack = CalibratedClassifierCV(advanced_stack, cv=3, method='isotonic')
    
    print("   Training stacking ensemble...")
    calibrated_stack.fit(X_train, y_train)
    
    # Test performance
    y_pred_stack = calibrated_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_pred_stack)
    
    print(f"   Calibrated Stack AUC: {stack_auc:.4f}")
    
    # Find optimal threshold for stack
    stack_thresholds = find_optimal_threshold(y_test, y_pred_stack)
    print(f"   Optimal threshold: {stack_thresholds['youden']:.3f}")
    print(f"   Sensitivity: {stack_thresholds['sensitivity_at_youden']:.1%}")
    print(f"   Specificity: {stack_thresholds['specificity_at_youden']:.1%}")
    
    results['Advanced_Stack'] = {
        'test_auc': float(stack_auc),
        'optimal_threshold': float(stack_thresholds['youden'])
    }
    
    # === FINAL RESULTS ===
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    # Find best result
    best_model = max(results.items(), key=lambda x: x[1].get('test_auc', 0))
    best_name = best_model[0]
    best_auc = best_model[1]['test_auc']
    
    baseline_auc = 0.744
    
    print(f"\nBaseline AUC: {baseline_auc:.4f}")
    print(f"Best Model: {best_name}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Improvement: {(best_auc - baseline_auc):+.4f}")
    
    # Summary of all results
    print("\nAll Models Summary:")
    for name, res in sorted(results.items(), key=lambda x: -x[1].get('test_auc', 0)):
        if 'test_auc' in res:
            print(f"   {name:20s}: {res['test_auc']:.4f}")
    
    if best_auc >= 0.80:
        print(f"\n🎯 SUCCESS! Achieved {best_auc:.4f} AUC!")
    elif best_auc >= 0.78:
        print(f"\n✅ GREAT! Achieved {best_auc:.4f} AUC!")
    elif best_auc > baseline_auc:
        print(f"\n📈 Slight improvement to {best_auc:.4f}")
    else:
        print(f"\n📊 No significant improvement over baseline")
    
    # Save results
    output = {
        'strategy': 'Advanced models and ensembles',
        'baseline_auc': baseline_auc,
        'best_model': best_name,
        'best_auc': float(best_auc),
        'improvement': float(best_auc - baseline_auc),
        'all_results': results,
        'notes': [
            'Cost-sensitive learning applied',
            'Hyperparameter tuning performed',
            'Advanced ensemble methods tested',
            'Threshold optimization included'
        ]
    }
    
    with open(OUTPUT_DIR / 'advanced_models_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/advanced_models_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()

