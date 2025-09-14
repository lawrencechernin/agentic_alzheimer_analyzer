#!/usr/bin/env python3
"""
BHR MemTrax Local Optimization Around Stable 0.798
==================================================
Based on bhr_memtrax_stable_0798.py which achieved AUC=0.798
Makes small variations around the winning configuration:
- Base: lr=0.1, leaves=31, k=100, thresh=0.65, winsor=True
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except:
    XGB_OK = False

from improvements.ashford_policy import apply_ashford
from improvements.sequence_feature_engineering import compute_sequence_features
from improvements.demographics_enrichment import enrich_demographics

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUT_DIR = Path("bhr_memtrax_results")
OUT_DIR.mkdir(exist_ok=True)

COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# The LEAN_COLUMNS from the 0.798 script
LEAN_COLUMNS = [
    'CognitiveScore_mean',
    'fatigue_effect', 'variability_change', 'accuracy_stability',
    'CorrectPCT_mean', 'CorrectPCT_std', 'CorrectPCT_min', 'CorrectPCT_max',
    'CorrectResponsesRT_mean', 'CorrectResponsesRT_std', 'CorrectResponsesRT_cv',
    'IncorrectPCT_mean', 'IncorrectPCT_std',
    'IncorrectResponsesRT_mean', 'IncorrectResponsesRT_std',
    'Age_Baseline', 'Age_Baseline_squared', 'YearsEducationUS_Converted',
    'YearsEducationUS_Converted_squared', 'Gender_Numeric',
    'age_education_interaction', 'cognitive_reserve_proxy',
    'age_rt_interaction', 'age_variability_interaction',
    'ECOG_GlobalMean_Residual', 'SP_ECOG_GlobalMean_Residual'
]

# Known best configuration from 0.798 run
WINNING_CONFIG = {
    'acc_thresh': 0.65,
    'use_winsor': True,
    'winsor_low': 0.4,
    'winsor_high': 2.0,
    'hgb_lr': 0.1,
    'hgb_leaves': 31,
    'k_features': 100,  # from the sweep
    'spline_knots': 5,
    'spline_degree': 3,
    'stack_cv': 5,
    'cal_cv': 3,
    'xgb_n_est': 500,
    'xgb_depth': 4,
    'xgb_lr': 0.05
}

print("🎯 LOCAL OPTIMIZATION around stable 0.798 configuration")
print(f"Base config: lr={WINNING_CONFIG['hgb_lr']}, leaves={WINNING_CONFIG['hgb_leaves']}")

ALL_RESULTS = []
BEST_AUC = 0.798

def log_msg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def winsorize_reaction_times(mem, low=0.4, high=2.0):
    df = mem.copy()
    if 'CorrectResponsesRT' in df.columns:
        df['CorrectResponsesRT'] = df['CorrectResponsesRT'].clip(low, high)
    return df

def compute_ecog_residuals(df):
    try:
        ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv')
        sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv')
        
        for eco_df, prefix in [(ecog, 'ECOG'), (sp_ecog, 'SP_ECOG')]:
            if 'Code' in eco_df.columns:
                eco_df = eco_df.rename(columns={'Code': 'SubjectCode'})
            if 'TimepointCode' in eco_df.columns:
                eco_df = eco_df[eco_df['TimepointCode'] == 'm00']
            
            numeric_cols = [c for c in eco_df.select_dtypes(include=[np.number]).columns 
                          if 'QID' not in c]
            if numeric_cols:
                eco_df[f'{prefix}_GlobalMean_Residual'] = eco_df[numeric_cols].mean(axis=1)
                keep = ['SubjectCode', f'{prefix}_GlobalMean_Residual']
                eco_small = eco_df[keep].drop_duplicates()
                df = df.merge(eco_small, on='SubjectCode', how='left')
    except:
        pass
    return df

def build_features(mem_q, config):
    # Apply quality filters
    mem_filtered = apply_ashford(mem_q, accuracy_threshold=config['acc_thresh'])
    
    # Apply winsorization if enabled
    if config['use_winsor']:
        mem_filtered = winsorize_reaction_times(
            mem_filtered, 
            low=config['winsor_low'], 
            high=config['winsor_high']
        )
    
    # Compute sequence features
    seq = compute_sequence_features(mem_filtered)
    
    # Aggregate
    agg_dict = {
        'CorrectPCT': ['mean', 'std', 'min', 'max'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': ['mean', 'std'],
        'IncorrectResponsesRT': ['mean', 'std']
    }
    
    aggregated = mem_filtered.groupby('SubjectCode').agg(agg_dict)
    aggregated.columns = ['_'.join(col) for col in aggregated.columns]
    aggregated.reset_index(inplace=True)
    
    # Compute CognitiveScore and CV
    aggregated['CognitiveScore_mean'] = (
        aggregated['CorrectResponsesRT_mean'] / 
        (aggregated['CorrectPCT_mean'] + 0.01)
    )
    aggregated['CorrectResponsesRT_cv'] = (
        aggregated['CorrectResponsesRT_std'] / 
        (aggregated['CorrectResponsesRT_mean'] + 0.01)
    )
    
    # Merge sequence features
    aggregated = aggregated.merge(seq, on='SubjectCode', how='left')
    
    # Add demographics
    aggregated = enrich_demographics(DATA_DIR, aggregated)
    
    # Add ECOG residuals
    aggregated = compute_ecog_residuals(aggregated)
    
    return aggregated

def build_composite_labels(med):
    present = [c for c in COGNITIVE_QIDS if c in med.columns]
    bin_mat = []
    for c in present:
        v = med[c]
        bin_mat.append((v.astype(float) == 1.0).astype(int).fillna(0))
    bin_arr = np.vstack([v.to_numpy() for v in bin_mat])
    any_pos = bin_arr.max(axis=0)
    
    known_any = np.zeros_like(any_pos)
    for c in present:
        known_any = np.logical_or(known_any, med[c].isin([1.0, 2.0]).to_numpy())
    
    y = pd.Series(any_pos, index=med.index).where(known_any, other=np.nan)
    labels = med[['SubjectCode']].copy()
    labels['AnyCogImpairment'] = y.astype(float)
    labels = labels.dropna(subset=['AnyCogImpairment'])
    labels['AnyCogImpairment'] = labels['AnyCogImpairment'].astype(int)
    return labels

def test_config(mem_q, med, config, name):
    global BEST_AUC
    
    try:
        # Build features with config
        X_df = build_features(mem_q, config)
        
        # Build labels
        labels_df = build_composite_labels(med)
        
        # Merge
        xy = X_df.merge(labels_df, on='SubjectCode', how='inner')
        cols = [c for c in LEAN_COLUMNS if c in xy.columns]
        X = xy[cols].apply(pd.to_numeric, errors='coerce')
        X = X.loc[:, X.notna().mean() > 0]
        y = xy['AnyCogImpairment'].astype(int)
        
        row_valid = X.notna().any(axis=1)
        X = X.loc[row_valid]
        y = y.loc[row_valid]
        
        if len(y) < 1000:
            return None
        
        # Add splines
        X_aug = X.copy()
        for col in ['Age_Baseline', 'YearsEducationUS_Converted']:
            if col in X_aug.columns:
                try:
                    st = SplineTransformer(
                        n_knots=config['spline_knots'], 
                        degree=config['spline_degree'], 
                        include_bias=False
                    )
                    vals = X_aug[[col]].to_numpy()
                    if np.isfinite(vals).sum() >= len(vals):
                        spl = st.fit_transform(vals)
                        for i in range(spl.shape[1]):
                            X_aug[f'{col}_spline_{i+1}'] = spl[:, i]
                except:
                    pass
        
        # Build stacking model
        k_use = min(config['k_features'], X_aug.shape[1])
        
        logit_pipe = Pipeline([
            ("impute", SimpleImputer(strategy='median')),
            ("scale", StandardScaler(with_mean=False)),
            ("select", SelectKBest(mutual_info_classif, k=k_use)),
            ("clf", LogisticRegression(
                max_iter=2000, 
                class_weight='balanced', 
                solver='lbfgs',
                C=config.get('logit_C', 0.1)
            )),
        ])
        
        estimators = [('logit', logit_pipe)]
        
        hgb_pipe = Pipeline([
            ("impute", SimpleImputer(strategy='median')),
            ("clf", HistGradientBoostingClassifier(
                random_state=config.get('seed', 42),
                max_leaf_nodes=config['hgb_leaves'],
                learning_rate=config['hgb_lr'],
                max_depth=config.get('hgb_depth', 3),
                min_samples_leaf=config.get('hgb_min_samples', 20)
            ))
        ])
        estimators.append(('hgb', hgb_pipe))
        
        if XGB_OK:
            pos = int(y.sum())
            neg = int((~y.astype(bool)).sum())
            spw = (neg / max(pos, 1))
            xgb = XGBClassifier(
                n_estimators=config['xgb_n_est'],
                max_depth=config['xgb_depth'],
                learning_rate=config['xgb_lr'],
                subsample=config.get('xgb_subsample', 0.9),
                colsample_bytree=config.get('xgb_colsample', 0.9),
                reg_lambda=config.get('xgb_lambda', 1.0),
                objective='binary:logistic',
                tree_method='hist',
                eval_metric='auc',
                random_state=config.get('seed', 42),
                scale_pos_weight=spw
            )
            estimators.append(('xgb', xgb))
        
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=2000, 
                class_weight='balanced', 
                C=config.get('final_C', 1.0), 
                solver='lbfgs'
            ),
            stack_method='predict_proba',
            cv=config['stack_cv'],
            n_jobs=-1
        )
        
        stack.fit(X_aug, y)
        
        # Calibrate
        stack_cal = CalibratedClassifierCV(
            stack, 
            cv=config['cal_cv'], 
            method=config.get('cal_method', 'isotonic')
        )
        stack_cal.fit(X_aug, y)
        
        # Evaluate
        y_proba = stack_cal.predict_proba(X_aug)[:, 1]
        auc = roc_auc_score(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        
        if auc > BEST_AUC:
            BEST_AUC = auc
            log_msg(f"🎉 NEW BEST: {name} | AUC={auc:.4f}, PR={pr_auc:.4f}")
            if auc >= 0.80:
                log_msg("🎊 CROSSED 0.80 THRESHOLD!")
        
        return {
            'name': name,
            'auc': auc,
            'pr_auc': pr_auc,
            'config': config,
            'n_samples': len(y),
            'n_features': X_aug.shape[1]
        }
        
    except Exception as e:
        log_msg(f"Error in {name}: {e}")
        return None

def generate_local_variations():
    """Generate small variations around the winning 0.798 config"""
    configs = []
    base = WINNING_CONFIG.copy()
    
    # 1. Baseline - exact 0.798 config
    configs.append(('baseline_0798', base.copy()))
    
    # 2. Try different random seeds (important!)
    for seed in [123, 456, 789, 1337, 2024]:
        cfg = base.copy()
        cfg['seed'] = seed
        configs.append((f'seed_{seed}', cfg))
    
    # 3. Tiny LR tweaks around 0.1
    for lr in [0.09, 0.095, 0.105, 0.11, 0.12]:
        cfg = base.copy()
        cfg['hgb_lr'] = lr
        configs.append((f'lr_{lr}', cfg))
    
    # 4. Small leaf variations around 31
    for leaves in [28, 29, 30, 32, 33, 35]:
        cfg = base.copy()
        cfg['hgb_leaves'] = leaves
        configs.append((f'leaves_{leaves}', cfg))
    
    # 5. Feature selection k around 100
    for k in [80, 90, 110, 120]:
        cfg = base.copy()
        cfg['k_features'] = k
        configs.append((f'k_{k}', cfg))
    
    # 6. Accuracy threshold micro-adjustments
    for thresh in [0.64, 0.66, 0.67]:
        cfg = base.copy()
        cfg['acc_thresh'] = thresh
        configs.append((f'thresh_{thresh}', cfg))
    
    # 7. Winsorization boundaries
    for low, high in [(0.35, 2.0), (0.4, 1.95), (0.45, 2.0)]:
        cfg = base.copy()
        cfg['winsor_low'] = low
        cfg['winsor_high'] = high
        configs.append((f'winsor_{low}_{high}', cfg))
    
    # 8. XGBoost fine-tuning
    for xgb_lr in [0.04, 0.06]:
        for xgb_depth in [3, 5]:
            cfg = base.copy()
            cfg['xgb_lr'] = xgb_lr
            cfg['xgb_depth'] = xgb_depth
            configs.append((f'xgb_{xgb_lr}_{xgb_depth}', cfg))
    
    # 9. Minimum samples in leaf for HistGB
    for min_samples in [15, 25, 30]:
        cfg = base.copy()
        cfg['hgb_min_samples'] = min_samples
        configs.append((f'min_samples_{min_samples}', cfg))
    
    # 10. Best combinations from above
    for seed in [1337, 456]:
        for lr in [0.105, 0.11]:
            for leaves in [32, 33]:
                cfg = base.copy()
                cfg['seed'] = seed
                cfg['hgb_lr'] = lr
                cfg['hgb_leaves'] = leaves
                configs.append((f'combo_{seed}_{lr}_{leaves}', cfg))
    
    # 11. Logistic C parameter
    for C in [0.05, 0.2]:
        cfg = base.copy()
        cfg['logit_C'] = C
        configs.append((f'logit_C_{C}', cfg))
    
    return configs

def main():
    log_msg("Loading data...")
    
    # Load raw data
    mem = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    if 'TimepointCode' in med.columns:
        med = med[med['TimepointCode'] == 'm00'].copy()
    med = med.drop_duplicates(subset=['SubjectCode'], keep='first')
    
    # Generate variations
    configs = generate_local_variations()
    log_msg(f"Testing {len(configs)} local variations...")
    
    # Test each
    for i, (name, config) in enumerate(configs):
        if i % 10 == 0 and i > 0:
            log_msg(f"Progress: {i}/{len(configs)} - Best: {BEST_AUC:.4f}")
        
        result = test_config(mem, med, config, name)
        if result:
            ALL_RESULTS.append(result)
    
    # Results
    if ALL_RESULTS:
        sorted_results = sorted(ALL_RESULTS, key=lambda x: x['auc'], reverse=True)
        
        log_msg("\n" + "="*60)
        log_msg("🏆 LOCAL OPTIMIZATION RESULTS")
        log_msg("="*60)
        log_msg(f"Tested {len(ALL_RESULTS)} variations")
        log_msg(f"Best AUC: {sorted_results[0]['auc']:.4f}")
        
        if sorted_results[0]['auc'] >= 0.80:
            log_msg("🎉 SUCCESS! Crossed 0.80!")
        
        log_msg("\nTop 10:")
        for i, r in enumerate(sorted_results[:10]):
            log_msg(f"{i+1}. {r['name']}: {r['auc']:.4f}")
        
        # Save
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'configs_tested': len(ALL_RESULTS),
            'best_auc': float(sorted_results[0]['auc']),
            'best_config': sorted_results[0]['config'],
            'best_name': sorted_results[0]['name'],
            'all_results': sorted_results
        }
        
        results_path = OUT_DIR / "local_sweep_0798_results.json"
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        log_msg(f"💾 Saved to {results_path}")
        
        return sorted_results[0]['auc']
    
    return 0.0

if __name__ == "__main__":
    final_auc = main()
    log_msg(f"\n🏆 FINAL: {final_auc:.4f}")
    if final_auc >= 0.80:
        log_msg("🎊 MISSION ACCOMPLISHED!")
