#!/usr/bin/env python3
"""
Multi-Target Support for Alzheimer's Research
Support for predicting multiple outcomes simultaneously (CDR, MMSE, diagnosis, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import logging

class MultiTargetAlzheimerPredictor:
    """
    Multi-target prediction system for Alzheimer's research
    Can predict multiple outcomes simultaneously (CDR, MMSE, diagnosis status, etc.)
    """
    
    def __init__(self, target_config: Dict[str, str] = None):
        self.target_config = target_config or {}
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
        
        # Common Alzheimer's targets and their types
        self.common_targets = {
            'CDR': 'multiclass',          # Clinical Dementia Rating
            'MMSE': 'regression',         # Mini-Mental State Exam score
            'diagnosis': 'binary',        # Normal vs Impaired
            'cognitive_status': 'multiclass',  # Normal/MCI/Dementia
            'progression_risk': 'binary', # High vs Low risk
            'brain_atrophy': 'regression' # Continuous measure
        }
    
    def detect_targets(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect potential target variables in the dataset
        """
        detected_targets = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # CDR detection
            if 'cdr' in col_lower and col not in ['cdr_binary', 'cdr_severity']:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 5 and all(isinstance(v, (int, float)) for v in unique_vals):
                    detected_targets[col] = 'multiclass'
            
            # MMSE detection
            elif 'mmse' in col_lower and 'category' not in col_lower:
                if df[col].dtype in ['int64', 'float64']:
                    detected_targets[col] = 'regression'
            
            # Diagnosis detection
            elif any(term in col_lower for term in ['diagnosis', 'group', 'status']):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    detected_targets[col] = 'binary'
                elif len(unique_vals) <= 5:
                    detected_targets[col] = 'multiclass'
            
            # Cognitive measures
            elif any(term in col_lower for term in ['adas', 'moca', 'cognitive']):
                if df[col].dtype in ['int64', 'float64']:
                    detected_targets[col] = 'regression'
        
        self.logger.info(f"🎯 Detected {len(detected_targets)} potential targets: {list(detected_targets.keys())}")
        return detected_targets
    
    def prepare_targets(self, df: pd.DataFrame, target_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare target variables for multi-target learning
        """
        target_info = {}
        prepared_targets = pd.DataFrame()
        
        for target in target_columns:
            if target not in df.columns:
                self.logger.warning(f"Target {target} not found in dataset")
                continue
            
            # Get target type
            target_type = self.target_config.get(target, self.common_targets.get(target, 'auto'))
            
            if target_type == 'auto':
                target_type = self._infer_target_type(df[target])
            
            # Prepare target based on type
            if target_type == 'binary':
                prepared_target = self._prepare_binary_target(df[target])
            elif target_type == 'multiclass':
                prepared_target = self._prepare_multiclass_target(df[target])
            elif target_type == 'regression':
                prepared_target = self._prepare_regression_target(df[target])
            else:
                prepared_target = df[target].copy()
            
            prepared_targets[target] = prepared_target
            target_info[target] = {
                'type': target_type,
                'n_classes': len(prepared_target.dropna().unique()) if target_type != 'regression' else None,
                'missing_rate': prepared_target.isna().mean(),
                'distribution': prepared_target.value_counts().to_dict() if target_type != 'regression' else None
            }
            
            self.logger.info(f"   {target} ({target_type}): {target_info[target]['n_classes'] or 'continuous'} outcomes")
        
        return prepared_targets, target_info
    
    def _infer_target_type(self, target_series: pd.Series) -> str:
        """
        Automatically infer target type from data
        """
        unique_vals = target_series.dropna().unique()
        n_unique = len(unique_vals)
        
        if target_series.dtype in ['object', 'category']:
            return 'multiclass' if n_unique > 2 else 'binary'
        elif n_unique == 2:
            return 'binary'
        elif n_unique <= 5 and all(isinstance(v, (int, float)) and v == int(v) for v in unique_vals):
            return 'multiclass'
        else:
            return 'regression'
    
    def _prepare_binary_target(self, target_series: pd.Series) -> pd.Series:
        """Prepare binary target"""
        unique_vals = target_series.dropna().unique()
        if len(unique_vals) == 2:
            # Map to 0, 1
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            return target_series.map(mapping)
        return target_series
    
    def _prepare_multiclass_target(self, target_series: pd.Series) -> pd.Series:
        """Prepare multiclass target"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        non_null_mask = target_series.notna()
        result = target_series.copy()
        result[non_null_mask] = le.fit_transform(target_series[non_null_mask])
        return result
    
    def _prepare_regression_target(self, target_series: pd.Series) -> pd.Series:
        """Prepare regression target"""
        return pd.to_numeric(target_series, errors='coerce')
    
    def fit_multi_target_models(self, X: pd.DataFrame, y: pd.DataFrame, 
                               target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit models for multi-target prediction
        """
        self.logger.info(f"🤖 Training multi-target models for {len(y.columns)} targets...")
        
        results = {
            'individual_models': {},
            'multi_output_models': {},
            'target_info': target_info,
            'performance_summary': {}
        }
        
        # Separate targets by type
        classification_targets = []
        regression_targets = []
        
        for target, info in target_info.items():
            if info['type'] in ['binary', 'multiclass']:
                classification_targets.append(target)
            else:
                regression_targets.append(target)
        
        # Train individual models for each target
        results['individual_models'] = self._fit_individual_models(X, y, target_info)
        
        # Train multi-output models
        if len(classification_targets) > 1:
            results['multi_output_models']['classification'] = self._fit_multi_output_classifier(
                X, y[classification_targets], target_info
            )
        
        if len(regression_targets) > 1:
            results['multi_output_models']['regression'] = self._fit_multi_output_regressor(
                X, y[regression_targets], target_info
            )
        
        # Mixed multi-output (if we have both classification and regression)
        if classification_targets and regression_targets:
            results['multi_output_models']['mixed'] = self._fit_mixed_multi_output(
                X, y, classification_targets, regression_targets, target_info
            )
        
        # Performance summary
        results['performance_summary'] = self._create_performance_summary(results)
        
        return results
    
    def _fit_individual_models(self, X: pd.DataFrame, y: pd.DataFrame, 
                              target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit individual models for each target
        """
        individual_results = {}
        
        for target in y.columns:
            if target not in target_info:
                continue
            
            target_type = target_info[target]['type']
            
            self.logger.info(f"   Training {target} ({target_type})...")
            
            # Remove rows with missing target
            valid_mask = y[target].notna()
            X_target = X[valid_mask]
            y_target = y[target][valid_mask]
            
            if len(y_target) < 10:
                self.logger.warning(f"   {target}: Insufficient data ({len(y_target)} samples)")
                continue
            
            # Select appropriate model
            if target_type in ['binary', 'multiclass']:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring = 'neg_mean_squared_error'
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_target, y_target, cv=5, scoring=scoring)
                
                # Fit final model
                model.fit(X_target, y_target)
                
                individual_results[target] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'target_type': target_type,
                    'n_samples': len(y_target)
                }
                
                score_name = 'accuracy' if target_type in ['binary', 'multiclass'] else 'neg_mse'
                self.logger.info(f"     {score_name}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                self.logger.error(f"   {target} failed: {e}")
        
        return individual_results
    
    def _fit_multi_output_classifier(self, X: pd.DataFrame, y: pd.DataFrame, 
                                   target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit multi-output classifier
        """
        # Remove rows with any missing classification targets
        valid_mask = y.notna().all(axis=1)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 10:
            self.logger.warning("Insufficient data for multi-output classification")
            return {}
        
        self.logger.info(f"   Training multi-output classifier ({len(y_valid)} samples, {len(y.columns)} targets)...")
        
        base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        multi_classifier = MultiOutputClassifier(base_classifier)
        
        try:
            # Fit model
            multi_classifier.fit(X_valid, y_valid)
            
            # Evaluate each target
            predictions = multi_classifier.predict(X_valid)
            target_scores = {}
            
            for i, target in enumerate(y.columns):
                score = accuracy_score(y_valid.iloc[:, i], predictions[:, i])
                target_scores[target] = score
                self.logger.info(f"     {target} accuracy: {score:.3f}")
            
            return {
                'model': multi_classifier,
                'target_scores': target_scores,
                'overall_score': np.mean(list(target_scores.values())),
                'n_samples': len(y_valid)
            }
            
        except Exception as e:
            self.logger.error(f"Multi-output classifier failed: {e}")
            return {}
    
    def _fit_multi_output_regressor(self, X: pd.DataFrame, y: pd.DataFrame, 
                                  target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit multi-output regressor
        """
        # Remove rows with any missing regression targets
        valid_mask = y.notna().all(axis=1)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 10:
            self.logger.warning("Insufficient data for multi-output regression")
            return {}
        
        self.logger.info(f"   Training multi-output regressor ({len(y_valid)} samples, {len(y.columns)} targets)...")
        
        base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        multi_regressor = MultiOutputRegressor(base_regressor)
        
        try:
            # Fit model
            multi_regressor.fit(X_valid, y_valid)
            
            # Evaluate each target
            predictions = multi_regressor.predict(X_valid)
            target_scores = {}
            
            for i, target in enumerate(y.columns):
                mse = mean_squared_error(y_valid.iloc[:, i], predictions[:, i])
                target_scores[target] = mse
                self.logger.info(f"     {target} MSE: {mse:.3f}")
            
            return {
                'model': multi_regressor,
                'target_scores': target_scores,
                'overall_score': np.mean(list(target_scores.values())),
                'n_samples': len(y_valid)
            }
            
        except Exception as e:
            self.logger.error(f"Multi-output regressor failed: {e}")
            return {}
    
    def _fit_mixed_multi_output(self, X: pd.DataFrame, y: pd.DataFrame,
                               classification_targets: List[str], regression_targets: List[str],
                               target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle mixed multi-output (both classification and regression)
        """
        results = {}
        
        # Train separate multi-output models for each type
        if classification_targets:
            y_class = y[classification_targets]
            results['classification'] = self._fit_multi_output_classifier(X, y_class, target_info)
        
        if regression_targets:
            y_reg = y[regression_targets]
            results['regression'] = self._fit_multi_output_regressor(X, y_reg, target_info)
        
        return results
    
    def _create_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create performance summary across all models
        """
        summary = {
            'best_individual_models': {},
            'multi_output_performance': {},
            'recommendations': []
        }
        
        # Summarize individual models
        if 'individual_models' in results:
            for target, model_info in results['individual_models'].items():
                summary['best_individual_models'][target] = {
                    'cv_score': model_info['cv_mean'],
                    'type': model_info['target_type'],
                    'n_samples': model_info['n_samples']
                }
        
        # Summarize multi-output models
        if 'multi_output_models' in results:
            for model_type, model_info in results['multi_output_models'].items():
                if model_info:
                    if isinstance(model_info, dict) and 'overall_score' in model_info:
                        summary['multi_output_performance'][model_type] = model_info['overall_score']
                    elif isinstance(model_info, dict):
                        # Handle nested structure (mixed models)
                        for sub_type, sub_info in model_info.items():
                            if 'overall_score' in sub_info:
                                summary['multi_output_performance'][f"{model_type}_{sub_type}"] = sub_info['overall_score']
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(results)
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on multi-target results
        """
        recommendations = []
        
        individual_results = results.get('individual_models', {})
        multi_output_results = results.get('multi_output_models', {})
        
        # Sample size recommendations
        small_sample_targets = [target for target, info in individual_results.items() 
                              if info['n_samples'] < 100]
        if small_sample_targets:
            recommendations.append(f"⚠️ Small sample sizes for: {', '.join(small_sample_targets)} - consider data augmentation")
        
        # Performance recommendations
        low_performance_targets = [target for target, info in individual_results.items() 
                                 if info['cv_mean'] < 0.7 and info['target_type'] in ['binary', 'multiclass']]
        if low_performance_targets:
            recommendations.append(f"📊 Low performance targets: {', '.join(low_performance_targets)} - consider feature engineering")
        
        # Multi-output vs individual comparison
        if multi_output_results and individual_results:
            recommendations.append("🔍 Compare multi-output vs individual models to determine best approach")
        
        # Alzheimer's-specific recommendations
        if any('CDR' in target or 'MMSE' in target for target in individual_results.keys()):
            recommendations.append("🧠 Alzheimer's targets detected - consider domain-specific feature engineering")
        
        return recommendations


def create_multi_target_pipeline(df: pd.DataFrame, target_columns: List[str] = None,
                               feature_columns: List[str] = None) -> Dict[str, Any]:
    """
    Create complete multi-target prediction pipeline
    """
    predictor = MultiTargetAlzheimerPredictor()
    
    # Auto-detect targets if not specified
    if target_columns is None:
        detected_targets = predictor.detect_targets(df)
        target_columns = list(detected_targets.keys())
    
    if not target_columns:
        return {'error': 'No suitable target variables found'}
    
    # Prepare features
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in target_columns]
    
    X = df[feature_columns]
    
    # Prepare targets
    y, target_info = predictor.prepare_targets(df, target_columns)
    
    if y.empty:
        return {'error': 'No valid target data after preparation'}
    
    # Fit models
    results = predictor.fit_multi_target_models(X, y, target_info)
    
    return results


if __name__ == "__main__":
    # Test multi-target system
    print("🧪 Testing Multi-Target Alzheimer's Prediction System...")
    
    # Create sample data with multiple Alzheimer's-related targets
    np.random.seed(42)
    n_samples = 300
    
    data = pd.DataFrame({
        # Features
        'Age': np.random.normal(70, 10, n_samples),
        'EDUC': np.random.normal(12, 4, n_samples),
        'eTIV': np.random.normal(1500, 200, n_samples),
        'nWBV': np.random.normal(0.75, 0.1, n_samples),
        
        # Multiple targets
        'CDR': np.random.choice([0, 0.5, 1], n_samples, p=[0.6, 0.3, 0.1]),
        'MMSE': np.random.normal(25, 5, n_samples),
        'diagnosis': np.random.choice(['Normal', 'Impaired'], n_samples, p=[0.7, 0.3]),
    })
    
    # Test pipeline
    results = create_multi_target_pipeline(data)
    
    print(f"\n✅ Multi-target test complete!")
    if 'performance_summary' in results:
        print(f"Individual models: {len(results['performance_summary']['best_individual_models'])}")
        print(f"Multi-output models: {len(results['performance_summary']['multi_output_performance'])}")
    else:
        print(f"Results: {list(results.keys())}")