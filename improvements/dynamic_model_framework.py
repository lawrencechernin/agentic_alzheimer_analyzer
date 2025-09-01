#!/usr/bin/env python3
"""
Dynamic Model Selection Framework for Alzheimer's Research
Automatically selects and optimizes models based on data characteristics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer, classification_report
import logging

try:
    from .clinical_evaluation_metrics import ClinicalEvaluator
    CLINICAL_EVAL_AVAILABLE = True
except ImportError:
    CLINICAL_EVAL_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Do not import lightgbm at module import time to avoid pulling dask with incompatible pandas
LIGHTGBM_AVAILABLE = False
try:
    import importlib
    _lgbm_spec = importlib.util.find_spec('lightgbm')
    LIGHTGBM_AVAILABLE = _lgbm_spec is not None
except Exception:
    LIGHTGBM_AVAILABLE = False


def _safe_create_lgbm(**kwargs):
    """Lazily import LightGBM and create classifier if possible, else return None."""
    try:
        from lightgbm import LGBMClassifier  # noqa: WPS433
        return LGBMClassifier(**kwargs)
    except Exception as e:
        logging.getLogger(__name__).warning(f"LightGBM unavailable or incompatible: {e}. Skipping LightGBM.")
        return None


class DynamicModelSelector:
    """
    Automatically selects and optimizes ML models based on data characteristics
    Specialized for Alzheimer's research but generalizable across datasets
    """
    
    def __init__(self, target_type: str = 'multiclass', sample_size: int = 100):
        self.target_type = target_type
        self.sample_size = sample_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize clinical evaluator if available
        self.clinical_evaluator = ClinicalEvaluator() if CLINICAL_EVAL_AVAILABLE else None
        
        # Model pools based on data characteristics
        self.model_pools = self._initialize_model_pools()
        
    def _initialize_model_pools(self) -> Dict[str, Dict]:
        """Initialize different model pools for different scenarios"""
        
        pools = {
            'small_dataset': {  # < 500 samples
                'models': [
                    ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000)),
                    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
                    ('SVM', SVC(probability=True, random_state=42))
                ],
                'reason': 'Small dataset - avoiding overfitting'
            },
            
            'medium_dataset': {  # 500-5000 samples
                'models': [
                    ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=42)),
                    ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                    ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000))
                ],
                'reason': 'Medium dataset - balanced complexity'
            },
            
            'large_dataset': {  # > 5000 samples
                'models': [
                    ('RandomForest', RandomForestClassifier(n_estimators=300, random_state=42)),
                    ('GradientBoosting', GradientBoostingClassifier(n_estimators=200, random_state=42)),
                ],
                'reason': 'Large dataset - can handle complex models'
            },
            
            'alzheimer_specialized': {  # For CDR, MMSE, cognitive targets
                'models': [
                    ('RandomForest', RandomForestClassifier(
                        n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
                    )),
                    ('GradientBoosting', GradientBoostingClassifier(
                        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
                    ))
                ],
                'reason': 'Optimized for cognitive assessment data'
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            for pool_name, pool_data in pools.items():
                if pool_name != 'small_dataset':  # XGBoost can overfit on small data
                    pool_data['models'].append(
                        ('XGBoost', XGBClassifier(
                            n_estimators=150, learning_rate=0.1, max_depth=6,
                            random_state=42, eval_metric='mlogloss'
                        ))
                    )
        
        # Add LightGBM if available and creatable
        if LIGHTGBM_AVAILABLE:
            for pool_name, pool_data in pools.items():
                if pool_name in ['medium_dataset', 'large_dataset', 'alzheimer_specialized']:
                    lgbm = _safe_create_lgbm(
                        n_estimators=150, learning_rate=0.1, max_depth=6,
                        random_state=42, verbose=-1
                    )
                    if lgbm is not None:
                        pool_data['models'].append(('LightGBM', lgbm))
        
        return pools
    
    def select_model_pool(self, X: pd.DataFrame, y: pd.Series, 
                         domain: str = 'general') -> Tuple[List, str]:
        """
        Automatically select appropriate model pool based on data characteristics
        """
        n_samples, n_features = X.shape
        n_classes = len(y.unique())
        
        self.logger.info(f"📊 Data characteristics: {n_samples} samples, {n_features} features, {n_classes} classes")
        
        # Alzheimer's-specific detection
        alzheimer_indicators = ['CDR', 'MMSE', 'ADAS', 'MoCA', 'brain', 'cognitive', 'dementia']
        has_alzheimer_features = any(indicator.lower() in ' '.join(X.columns).lower() 
                                   for indicator in alzheimer_indicators)
        
        # Select pool based on characteristics
        if has_alzheimer_features and domain in ['alzheimer', 'cognitive', 'neurological']:
            pool_name = 'alzheimer_specialized'
        elif n_samples < 500:
            pool_name = 'small_dataset'
        elif n_samples < 5000:
            pool_name = 'medium_dataset'
        else:
            pool_name = 'large_dataset'
        
        selected_pool = self.model_pools[pool_name]
        self.logger.info(f"🎯 Selected model pool: {pool_name}")
        self.logger.info(f"   Reason: {selected_pool['reason']}")
        
        return selected_pool['models'], pool_name
    
    def evaluate_model_comprehensive(self, model: Any, X: pd.DataFrame, 
                                   y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Comprehensive model evaluation including F1, precision, recall, and accuracy
        """
        # Define scoring metrics
        scoring = {
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
            'accuracy': 'accuracy'
        }
        
        # Perform cross-validation
        try:
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                      return_train_score=False, n_jobs=-1)
            
            metrics = {
                'f1_weighted_mean': cv_results['test_f1_weighted'].mean(),
                'f1_weighted_std': cv_results['test_f1_weighted'].std(),
                'f1_macro_mean': cv_results['test_f1_macro'].mean(),
                'f1_macro_std': cv_results['test_f1_macro'].std(),
                'precision_weighted_mean': cv_results['test_precision_weighted'].mean(),
                'precision_weighted_std': cv_results['test_precision_weighted'].std(),
                'recall_weighted_mean': cv_results['test_recall_weighted'].mean(),
                'recall_weighted_std': cv_results['test_recall_weighted'].std(),
                'accuracy_mean': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
            }
            
            return metrics
        except Exception as e:
            self.logger.warning(f"Comprehensive evaluation failed: {e}")
            return {
                'f1_weighted_mean': 0.0, 'f1_weighted_std': 0.0,
                'f1_macro_mean': 0.0, 'f1_macro_std': 0.0,
                'precision_weighted_mean': 0.0, 'precision_weighted_std': 0.0,
                'recall_weighted_mean': 0.0, 'recall_weighted_std': 0.0,
                'accuracy_mean': 0.0, 'accuracy_std': 0.0
            }
    
    def optimize_models(self, X: pd.DataFrame, y: pd.Series, 
                       domain: str = 'alzheimer') -> Dict[str, Any]:
        """
        Automatically optimize models for the given dataset
        """
        models, pool_name = self.select_model_pool(X, y, domain)
        
        results = {
            'pool_used': pool_name,
            'models_tested': [],
            'best_model': None,
            'best_score': 0,
            'optimization_notes': []
        }
        
        self.logger.info(f"🔍 Testing {len(models)} models from {pool_name} pool...")
        
        for name, model in models:
            try:
                # Quick cross-validation to assess performance
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                model_result = {
                    'name': name,
                    'cv_mean': mean_score,
                    'cv_std': std_score,
                    'cv_scores': cv_scores.tolist()
                }
                
                results['models_tested'].append(model_result)
                
                self.logger.info(f"   {name}: {mean_score:.3f} ± {std_score:.3f}")
                
                # Track best model
                if mean_score > results['best_score']:
                    results['best_score'] = mean_score
                    results['best_model'] = {
                        'name': name,
                        'model': model,
                        'cv_performance': mean_score
                    }
                
            except Exception as e:
                self.logger.warning(f"   {name} failed: {e}")
                results['optimization_notes'].append(f"{name} failed: {str(e)}")
        
        if results['best_model']:
            self.logger.info(f"✅ Best model: {results['best_model']['name']} "
                           f"({results['best_score']:.3f} CV accuracy)")
        
        return results
    
    def get_model_recommendations(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Get recommendations for model selection based on data analysis
        """
        recommendations = []
        n_samples, n_features = X.shape
        n_classes = len(y.unique())
        
        # Sample size recommendations
        if n_samples < 100:
            recommendations.append("⚠️ Very small dataset - consider data augmentation or simpler models")
        elif n_samples < 500:
            recommendations.append("📊 Small dataset - regularized models recommended")
        
        # Feature to sample ratio
        if n_features > n_samples:
            recommendations.append("🔍 High-dimensional data - feature selection strongly recommended")
        elif n_features > n_samples * 0.1:
            recommendations.append("📈 Many features relative to samples - consider dimensionality reduction")
        
        # Class imbalance
        class_counts = y.value_counts()
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        imbalance_ratio = max_class_size / min_class_size
        
        if imbalance_ratio > 10:
            recommendations.append("⚖️ Severe class imbalance - consider SMOTE or class weighting")
        elif imbalance_ratio > 3:
            recommendations.append("📊 Moderate class imbalance - balanced models recommended")
        
        # Alzheimer's-specific recommendations
        alzheimer_cols = [col for col in X.columns if any(term in col.lower() 
                         for term in ['cdr', 'mmse', 'adas', 'brain', 'volume'])]
        if alzheimer_cols:
            recommendations.append("🧠 Alzheimer's features detected - ensemble methods work well")
            if len(alzheimer_cols) > 5:
                recommendations.append("🔬 Rich neurological features - consider specialized preprocessing")
        
        return recommendations


def create_adaptive_pipeline(X: pd.DataFrame, y: pd.Series, 
                           domain: str = 'alzheimer') -> Dict[str, Any]:
    """
    Create an adaptive ML pipeline based on data characteristics
    """
    selector = DynamicModelSelector()
    
    # Get recommendations
    recommendations = selector.get_model_recommendations(X, y)
    
    # Optimize models
    results = selector.optimize_models(X, y, domain)
    
    # Add recommendations to results
    results['recommendations'] = recommendations
    
    return results


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Dynamic Model Selection Framework...")
    
    # Create sample data (simulating CDR prediction)
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate Alzheimer's-like features
    data = {
        'Age': np.random.normal(70, 10, n_samples),
        'MMSE': np.random.normal(25, 5, n_samples),
        'eTIV': np.random.normal(1500, 200, n_samples),
        'nWBV': np.random.normal(0.75, 0.1, n_samples),
        'Education': np.random.normal(12, 4, n_samples),
    }
    
    X = pd.DataFrame(data)
    # Simulate CDR based on features
    y = pd.Series(np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]))
    
    # Test the framework
    results = create_adaptive_pipeline(X, y, domain='alzheimer')
    
    print(f"\n✅ Framework test complete!")
    print(f"Best model: {results['best_model']['name']}")
    print(f"Performance: {results['best_score']:.3f}")