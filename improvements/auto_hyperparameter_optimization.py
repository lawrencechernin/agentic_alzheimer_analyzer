#!/usr/bin/env python3
"""
Automated Hyperparameter Optimization for Alzheimer's Research
Smart hyperparameter tuning with domain-specific parameter spaces
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score
import logging
from scipy.stats import randint, uniform

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class AutoHyperparameterOptimizer:
    """
    Automated hyperparameter optimization specialized for Alzheimer's research
    """
    
    def __init__(self, optimization_budget: int = 100, scoring: str = 'f1_weighted'):
        self.optimization_budget = optimization_budget
        self.scoring = scoring
        self.logger = logging.getLogger(__name__)
        
        # Define parameter spaces for different models
        self.parameter_spaces = self._define_parameter_spaces()
        
    def _define_parameter_spaces(self) -> Dict[str, Dict]:
        """
        Define hyperparameter search spaces optimized for cognitive data
        """
        spaces = {
            'RandomForest': {
                'grid_search': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 15, 20, 25, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5],
                    'bootstrap': [True, False]
                },
                'random_search': {
                    'n_estimators': randint(50, 500),
                    'max_depth': [5, 10, 15, 20, 25, 30, None],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5],
                    'bootstrap': [True, False]
                }
            },
            
            'GradientBoosting': {
                'grid_search': {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.1, 0.15, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 3, 5, 10],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'random_search': {
                    'n_estimators': randint(50, 300),
                    'learning_rate': uniform(0.01, 0.25),
                    'max_depth': randint(3, 12),
                    'min_samples_split': randint(2, 25),
                    'min_samples_leaf': randint(1, 15),
                    'subsample': uniform(0.6, 0.4)
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            spaces['XGBoost'] = {
                'grid_search': {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.1, 0.15, 0.2],
                    'max_depth': [3, 5, 6, 7, 8],
                    'min_child_weight': [1, 3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.3]
                },
                'random_search': {
                    'n_estimators': randint(50, 300),
                    'learning_rate': uniform(0.01, 0.25),
                    'max_depth': randint(3, 10),
                    'min_child_weight': randint(1, 10),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'gamma': uniform(0, 0.5)
                }
            }
        
        return spaces
    
    def optimize_model(self, model_name: str, base_model: Any, X: pd.DataFrame, 
                      y: pd.Series, method: str = 'random') -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model
        """
        if model_name not in self.parameter_spaces:
            self.logger.warning(f"No parameter space defined for {model_name}")
            return {'model': base_model, 'best_params': {}, 'best_score': 0}
        
        param_space = self.parameter_spaces[model_name]
        
        self.logger.info(f"🔍 Optimizing {model_name} using {method} search...")
        
        if method == 'grid' and 'grid_search' in param_space:
            return self._grid_search_optimize(base_model, param_space['grid_search'], X, y)
        elif method == 'random' and 'random_search' in param_space:
            return self._random_search_optimize(base_model, param_space['random_search'], X, y)
        elif method == 'optuna' and OPTUNA_AVAILABLE:
            return self._optuna_optimize(model_name, base_model, X, y)
        else:
            # Fallback to random search
            return self._random_search_optimize(base_model, 
                                              param_space.get('random_search', param_space.get('grid_search', {})), 
                                              X, y)
    
    def _grid_search_optimize(self, model: Any, param_grid: Dict, 
                            X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform grid search optimization
        """
        # Limit grid size for computational efficiency
        total_combinations = 1
        for param_values in param_grid.values():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        if total_combinations > 500:
            self.logger.info(f"   Large parameter space ({total_combinations} combinations), using subset...")
            # Reduce parameter space
            reduced_grid = {}
            for param, values in param_grid.items():
                if isinstance(values, list) and len(values) > 3:
                    reduced_grid[param] = values[:3]  # Take first 3 values
                else:
                    reduced_grid[param] = values
            param_grid = reduced_grid
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=self.scoring, n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'optimization_method': 'grid_search'
        }
    
    def _random_search_optimize(self, model: Any, param_dist: Dict,
                              X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform randomized search optimization
        """
        n_iter = min(self.optimization_budget, 50)  # Reasonable default
        
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=n_iter, cv=5, scoring=self.scoring, 
            n_jobs=-1, random_state=42, verbose=0
        )
        
        random_search.fit(X, y)
        
        return {
            'model': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'optimization_method': 'random_search'
        }
    
    def _optuna_optimize(self, model_name: str, model: Any, 
                        X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform Optuna-based optimization (if available)
        """
        def objective(trial):
            # Define parameter suggestions based on model type
            if model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }
                optimized_model = RandomForestClassifier(random_state=42, **params)
                
            elif model_name == 'GradientBoosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 25),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                optimized_model = GradientBoostingClassifier(random_state=42, **params)
                
            elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5)
                }
                optimized_model = XGBClassifier(random_state=42, eval_metric='mlogloss', **params)
            else:
                return 0.0
            
            # Cross-validation score
            scores = cross_val_score(optimized_model, X, y, cv=5, scoring=self.scoring)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=min(self.optimization_budget, 100))
        
        # Create best model
        best_params = study.best_params
        if model_name == 'RandomForest':
            best_model = RandomForestClassifier(random_state=42, **best_params)
        elif model_name == 'GradientBoosting':
            best_model = GradientBoostingClassifier(random_state=42, **best_params)
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            best_model = XGBClassifier(random_state=42, eval_metric='mlogloss', **best_params)
        
        best_model.fit(X, y)
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': study.best_value,
            'optimization_method': 'optuna'
        }
    
    def optimize_multiple_models(self, models: Dict[str, Any], X: pd.DataFrame, 
                                y: pd.Series, method: str = 'random') -> Dict[str, Any]:
        """
        Optimize multiple models and return the best performing one
        """
        self.logger.info(f"🚀 Optimizing {len(models)} models using {method} search...")
        
        results = {
            'optimized_models': {},
            'best_model': None,
            'best_score': 0,
            'optimization_summary': []
        }
        
        for model_name, model in models.items():
            try:
                optimization_result = self.optimize_model(model_name, model, X, y, method)
                
                results['optimized_models'][model_name] = optimization_result
                
                # Track best model
                if optimization_result['best_score'] > results['best_score']:
                    results['best_score'] = optimization_result['best_score']
                    results['best_model'] = {
                        'name': model_name,
                        'model': optimization_result['model'],
                        'params': optimization_result['best_params'],
                        'score': optimization_result['best_score']
                    }
                
                # Add to summary
                results['optimization_summary'].append({
                    'model': model_name,
                    'best_score': optimization_result['best_score'],
                    'method': optimization_result['optimization_method'],
                    'best_params': optimization_result['best_params']
                })
                
                self.logger.info(f"   {model_name}: {optimization_result['best_score']:.3f}")
                
            except Exception as e:
                self.logger.error(f"   {model_name} optimization failed: {e}")
        
        if results['best_model']:
            self.logger.info(f"✅ Best optimized model: {results['best_model']['name']} "
                           f"({results['best_score']:.3f} {self.scoring})")
        
        return results
    
    def get_optimization_recommendations(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Get recommendations for optimization strategy based on data characteristics
        """
        recommendations = []
        n_samples, n_features = X.shape
        
        # Budget recommendations based on data size
        if n_samples < 1000:
            recommendations.append("💡 Small dataset: Grid search recommended for thorough exploration")
            recommendations.append("⚡ Consider lower optimization budget (20-50 trials)")
        elif n_samples < 5000:
            recommendations.append("🎯 Medium dataset: Random search provides good balance")
            recommendations.append("📊 Recommended budget: 50-100 trials")
        else:
            recommendations.append("🚀 Large dataset: Random search or Optuna for efficiency")
            recommendations.append("⚡ Higher budget justified (100+ trials)")
        
        # Feature-specific recommendations
        if n_features > n_samples * 0.1:
            recommendations.append("🔍 High-dimensional data: Focus on regularization parameters")
        
        # Alzheimer's-specific recommendations
        alzheimer_cols = [col for col in X.columns if any(term in col.lower() 
                         for term in ['cdr', 'mmse', 'brain', 'cognitive'])]
        if alzheimer_cols:
            recommendations.append("🧠 Alzheimer's features detected: Tree-based models often perform well")
            recommendations.append("📈 Consider ensemble methods with optimized base learners")
        
        return recommendations


def create_optimization_pipeline(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                               budget: int = 50, method: str = 'random') -> Dict[str, Any]:
    """
    Create complete hyperparameter optimization pipeline
    """
    optimizer = AutoHyperparameterOptimizer(optimization_budget=budget, scoring='f1_weighted')
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations(X, y)
    
    # Optimize models
    results = optimizer.optimize_multiple_models(models, X, y, method)
    
    # Add recommendations
    results['optimization_recommendations'] = recommendations
    
    return results


if __name__ == "__main__":
    # Test the optimization pipeline
    print("🧪 Testing Automated Hyperparameter Optimization...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'Age': np.random.normal(70, 10, n_samples),
        'MMSE': np.random.normal(25, 5, n_samples),
        'eTIV': np.random.normal(1500, 200, n_samples),
        'nWBV': np.random.normal(0.75, 0.1, n_samples),
    })
    
    y = pd.Series(np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]))
    
    # Test models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Test optimization
    results = create_optimization_pipeline(models, X, y, budget=20, method='random')
    
    print(f"\n✅ Optimization test complete!")
    print(f"Best model: {results['best_model']['name']}")
    print(f"Best score: {results['best_score']:.3f}")