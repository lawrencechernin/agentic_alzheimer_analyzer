#!/usr/bin/env python3
"""
Clinical Evaluation Metrics for Alzheimer's Research
Comprehensive evaluation including F1, precision, recall, and clinical-relevant metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report,
    make_scorer
)
import logging


class ClinicalEvaluator:
    """
    Comprehensive clinical evaluation for Alzheimer's prediction models
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model_comprehensive(self, model: Any, X: pd.DataFrame, 
                                   y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Comprehensive clinical evaluation including all relevant metrics
        """
        self.logger.info(f"🔍 Comprehensive clinical evaluation with {cv}-fold CV...")
        
        # Define comprehensive scoring metrics
        scoring = {
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'f1_micro': make_scorer(f1_score, average='micro', zero_division=0),
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'accuracy': 'accuracy'
        }
        
        # Add ROC AUC for binary/multiclass scenarios
        n_classes = len(y.unique())
        if n_classes == 2:
            scoring['roc_auc'] = 'roc_auc'
        elif n_classes > 2:
            scoring['roc_auc'] = make_scorer(roc_auc_score, average='weighted', 
                                           multi_class='ovr', needs_proba=True)
        
        try:
            # Perform cross-validation
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                      return_train_score=False, n_jobs=-1)
            
            # Calculate comprehensive metrics
            metrics = {
                # F1 Scores
                'f1_weighted_mean': cv_results['test_f1_weighted'].mean(),
                'f1_weighted_std': cv_results['test_f1_weighted'].std(),
                'f1_macro_mean': cv_results['test_f1_macro'].mean(),
                'f1_macro_std': cv_results['test_f1_macro'].std(),
                'f1_micro_mean': cv_results['test_f1_micro'].mean(),
                'f1_micro_std': cv_results['test_f1_micro'].std(),
                
                # Precision Scores
                'precision_weighted_mean': cv_results['test_precision_weighted'].mean(),
                'precision_weighted_std': cv_results['test_precision_weighted'].std(),
                'precision_macro_mean': cv_results['test_precision_macro'].mean(),
                'precision_macro_std': cv_results['test_precision_macro'].std(),
                
                # Recall Scores
                'recall_weighted_mean': cv_results['test_recall_weighted'].mean(),
                'recall_weighted_std': cv_results['test_recall_weighted'].std(),
                'recall_macro_mean': cv_results['test_recall_macro'].mean(),
                'recall_macro_std': cv_results['test_recall_macro'].std(),
                
                # Accuracy
                'accuracy_mean': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
                
                # Additional metadata
                'n_classes': n_classes,
                'n_samples': len(X),
                'n_features': len(X.columns)
            }
            
            # Add ROC AUC if available
            if 'test_roc_auc' in cv_results:
                metrics['roc_auc_mean'] = cv_results['test_roc_auc'].mean()
                metrics['roc_auc_std'] = cv_results['test_roc_auc'].std()
            
            # Calculate clinical decision metrics
            clinical_metrics = self._calculate_clinical_metrics(metrics)
            metrics.update(clinical_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Comprehensive evaluation failed: {e}")
            return self._get_default_metrics()
    
    def _calculate_clinical_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate clinical decision-relevant metrics
        """
        clinical = {
            # Clinical Quality Score (weighted combination for medical applications)
            'clinical_quality_score': (
                metrics['f1_weighted_mean'] * 0.4 +  # Primary metric for imbalanced clinical data
                metrics['precision_weighted_mean'] * 0.3 +  # Minimize false positives
                metrics['recall_weighted_mean'] * 0.2 +     # Minimize false negatives
                metrics['accuracy_mean'] * 0.1              # Overall correctness
            ),
            
            # Reliability indicators
            'stability_score': 1 - (metrics['f1_weighted_std'] / max(metrics['f1_weighted_mean'], 0.001)),
            'precision_reliability': metrics['precision_weighted_mean'] - metrics['precision_weighted_std'],
            'recall_reliability': metrics['recall_weighted_mean'] - metrics['recall_weighted_std'],
            
            # Clinical interpretation flags
            'high_precision': metrics['precision_weighted_mean'] >= 0.80,
            'high_recall': metrics['recall_weighted_mean'] >= 0.80,
            'high_f1': metrics['f1_weighted_mean'] >= 0.80,
            'clinically_acceptable': metrics['f1_weighted_mean'] >= 0.75 and 
                                   metrics['precision_weighted_mean'] >= 0.75 and
                                   metrics['recall_weighted_mean'] >= 0.75
        }
        
        return clinical
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """
        Return default metrics when evaluation fails
        """
        return {
            'f1_weighted_mean': 0.0, 'f1_weighted_std': 0.0,
            'f1_macro_mean': 0.0, 'f1_macro_std': 0.0,
            'f1_micro_mean': 0.0, 'f1_micro_std': 0.0,
            'precision_weighted_mean': 0.0, 'precision_weighted_std': 0.0,
            'precision_macro_mean': 0.0, 'precision_macro_std': 0.0,
            'recall_weighted_mean': 0.0, 'recall_weighted_std': 0.0,
            'recall_macro_mean': 0.0, 'recall_macro_std': 0.0,
            'accuracy_mean': 0.0, 'accuracy_std': 0.0,
            'clinical_quality_score': 0.0,
            'stability_score': 0.0,
            'clinically_acceptable': False
        }
    
    def print_clinical_summary(self, metrics: Dict[str, Any], model_name: str = "Model"):
        """
        Print formatted clinical evaluation summary
        """
        print(f"\n📊 Clinical Evaluation Summary - {model_name}")
        print("=" * 50)
        
        # Primary clinical metrics
        print(f"🎯 F1 Score (Weighted): {metrics['f1_weighted_mean']:.3f} ± {metrics['f1_weighted_std']:.3f}")
        print(f"🎯 F1 Score (Macro): {metrics['f1_macro_mean']:.3f} ± {metrics['f1_macro_std']:.3f}")
        print(f"🎯 Precision (Weighted): {metrics['precision_weighted_mean']:.3f} ± {metrics['precision_weighted_std']:.3f}")
        print(f"🎯 Recall (Weighted): {metrics['recall_weighted_mean']:.3f} ± {metrics['recall_weighted_std']:.3f}")
        print(f"🎯 Accuracy: {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
        
        if 'roc_auc_mean' in metrics:
            print(f"🎯 ROC AUC: {metrics['roc_auc_mean']:.3f} ± {metrics['roc_auc_std']:.3f}")
        
        # Clinical decision metrics
        print(f"\n🏥 Clinical Quality Score: {metrics['clinical_quality_score']:.3f}")
        print(f"🏥 Stability Score: {metrics['stability_score']:.3f}")
        print(f"🏥 Clinically Acceptable: {'✅ Yes' if metrics['clinically_acceptable'] else '❌ No'}")
        
        # Performance flags
        flags = []
        if metrics['high_precision']: flags.append("High Precision")
        if metrics['high_recall']: flags.append("High Recall")
        if metrics['high_f1']: flags.append("High F1")
        
        if flags:
            print(f"🏆 Performance Flags: {', '.join(flags)}")
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models based on clinical metrics
        """
        if not model_results:
            return {}
        
        # Find best models for different criteria
        best_f1 = max(model_results, key=lambda x: x['metrics']['f1_weighted_mean'])
        best_precision = max(model_results, key=lambda x: x['metrics']['precision_weighted_mean'])
        best_recall = max(model_results, key=lambda x: x['metrics']['recall_weighted_mean'])
        best_clinical = max(model_results, key=lambda x: x['metrics']['clinical_quality_score'])
        
        comparison = {
            'best_f1_model': {
                'name': best_f1['name'],
                'f1_score': best_f1['metrics']['f1_weighted_mean']
            },
            'best_precision_model': {
                'name': best_precision['name'],
                'precision': best_precision['metrics']['precision_weighted_mean']
            },
            'best_recall_model': {
                'name': best_recall['name'],
                'recall': best_recall['metrics']['recall_weighted_mean']
            },
            'best_clinical_model': {
                'name': best_clinical['name'],
                'clinical_score': best_clinical['metrics']['clinical_quality_score']
            },
            'clinically_acceptable_models': [
                result['name'] for result in model_results 
                if result['metrics']['clinically_acceptable']
            ]
        }
        
        return comparison


if __name__ == "__main__":
    # Test the clinical evaluator
    print("🧪 Testing Clinical Evaluation Metrics...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create sample clinical data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, 
                              n_informative=8, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Test evaluation
    evaluator = ClinicalEvaluator()
    model = RandomForestClassifier(random_state=42)
    
    metrics = evaluator.evaluate_model_comprehensive(model, X, y)
    evaluator.print_clinical_summary(metrics, "Random Forest")
    
    print("✅ Clinical evaluation test complete!")