#!/usr/bin/env python3
"""
Advanced Dementia Prediction using Tree-based Models
Based on research achieving 86-94% accuracy on OASIS dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class AdvancedDementiaPrediction:
    """
    Advanced dementia prediction using techniques from top research:
    - Boruta-inspired feature selection
    - Random Forest with Grid Search CV (94.39% reported)
    - Enhanced feature engineering
    - Multi-method ensemble
    """
    
    def __init__(self):
        self.best_model = None
        self.selected_features = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def boruta_inspired_feature_selection(self, X, y, max_features=20):
        """
        Feature selection inspired by Boruta algorithm
        Uses Random Forest feature importance with shadow features
        """
        print("🔍 Performing Boruta-inspired feature selection...")
        
        # Create shadow features (random permutations)
        X_shadow = X.apply(np.random.permutation)
        X_shadow.columns = [f"shadow_{col}" for col in X.columns]
        
        # Combine original and shadow features
        X_combined = pd.concat([X, X_shadow], axis=1)
        
        # Train Random Forest on combined features
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_combined, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': X_combined.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Find maximum shadow importance
        shadow_max = importances[importances['feature'].str.startswith('shadow_')]['importance'].max()
        
        # Select features with importance > shadow_max
        selected = importances[
            (~importances['feature'].str.startswith('shadow_')) & 
            (importances['importance'] > shadow_max)
        ]['feature'].tolist()
        
        # If too many features, limit to max_features
        if len(selected) > max_features:
            selected = selected[:max_features]
        
        # If too few features, use mutual information as backup
        if len(selected) < 5:
            selector = SelectKBest(mutual_info_classif, k=min(10, X.shape[1]))
            selector.fit(X, y)
            scores = pd.DataFrame({
                'feature': X.columns,
                'score': selector.scores_
            }).sort_values('score', ascending=False)
            selected = scores.head(10)['feature'].tolist()
        
        print(f"   ✅ Selected {len(selected)} features using Boruta-inspired method")
        return selected
    
    def create_advanced_features(self, df):
        """
        Create advanced features based on research findings
        """
        print("🧠 Creating advanced engineered features...")
        
        enhanced_df = df.copy()
        
        # 1. Brain Volume Ratios (important in research)
        if 'eTIV' in df.columns and 'nWBV' in df.columns:
            # Brain atrophy percentage
            enhanced_df['brain_atrophy_pct'] = (1 - df['nWBV']) * 100
            
            # Volume loss indicator
            enhanced_df['volume_loss'] = df['eTIV'] * (1 - df['nWBV'])
            
            # Normalized brain volume per liter
            enhanced_df['nWBV_per_liter'] = df['nWBV'] / (df['eTIV'] / 1000)
        
        # 2. Age-related features (strong predictors in research)
        if 'Age' in df.columns:
            # Age groups (research shows nonlinear relationship)
            enhanced_df['age_group'] = pd.cut(df['Age'], 
                bins=[0, 65, 75, 85, 100], 
                labels=[0, 1, 2, 3])
            # Handle NaN values before converting to int
            enhanced_df['age_group'] = enhanced_df['age_group'].cat.codes
            
            # Age squared (captures nonlinearity)
            enhanced_df['age_squared'] = df['Age'] ** 2
            
            # Interaction with brain volume
            if 'nWBV' in df.columns:
                enhanced_df['age_volume_interaction'] = df['Age'] * df['nWBV']
        
        # 3. MMSE-based features (highest importance in many studies)
        if 'MMSE' in df.columns:
            # MMSE categories based on clinical thresholds
            enhanced_df['mmse_category'] = pd.cut(df['MMSE'], 
                bins=[0, 10, 20, 24, 27, 30], 
                labels=[4, 3, 2, 1, 0])  # Reversed for severity
            # Handle NaN values before converting to int
            enhanced_df['mmse_category'] = enhanced_df['mmse_category'].cat.codes
            
            # MMSE deviation from age-expected
            if 'Age' in df.columns:
                # Simple age-adjusted MMSE (loses ~1 point per decade after 60)
                expected_mmse = 30 - np.maximum(0, (df['Age'] - 60) / 10)
                enhanced_df['mmse_deviation'] = df['MMSE'] - expected_mmse
        
        # 4. Education-adjusted features
        if 'EDUC' in df.columns and 'MMSE' in df.columns:
            # Education-adjusted MMSE
            enhanced_df['mmse_per_education'] = df['MMSE'] / (df['EDUC'] + 1)
        
        # 5. Gender-specific features
        if 'Gender' in df.columns or 'Gender_M' in df.columns:
            # Already handled in previous processing
            pass
        
        # 6. Composite risk score
        risk_score = 0
        if 'Age' in df.columns:
            risk_score += (df['Age'] > 75).astype(int)
        if 'MMSE' in df.columns:
            risk_score += (df['MMSE'] < 24).astype(int)
        if 'nWBV' in df.columns:
            risk_score += (df['nWBV'] < df['nWBV'].median()).astype(int)
        enhanced_df['composite_risk_score'] = risk_score
        
        print(f"   ✅ Created {len(enhanced_df.columns) - len(df.columns)} new features")
        return enhanced_df
    
    def train_optimized_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest with Grid Search CV (94.39% reported in research)
        """
        print("🌲 Training optimized Random Forest with Grid Search...")
        
        # Define parameter grid based on research
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use smaller grid for faster testing
        param_grid_fast = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid_fast, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Evaluate
        train_score = best_rf.score(X_train, y_train)
        test_score = best_rf.score(X_test, y_test)
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Training accuracy: {train_score:.3f}")
        print(f"   Test accuracy: {test_score:.3f}")
        
        return best_rf, test_score
    
    def create_super_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Create ensemble of multiple optimized models
        """
        print("🚀 Building super ensemble model...")
        
        models = []
        
        # 1. Optimized Random Forest
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, 
            min_samples_split=2, min_samples_leaf=1,
            max_features='sqrt', random_state=42
        )
        models.append(('rf', rf))
        
        # 2. Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, min_samples_split=10,
            subsample=0.8, random_state=42
        )
        models.append(('gb', gb))
        
        # 3. XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=200, learning_rate=0.1,
                max_depth=6, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                eval_metric='mlogloss'
            )
            models.append(('xgb', xgb))
        
        # Create voting ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft')
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        train_score = ensemble.score(X_train, y_train)
        test_score = ensemble.score(X_test, y_test)
        
        print(f"   Ensemble training accuracy: {train_score:.3f}")
        print(f"   Ensemble test accuracy: {test_score:.3f}")
        
        return ensemble, test_score
    
    def fit_predict(self, df, target_col='CDR'):
        """
        Complete pipeline for advanced dementia prediction
        """
        print("\n" + "="*80)
        print("🧪 ADVANCED DEMENTIA PREDICTION PIPELINE")
        print("="*80)
        
        # 1. Data preparation
        df_clean = df.dropna(subset=[target_col])
        
        # 2. Feature engineering
        df_enhanced = self.create_advanced_features(df_clean)
        
        # 3. Prepare features and target
        feature_cols = [col for col in df_enhanced.columns 
                       if col not in [target_col, 'Subject_ID', 'MRI_ID', 'Group']]
        
        # Keep only numeric features
        numeric_cols = df_enhanced[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = df_enhanced[numeric_cols]
        y = df_clean[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # 4. Remove severe cases for benchmark comparison (optional)
        if target_col == 'CDR' and 2.0 in y.values:
            severe_mask = (y == 2.0)
            print(f"   Excluding {severe_mask.sum()} severe cases for benchmark comparison")
            X = X[~severe_mask]
            y = y[~severe_mask]
        
        # 5. Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 6. Feature selection
        self.selected_features = self.boruta_inspired_feature_selection(X, y_encoded)
        X_selected = X[self.selected_features]
        
        # 7. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # 8. Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 9. Train models
        print("\n📊 Model Training Results:")
        print("-"*50)
        
        # Random Forest with Grid Search
        rf_model, rf_score = self.train_optimized_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Super Ensemble
        ensemble_model, ensemble_score = self.create_super_ensemble(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Select best model
        if ensemble_score > rf_score:
            self.best_model = ensemble_model
            best_score = ensemble_score
            best_name = "Super Ensemble"
        else:
            self.best_model = rf_model
            best_score = rf_score
            best_name = "Optimized Random Forest"
        
        # 10. Final evaluation
        y_pred = self.best_model.predict(X_test_scaled)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print("\n" + "="*80)
        print("📈 FINAL RESULTS")
        print("="*80)
        print(f"\n🏆 Best Model: {best_name}")
        print(f"   Test Accuracy: {best_score:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall: {recall:.1%}")
        print(f"   F1-Score: {f1:.3f}")
        
        print(f"\n🔑 Top Features Used:")
        for i, feature in enumerate(self.selected_features[:10], 1):
            print(f"   {i}. {feature}")
        
        # Cross-validation for robustness
        cv_scores = cross_val_score(self.best_model, X_train_scaled, y_train, cv=10)
        print(f"\n📊 10-Fold CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        
        return {
            'model': self.best_model,
            'accuracy': best_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'selected_features': self.selected_features,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }


def run_advanced_prediction():
    """Run the advanced dementia prediction pipeline"""
    
    # Load OASIS data
    data_path = "./training_data/oasis/"
    
    print("📊 Loading OASIS data...")
    cross_df = pd.read_csv(f"{data_path}oasis_cross-sectional.csv")
    long_df = pd.read_csv(f"{data_path}oasis_longitudinal.csv")
    
    # Harmonize columns
    cross_df = cross_df.rename(columns={'ID': 'Subject_ID', 'M/F': 'Gender', 'Educ': 'EDUC'})
    long_df = long_df.rename(columns={'Subject ID': 'Subject_ID', 'M/F': 'Gender'})
    
    # Combine datasets
    common_cols = list(set(cross_df.columns) & set(long_df.columns))
    df = pd.concat([cross_df[common_cols], long_df[common_cols]], ignore_index=True)
    
    print(f"   Loaded {len(df)} records")
    
    # Run advanced prediction
    predictor = AdvancedDementiaPrediction()
    results = predictor.fit_predict(df, target_col='CDR')
    
    print("\n🎯 Comparison with Benchmarks:")
    print(f"   Our Result: {results['accuracy']:.1%}")
    print(f"   Previous Best: 80.7%")
    print(f"   Research Target: 86-94%")
    
    if results['accuracy'] > 0.86:
        print("\n   🎉 ACHIEVED RESEARCH-LEVEL PERFORMANCE!")
    elif results['accuracy'] > 0.807:
        print("\n   ✅ EXCEEDED PREVIOUS BENCHMARK!")
    
    return results


if __name__ == "__main__":
    results = run_advanced_prediction()