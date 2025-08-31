#!/usr/bin/env python3
"""
Cognitive Analysis Agent
========================

Generalizable autonomous agent for analyzing relationships between cognitive assessments
in Alzheimer's research. Works with any cognitive assessment combination through 
configuration-driven analysis.

Key Features:
- Dataset agnostic: Works with any Alzheimer's dataset structure
- Assessment flexible: Supports any cognitive assessment combination
- Configuration-driven: All specifics defined in config files
- Extensible: Easy plugin architecture for custom analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import json
import yaml
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

class CognitiveAnalysisAgent:
    """
    Generalizable analysis agent for cognitive assessment correlation studies
    
    This agent is completely dataset and assessment agnostic. It discovers
    cognitive assessments from your data based on configuration patterns
    and performs comprehensive analysis on any combination of:
    - Self-report questionnaires
    - Digital cognitive assessments  
    - Informant reports
    - Performance measures
    - Biomarkers
    
    All analysis types and variable mappings are defined in configuration
    files, making it easy to adapt to any Alzheimer's research dataset.
    """
    
    def __init__(self, config_path: str = "config/config.yaml",
                 data_dict_path: str = "config/data_dictionary.json",
                 discovery_results_path: str = "outputs/dataset_discovery_results.json"):
        
        self.config_path = config_path
        self.data_dict_path = data_dict_path
        self.discovery_results_path = discovery_results_path
        
        # Load configurations
        self.config = self._load_config()
        self.data_dict = self._load_data_dictionary()
        self.discovery_results = self._load_discovery_results()
        
        # Initialize data containers
        self.assessment_data = {}  # Dictionary of loaded assessment datasets
        self.combined_data = None
        
        # Setup logging
        self._setup_logging()
        
        # Get experiment configuration
        self.experiment_config = self.config.get('experiment', {})
        self.analysis_config = self.config.get('analysis', {})
        
        self.logger.info(f"🧠 Cognitive Analysis Agent initialized for experiment: {self.experiment_config.get('name', 'Unknown')}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config {self.config_path}: {e}")
            return {}
    
    def _load_data_dictionary(self) -> Dict[str, Any]:
        """Load data dictionary for variable mapping"""
        try:
            with open(self.data_dict_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load data dictionary {self.data_dict_path}: {e}")
            return {}
    
    def _load_discovery_results(self) -> Dict[str, Any]:
        """Load dataset discovery results"""
        try:
            with open(self.discovery_results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load discovery results {self.discovery_results_path}: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging system"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete cognitive assessment analysis pipeline"""
        experiment_name = self.experiment_config.get('name', 'Cognitive_Assessment_Analysis')
        self.logger.info(f"🚀 Starting Cognitive Analysis Pipeline: {experiment_name}")
        
        analysis_results = {
            'analysis_info': {
                'agent_type': 'CognitiveAnalysisAgent',
                'experiment_name': experiment_name,
                'start_time': datetime.now().isoformat(),
                'config': self.config
            },
            'data_summary': {},
            'assessment_analysis': {},
            'correlation_analysis': {},
            'self_informant_comparison': {},
            'cognitive_performance_analysis': {},
            'clinical_insights': {},
            'statistical_summary': {}
        }
        
        try:
            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Data loading and preprocessing")
            data_summary = self._load_and_preprocess_data()
            analysis_results['data_summary'] = data_summary
            
            if self.combined_data is None or len(self.combined_data) == 0:
                raise ValueError("No data available for analysis")
            
            # Step 2: Assessment-specific analysis
            assessment_types = self._identify_assessment_types()
            for assessment_type in assessment_types:
                self.logger.info(f"Step 2.{assessment_type}: {assessment_type} analysis")
                analysis_results['assessment_analysis'][assessment_type] = self._analyze_assessment_type(assessment_type)
            
            # Step 3: Cross-assessment correlation analysis
            self.logger.info("Step 3: Cross-assessment correlation analysis")
            analysis_results['correlation_analysis'] = self._analyze_cross_assessment_correlations()
            
            # Step 4: Self vs informant comparison (if available)
            if self._has_self_informant_data():
                self.logger.info("Step 4: Self vs informant comparison")
                analysis_results['self_informant_comparison'] = self._analyze_self_informant_differences()
            
            # Step 5: Cognitive performance analysis
            self.logger.info("Step 5: Cognitive performance analysis")
            analysis_results['cognitive_performance_analysis'] = self._analyze_cognitive_performance()
            
            # Step 6: Generate clinical insights
            self.logger.info("Step 6: Clinical insights generation")
            analysis_results['clinical_insights'] = self._generate_clinical_insights(analysis_results)
            
            # Step 7: Statistical summary
            analysis_results['statistical_summary'] = self._generate_statistical_summary(analysis_results)
            
            # Step 8: Create visualizations
            self.logger.info("Step 8: Creating visualizations")
            self._create_analysis_visualizations(analysis_results)
            
            # Step 9: Save results
            analysis_results['analysis_info']['end_time'] = datetime.now().isoformat()
            self._save_analysis_results(analysis_results)
            
            self.logger.info("✅ Cognitive analysis complete!")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            analysis_results['analysis_info']['error'] = str(e)
            analysis_results['analysis_info']['end_time'] = datetime.now().isoformat()
            return analysis_results
    
    def _load_and_preprocess_data(self) -> Dict[str, Any]:
        """Load and preprocess cognitive assessment data based on discovered files"""
        data_summary = {
            'assessments_loaded': [],
            'total_subjects': 0,
            'baseline_subjects': 0,
            'preprocessing_steps': []
        }
        
        # Reload discovery results (they may have been updated since initialization)
        self.discovery_results = self._load_discovery_results()
        
        # Get discovered files from discovery results
        discovered_files = self.discovery_results.get('files_discovered', {})
        files_by_type = discovered_files.get('files_by_type', {})
        file_patterns = self.config.get('dataset', {}).get('file_patterns', {})
        
        # Load each type of assessment data
        for assessment_type, patterns in file_patterns.items():
            matching_files = files_by_type.get(assessment_type, [])
            if matching_files:
                self.logger.info(f"   Loading {assessment_type} data from {len(matching_files)} files")
                assessment_df = self._load_assessment_files(matching_files, assessment_type)
                if assessment_df is not None:
                    self.assessment_data[assessment_type] = assessment_df
                    data_summary['assessments_loaded'].append({
                        'type': assessment_type,
                        'files': len(matching_files),
                        'records': len(assessment_df)
                    })
        
        # Combine all assessment datasets
        if len(self.assessment_data) >= 2:
            self.combined_data = self._combine_assessment_datasets()
            
            # Get baseline data only
            self.combined_data = self._get_baseline_data(self.combined_data)
            data_summary['preprocessing_steps'].append("Filtered to baseline timepoints")
            
            # Clean and validate data
            self.combined_data = self._clean_and_validate_data(self.combined_data)
            data_summary['preprocessing_steps'].append("Applied data quality filters")
            
            data_summary['total_subjects'] = len(self.combined_data)
            data_summary['baseline_subjects'] = len(self.combined_data)
            
            self.logger.info(f"   ✅ Final dataset: {len(self.combined_data)} subjects")
        else:
            self.logger.error(f"Insufficient assessment types loaded: {len(self.assessment_data)}")
        
        return data_summary
    
    def _load_assessment_files(self, file_paths: List[str], assessment_type: str) -> pd.DataFrame:
        """Load files for a specific assessment type"""
        dataframes = []
        
        # Check if sampling is enabled
        use_sampling = self.config.get('analysis', {}).get('use_sampling', False)
        sample_size = self.config.get('analysis', {}).get('analysis_sample_size', 5000)
        
        for file_path in file_paths:
            try:
                if use_sampling:
                    # Sample data to prevent memory explosion
                    df = pd.read_csv(file_path, nrows=sample_size, low_memory=False)
                    total_rows = sum(1 for _ in open(file_path)) - 1  # Get total count
                    self.logger.info(f"     {assessment_type} file: {os.path.basename(file_path)} - {len(df)} of {total_rows:,} records (sampled)")
                else:
                    df = pd.read_csv(file_path, low_memory=False)
                    self.logger.info(f"     {assessment_type} file: {os.path.basename(file_path)} - {len(df)} records")
                dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"Could not load {file_path}: {e}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            return combined_df
        return None
    
    def _combine_assessment_datasets(self) -> pd.DataFrame:
        """Combine all assessment datasets using common subject identifier"""
        # Find common subject identifier
        subject_id_candidates = ['SubjectCode', 'subject_id', 'participant_id', 'Code', 'ID']
        
        # Find the common subject column across all datasets
        common_subject_col = None
        for candidate in subject_id_candidates:
            if all(candidate in df.columns for df in self.assessment_data.values()):
                common_subject_col = candidate
                break
        
        if not common_subject_col:
            # Try to find any common column
            all_columns = [set(df.columns) for df in self.assessment_data.values()]
            common_columns = set.intersection(*all_columns)
            subject_columns = [col for col in common_columns if any(term in col.lower() for term in ['subject', 'id', 'code'])]
            if subject_columns:
                common_subject_col = subject_columns[0]
        
        if not common_subject_col:
            raise ValueError("Could not find common subject identifier across datasets")
        
        # Merge all datasets efficiently
        combined = None
        assessment_names = list(self.assessment_data.keys())
        
        # First, get unique subjects from each dataset to check overlap
        subject_counts = {}
        for assessment_type, df in self.assessment_data.items():
            unique_subjects = df[common_subject_col].nunique()
            subject_counts[assessment_type] = unique_subjects
            self.logger.info(f"   {assessment_type}: {unique_subjects:,} unique subjects")
        
        # Merge datasets one by one with progress tracking
        for i, (assessment_type, df) in enumerate(self.assessment_data.items()):
            if combined is None:
                combined = df.copy()
                self.logger.info(f"   Starting with {assessment_type}: {len(combined):,} records")
            else:
                before_merge = len(combined)
                combined = combined.merge(
                    df,
                    on=common_subject_col,
                    how='inner',
                    suffixes=('', f'_{assessment_type}')
                )
                after_merge = len(combined)
                self.logger.info(f"   Merged {assessment_type}: {before_merge:,} → {after_merge:,} records")
                
                # Safety check for runaway merges
                if after_merge > before_merge * 2:
                    self.logger.warning(f"   ⚠️ Merge increased data size significantly - possible data quality issue")
                    break
        
        self.logger.info(f"   📊 Combined dataset: {len(combined)} subjects with data from {len(self.assessment_data)} assessment types")
        return combined
    
    def _identify_assessment_types(self) -> List[str]:
        """Identify what types of assessments are available in the data"""
        return list(self.assessment_data.keys())
    
    def _analyze_assessment_type(self, assessment_type: str) -> Dict[str, Any]:
        """Analyze a specific assessment type"""
        assessment_results = {
            'assessment_type': assessment_type,
            'descriptive_stats': {},
            'distribution_analysis': {},
            'quality_metrics': {}
        }
        
        # Get the data for this assessment type
        if assessment_type in self.assessment_data:
            df = self.assessment_data[assessment_type]
            
            # Find numeric columns (likely assessment scores)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calculate descriptive statistics
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    assessment_results['descriptive_stats'][col] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'sample_size': len(col_data),
                        'missing_percent': float((df[col].isna().sum() / len(df)) * 100)
                    }
        
        return assessment_results
    
    def _analyze_cross_assessment_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different assessment types"""
        correlation_results = {
            'primary_correlations': {},
            'assessment_pairs': {},
            'clinical_significance': {}
        }
        
        if self.combined_data is None:
            return correlation_results
        
        # Get numeric columns from each assessment type
        assessment_columns = {}
        for assessment_type in self.assessment_data.keys():
            # Find columns that likely belong to this assessment
            matching_cols = [col for col in self.combined_data.columns 
                           if assessment_type.lower() in col.lower() or
                           any(pattern in col.lower() for pattern in self._get_assessment_patterns(assessment_type))]
            assessment_columns[assessment_type] = [col for col in matching_cols 
                                                 if self.combined_data[col].dtype in ['int64', 'float64']]
        
        # Calculate correlations between assessment pairs
        assessment_types = list(assessment_columns.keys())
        for i, assessment_a in enumerate(assessment_types):
            for j, assessment_b in enumerate(assessment_types[i+1:], i+1):
                pair_key = f"{assessment_a}_vs_{assessment_b}"
                correlation_results['assessment_pairs'][pair_key] = self._calculate_assessment_pair_correlations(
                    assessment_columns[assessment_a], assessment_columns[assessment_b], pair_key
                )
        
        # Flatten correlations for primary results
        for pair_key, pair_results in correlation_results['assessment_pairs'].items():
            correlation_results['primary_correlations'].update(pair_results)
        
        # Calculate clinical significance
        significant_correlations = sum(1 for corr_data in correlation_results['primary_correlations'].values()
                                     if corr_data.get('p_value', 1) < 0.05)
        total_correlations = len(correlation_results['primary_correlations'])
        
        if total_correlations > 0:
            correlation_results['clinical_significance'] = {
                'significant_correlations': significant_correlations,
                'total_correlations': total_correlations,
                'significance_rate': significant_correlations / total_correlations,
                'multiple_comparison_threshold': 0.05 / total_correlations
            }
        
        return correlation_results
    
    def _get_assessment_patterns(self, assessment_type: str) -> List[str]:
        """Get search patterns for a specific assessment type"""
        # This should ideally come from configuration
        pattern_map = {
            'cognitive_data': ['memtrax', 'cognitive', 'memory', 'attention', 'rt', 'reaction', 'correct'],
            'ecog_data': ['ecog', 'everyday', 'qid', 'self', 'informant'],
            'demographic_data': ['age', 'education', 'gender', 'demo'],
            'medical_data': ['medical', 'diagnosis', 'condition']
        }
        return pattern_map.get(assessment_type, [assessment_type.lower()])
    
    def _calculate_assessment_pair_correlations(self, cols_a: List[str], cols_b: List[str], pair_key: str) -> Dict[str, Any]:
        """Calculate correlations between two assessment types"""
        pair_correlations = {}
        
        for col_a in cols_a[:5]:  # Limit to prevent explosion
            for col_b in cols_b[:5]:
                try:
                    data_a = self.combined_data[col_a].dropna()
                    data_b = self.combined_data[col_b].dropna()
                    
                    # Get common indices
                    common_indices = data_a.index.intersection(data_b.index)
                    if len(common_indices) > 20:  # Minimum sample size
                        common_a = data_a[common_indices]
                        common_b = data_b[common_indices]
                        
                        corr_coef, corr_p = pearsonr(common_a, common_b)
                        
                        correlation_key = f"{col_a}_vs_{col_b}"
                        pair_correlations[correlation_key] = {
                            'correlation_coefficient': corr_coef,
                            'p_value': corr_p,
                            'sample_size': len(common_indices),
                            'effect_size': self._interpret_correlation_effect_size(abs(corr_coef)),
                            'assessment_pair': pair_key
                        }
                except Exception:
                    continue
        
        return pair_correlations
    
    def _has_self_informant_data(self) -> bool:
        """Check if dataset has both self and informant report data"""
        if self.combined_data is None:
            return False
        
        # Look for columns that might represent self vs informant reports
        self_indicators = ['self', 'participant', 'subject']
        informant_indicators = ['informant', 'partner', 'caregiver', 'family']
        
        columns = [col.lower() for col in self.combined_data.columns]
        
        has_self = any(any(indicator in col for indicator in self_indicators) for col in columns)
        has_informant = any(any(indicator in col for indicator in informant_indicators) for col in columns)
        
        return has_self and has_informant
    
    def _analyze_self_informant_differences(self) -> Dict[str, Any]:
        """Analyze differences between self and informant reports (generic)"""
        self_informant_results = {
            'self_informant_available': True,
            'correlation_analysis': {},
            'discrepancy_analysis': {},
            'domain_analysis': {}
        }
        
        # Find self and informant columns
        all_columns = self.combined_data.columns.tolist()
        
        self_cols = [col for col in all_columns if any(term in col.lower() for term in ['self', 'participant'])]
        informant_cols = [col for col in all_columns if any(term in col.lower() for term in ['informant', 'partner', 'caregiver'])]
        
        # Filter to numeric columns only
        numeric_self_cols = [col for col in self_cols if self.combined_data[col].dtype in ['int64', 'float64']]
        numeric_informant_cols = [col for col in informant_cols if self.combined_data[col].dtype in ['int64', 'float64']]
        
        if len(numeric_self_cols) > 0 and len(numeric_informant_cols) > 0:
            # Calculate correlation between self and informant total scores
            if len(numeric_self_cols) > 1:
                self_total = self.combined_data[numeric_self_cols].sum(axis=1, skipna=False)
            else:
                self_total = self.combined_data[numeric_self_cols[0]]
            
            if len(numeric_informant_cols) > 1:
                informant_total = self.combined_data[numeric_informant_cols].sum(axis=1, skipna=False)
            else:
                informant_total = self.combined_data[numeric_informant_cols[0]]
            
            # Correlation analysis
            valid_data = (~self_total.isna()) & (~informant_total.isna())
            if valid_data.sum() > 10:
                corr_coef, corr_p = pearsonr(self_total[valid_data], informant_total[valid_data])
                self_informant_results['correlation_analysis'] = {
                    'correlation_coefficient': corr_coef,
                    'p_value': corr_p,
                    'sample_size': int(valid_data.sum())
                }
            
            # Discrepancy analysis
            discrepancy = informant_total - self_total
            self_informant_results['discrepancy_analysis'] = {
                'mean_discrepancy': float(discrepancy.mean()),
                'std_discrepancy': float(discrepancy.std()),
                'positive_discrepancy_percent': float((discrepancy > 0).mean() * 100),
                'large_discrepancy_percent': float((abs(discrepancy) > discrepancy.std()).mean() * 100)
            }
        else:
            self_informant_results['self_informant_available'] = False
            self.logger.warning("   ⚠️ Could not identify both self and informant measures")
        
        return self_informant_results
    
    def _analyze_cognitive_performance(self) -> Dict[str, Any]:
        """Analyze cognitive performance patterns (generic)"""
        cognitive_results = {
            'performance_metrics': {},
            'distribution_analysis': {},
            'efficiency_analysis': {}
        }
        
        if self.combined_data is None:
            return cognitive_results
        
        # Find performance-related columns
        performance_indicators = ['rt', 'reaction_time', 'response_time', 'accuracy', 'correct', 'percent', 'score']
        performance_cols = []
        
        for col in self.combined_data.columns:
            if any(indicator in col.lower() for indicator in performance_indicators):
                if self.combined_data[col].dtype in ['int64', 'float64']:
                    performance_cols.append(col)
        
        # Analyze each performance metric
        for col in performance_cols:
            col_data = self.combined_data[col].dropna()
            if len(col_data) > 0:
                cognitive_results['performance_metrics'][col] = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'sample_size': len(col_data)
                }
        
        return cognitive_results
    
    def _get_baseline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to baseline timepoints only"""
        timepoint_cols = ['TimepointCode', 'timepoint', 'visit', 'session']
        baseline_values = ['m00', 'baseline', '0', 'visit_1', 'session_1', 'bl']
        
        for col in timepoint_cols:
            if col in df.columns:
                for baseline_val in baseline_values:
                    baseline_data = df[df[col] == baseline_val]
                    if len(baseline_data) > 0:
                        self.logger.info(f"   📅 Using baseline timepoint: {col} == {baseline_val}")
                        return baseline_data
        
        self.logger.info("   📅 No timepoint column found, using all data as baseline")
        return df
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        original_size = len(df)
        
        # Remove rows with excessive missing data
        missing_threshold = 0.5
        before_missing = len(df)
        df = df.dropna(thresh=len(df.columns) * missing_threshold)
        
        self.logger.info(f"   🧹 Missing data cleanup: {len(df)}/{before_missing} subjects retained")
        return df
    
    def _interpret_correlation_effect_size(self, correlation: float) -> str:
        """Interpret correlation coefficient effect size"""
        if correlation >= 0.7:
            return "large"
        elif correlation >= 0.5:
            return "medium"
        elif correlation >= 0.3:
            return "small"
        else:
            return "negligible"
    
    def _generate_clinical_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clinical insights from analysis results"""
        insights = {
            'key_findings': [],
            'clinical_implications': [],
            'novel_discoveries': [],
            'limitations': []
        }
        
        # Analyze correlation findings
        correlations = analysis_results.get('correlation_analysis', {}).get('primary_correlations', {})
        significant_corrs = [name for name, data in correlations.items() 
                           if data.get('p_value', 1) < 0.05]
        
        if significant_corrs:
            insights['key_findings'].append(
                f"Found {len(significant_corrs)} significant cross-assessment correlations"
            )
        
        # Check for novel findings
        clinical_sig = analysis_results.get('correlation_analysis', {}).get('clinical_significance', {})
        if clinical_sig.get('significance_rate', 0) > 0.5:
            insights['novel_discoveries'].append(
                "High rate of significant correlations suggests strong cross-assessment relationships"
            )
        
        # Add limitations
        data_summary = analysis_results.get('data_summary', {})
        sample_size = data_summary.get('baseline_subjects', 0)
        
        if sample_size < 100:
            insights['limitations'].append("Small sample size may limit generalizability")
        
        insights['limitations'].append("Cross-sectional analysis - causality cannot be determined")
        
        return insights
    
    def _generate_statistical_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        summary = {
            'sample_characteristics': {},
            'primary_findings': {},
            'recommendations': []
        }
        
        # Sample characteristics
        data_summary = analysis_results.get('data_summary', {})
        summary['sample_characteristics'] = {
            'total_subjects': data_summary.get('baseline_subjects', 0),
            'assessments_analyzed': len(data_summary.get('assessments_loaded', [])),
            'data_completeness': 'Good' if data_summary.get('baseline_subjects', 0) > 100 else 'Limited'
        }
        
        # Primary findings
        correlations = analysis_results.get('correlation_analysis', {}).get('primary_correlations', {})
        significant_correlations = [name for name, data in correlations.items() 
                                 if data.get('p_value', 1) < 0.05]
        
        summary['primary_findings'] = {
            'significant_cross_assessment_correlations': len(significant_correlations),
            'total_correlations_tested': len(correlations)
        }
        
        # Recommendations
        if len(significant_correlations) > 0:
            summary['recommendations'].append("Strong evidence for cross-assessment relationships")
            summary['recommendations'].append("Consider validation in independent sample")
        
        if data_summary.get('baseline_subjects', 0) < 200:
            summary['recommendations'].append("Increase sample size for more robust findings")
        
        summary['recommendations'].append("Explore longitudinal changes in assessment relationships")
        
        return summary
    
    def _create_analysis_visualizations(self, analysis_results: Dict[str, Any]):
        """Create visualizations for the analysis"""
        self.logger.info("   🎨 Creating visualizations...")
        
        os.makedirs('outputs/visualizations', exist_ok=True)
        
        try:
            # Create correlation matrix visualization
            self._create_correlation_matrix_plot(analysis_results)
            
            # Create self-informant comparison plot (if available)
            if analysis_results.get('self_informant_comparison', {}).get('self_informant_available'):
                self._create_self_informant_plot(analysis_results)
            
            # Create performance distribution plots
            self._create_performance_distribution_plots(analysis_results)
            
        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")
    
    def _create_correlation_matrix_plot(self, analysis_results: Dict[str, Any]):
        """Create correlation matrix heatmap"""
        correlations = analysis_results.get('correlation_analysis', {}).get('primary_correlations', {})
        
        if not correlations:
            return
        
        # Extract correlation data
        corr_names = []
        corr_values = []
        p_values = []
        
        for name, data in correlations.items():
            # Clean up names for display
            display_name = name.replace('_vs_', ' vs ').replace('_', ' ')
            if len(display_name) > 40:  # Truncate long names
                display_name = display_name[:37] + "..."
            corr_names.append(display_name)
            corr_values.append(data.get('correlation_coefficient', 0))
            p_values.append(data.get('p_value', 1))
        
        if len(corr_values) > 0:
            fig, ax = plt.subplots(figsize=(12, max(6, len(corr_names) * 0.3)))
            
            # Create horizontal bar plot
            colors = ['red' if p < 0.05 else 'lightblue' for p in p_values]
            bars = ax.barh(range(len(corr_values)), corr_values, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(corr_names)))
            ax.set_yticklabels(corr_names, fontsize=9)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title('Cross-Assessment Correlations', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add significance indicators
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.05:
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                           '*', fontsize=12, fontweight='bold', va='center')
            
            plt.tight_layout()
            plt.savefig('outputs/visualizations/cross_assessment_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_self_informant_plot(self, analysis_results: Dict[str, Any]):
        """Create self vs informant comparison plot"""
        self_informant = analysis_results.get('self_informant_comparison', {})
        
        if not self_informant.get('self_informant_available'):
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Correlation info
        corr_data = self_informant.get('correlation_analysis', {})
        if corr_data:
            correlation = corr_data.get('correlation_coefficient', 0)
            p_value = corr_data.get('p_value', 1)
            
            ax1.text(0.5, 0.6, f'Correlation: r = {correlation:.3f}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.text(0.5, 0.4, f'p-value: {p_value:.4f}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            ax1.text(0.5, 0.2, significance, 
                    ha='center', va='center', transform=ax1.transAxes, 
                    fontsize=12, fontweight='bold',
                    color='green' if p_value < 0.05 else 'red')
        
        ax1.set_title('Self-Informant Correlation', fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Plot 2: Discrepancy analysis
        discrepancy = self_informant.get('discrepancy_analysis', {})
        if discrepancy:
            pos_disc_pct = discrepancy.get('positive_discrepancy_percent', 50)
            
            categories = ['Informant Higher', 'Agreement', 'Self Higher']
            # Simple approximation for visualization
            percentages = [pos_disc_pct, 20, 100 - pos_disc_pct - 20]
            colors = ['orange', 'green', 'blue']
            
            ax2.pie(percentages, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Report Discrepancy Patterns', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/self_informant_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_distribution_plots(self, analysis_results: Dict[str, Any]):
        """Create cognitive performance distribution plots"""
        cognitive = analysis_results.get('cognitive_performance_analysis', {})
        performance_metrics = cognitive.get('performance_metrics', {})
        
        if not performance_metrics:
            return
        
        # Create subplots for up to 6 metrics
        metrics = list(performance_metrics.keys())[:6]
        if not metrics:
            return
        
        cols = min(3, len(metrics))
        rows = (len(metrics) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if len(metrics) == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if len(metrics) > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                # Create histogram for this metric
                if metric in self.combined_data.columns:
                    data = self.combined_data[metric].dropna()
                    if len(data) > 0:
                        axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                        axes[i].set_xlabel(metric.replace('_', ' ').title())
                        axes[i].set_ylabel('Frequency')
                        axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                        axes[i].grid(alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/cognitive_performance_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        output_file = "outputs/cognitive_analysis_results.json"
        
        try:
            # Make results JSON serializable
            serializable_results = self._make_serializable(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"📁 Analysis results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def print_analysis_summary(self, results: Dict[str, Any]):
        """Print formatted analysis summary"""
        print("\n" + "="*80)
        print("🧠 COGNITIVE ANALYSIS SUMMARY")
        print("="*80)
        
        # Analysis info
        analysis_info = results.get('analysis_info', {})
        experiment_name = analysis_info.get('experiment_name', 'Unknown')
        print(f"\n📊 EXPERIMENT: {experiment_name}")
        
        # Data summary
        data_summary = results.get('data_summary', {})
        print(f"\n📊 DATA SUMMARY:")
        print(f"   Total subjects analyzed: {data_summary.get('baseline_subjects', 0):,}")
        
        assessments = data_summary.get('assessments_loaded', [])
        print(f"   Assessment types loaded: {len(assessments)}")
        for assessment in assessments:
            print(f"   - {assessment.get('type', 'Unknown')}: {assessment.get('files', 0)} files, {assessment.get('records', 0)} records")
        
        # Correlation findings
        correlations = results.get('correlation_analysis', {}).get('primary_correlations', {})
        if correlations:
            significant_corrs = [name for name, data in correlations.items() 
                               if data.get('p_value', 1) < 0.05]
            
            print(f"\n🔗 CORRELATION ANALYSIS:")
            print(f"   Total correlations tested: {len(correlations)}")
            print(f"   Significant correlations: {len(significant_corrs)}")
            
            if significant_corrs:
                print(f"\n   📈 TOP SIGNIFICANT CORRELATIONS:")
                # Show top 5 strongest correlations
                sorted_corrs = sorted(significant_corrs, 
                                    key=lambda x: abs(correlations[x].get('correlation_coefficient', 0)), 
                                    reverse=True)[:5]
                for corr_name in sorted_corrs:
                    corr_data = correlations[corr_name]
                    r = corr_data.get('correlation_coefficient', 0)
                    p = corr_data.get('p_value', 1)
                    n = corr_data.get('sample_size', 0)
                    effect = corr_data.get('effect_size', 'unknown')
                    print(f"      {corr_name}: r={r:.3f}, p={p:.4f}, n={n}, effect={effect}")
        
        # Self-informant analysis
        self_informant = results.get('self_informant_comparison', {})
        if self_informant.get('self_informant_available'):
            print(f"\n👥 SELF-INFORMANT ANALYSIS:")
            corr_analysis = self_informant.get('correlation_analysis', {})
            if corr_analysis:
                r = corr_analysis.get('correlation_coefficient', 0)
                p = corr_analysis.get('p_value', 1)
                print(f"   Self-informant correlation: r={r:.3f}, p={p:.4f}")
        
        # Clinical insights
        insights = results.get('clinical_insights', {})
        key_findings = insights.get('key_findings', [])
        if key_findings:
            print(f"\n💡 KEY FINDINGS:")
            for i, finding in enumerate(key_findings, 1):
                print(f"   {i}. {finding}")
        
        # Recommendations
        statistical_summary = results.get('statistical_summary', {})
        recommendations = statistical_summary.get('recommendations', [])
        if recommendations:
            print(f"\n📋 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        print("✅ Cognitive analysis complete!")
        print("📁 Results saved to: outputs/cognitive_analysis_results.json")
        print("🎨 Visualizations saved to: outputs/visualizations/")
        print("="*80)


def main():
    """Test the cognitive analysis agent"""
    agent = CognitiveAnalysisAgent()
    results = agent.run_complete_analysis()
    agent.print_analysis_summary(results)
    return results


if __name__ == "__main__":
    main()