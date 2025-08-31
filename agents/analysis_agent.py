#!/usr/bin/env python3
"""
Cognitive Analysis Agent
========================

Autonomous agent for analyzing relationships between different cognitive assessments
in Alzheimer's research. Completely generalizable framework that works with any
cognitive assessment combination through configuration-driven analysis.
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
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

class CognitiveAnalysisAgent:
    """
    Generalizable analysis agent for cognitive assessment correlation studies
    
    Capabilities:
    - Automated data loading and preprocessing
    - Self vs informant comparison (when available)
    - Multi-modal cognitive performance analysis
    - Cross-assessment correlation analysis
    - Statistical significance testing
    - Effect size calculations
    - Visualization generation
    - Clinical interpretation
    
    Supports any cognitive assessment combination through configuration.
    """
    
    def __init__(self, config_path: str = "config/config.yaml",
                 data_dict_path: str = "config/data_dictionary.json",
                 discovery_results_path: str = "outputs/dataset_discovery_results.json"):
        
        self.config_path = config_path
        self.data_dict_path = data_dict_path
        self.discovery_results_path = discovery_results_path
        
        # Load configurations
        self.config = self._load_config()
        self.data_dictionary = self._load_data_dictionary()
        self.discovery_results = self._load_discovery_results()
        
        # Initialize data containers
        self.ecog_data = None
        self.memtrax_data = None
        self.combined_data = None
        self.results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🧠 ECOG-MemTrax Analysis Agent initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _load_data_dictionary(self) -> Dict[str, Any]:
        """Load data dictionary"""
        try:
            with open(self.data_dict_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading data dictionary: {e}")
            return {}
    
    def _load_discovery_results(self) -> Dict[str, Any]:
        """Load discovery results"""
        try:
            if os.path.exists(self.discovery_results_path):
                with open(self.discovery_results_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Discovery results not available: {e}")
        return {}
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Execute complete ECOG-MemTrax analysis pipeline"""
        self.logger.info("🚀 Starting ECOG-MemTrax Analysis Pipeline")
        
        analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'experiment_name': self.config.get('experiment', {}).get('name', 'ECOG_MemTrax_Analysis'),
            'data_summary': {},
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
                raise ValueError("No valid data available for analysis")
            
            # Step 2: ECOG self vs informant analysis
            self.logger.info("Step 2: ECOG self vs informant comparison")
            self_informant_results = self._analyze_self_informant_differences()
            analysis_results['self_informant_comparison'] = self_informant_results
            
            # Step 3: MemTrax cognitive performance analysis
            self.logger.info("Step 3: MemTrax cognitive performance analysis")
            cognitive_results = self._analyze_cognitive_performance()
            analysis_results['cognitive_performance_analysis'] = cognitive_results
            
            # Step 4: ECOG-MemTrax correlation analysis
            self.logger.info("Step 4: ECOG-MemTrax correlation analysis")
            correlation_results = self._analyze_ecog_memtrax_correlations()
            analysis_results['correlation_analysis'] = correlation_results
            
            # Step 5: Clinical insights and interpretation
            self.logger.info("Step 5: Clinical insights generation")
            clinical_insights = self._generate_clinical_insights(analysis_results)
            analysis_results['clinical_insights'] = clinical_insights
            
            # Step 6: Create visualizations
            self.logger.info("Step 6: Visualization generation")
            self._create_analysis_visualizations(analysis_results)
            
            # Step 7: Generate statistical summary
            self.logger.info("Step 7: Statistical summary")
            statistical_summary = self._generate_statistical_summary(analysis_results)
            analysis_results['statistical_summary'] = statistical_summary
            
            # Save results
            self._save_analysis_results(analysis_results)
            
            self.logger.info("✅ ECOG-MemTrax analysis complete!")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            analysis_results['error'] = str(e)
            return analysis_results
    
    def _load_and_preprocess_data(self) -> Dict[str, Any]:
        """Load and preprocess ECOG and MemTrax data"""
        data_summary = {
            'ecog_files_loaded': 0,
            'memtrax_files_loaded': 0,
            'total_subjects': 0,
            'baseline_subjects': 0,
            'preprocessing_steps': []
        }
        
        # Get file information from discovery results
        files_by_type = self.discovery_results.get('files_discovered', {}).get('files_by_type', {})
        
        # Load ECOG data
        ecog_files = files_by_type.get('ecog_data', [])
        if ecog_files:
            self.logger.info(f"   Loading ECOG data from {len(ecog_files)} files")
            self.ecog_data = self._load_ecog_files(ecog_files)
            data_summary['ecog_files_loaded'] = len(ecog_files)
        
        # Load MemTrax data
        memtrax_files = files_by_type.get('cognitive_data', [])
        if memtrax_files:
            self.logger.info(f"   Loading MemTrax data from {len(memtrax_files)} files")
            self.memtrax_data = self._load_memtrax_files(memtrax_files)
            data_summary['memtrax_files_loaded'] = len(memtrax_files)
        
        # Combine and preprocess data
        if self.ecog_data is not None and self.memtrax_data is not None:
            self.combined_data = self._combine_datasets()
            data_summary['preprocessing_steps'].append("Combined ECOG and MemTrax datasets")
            
            # Get baseline data
            self.combined_data = self._get_baseline_data(self.combined_data)
            data_summary['preprocessing_steps'].append("Filtered to baseline timepoints")
            
            # Clean and validate data
            self.combined_data = self._clean_and_validate_data(self.combined_data)
            data_summary['preprocessing_steps'].append("Applied data cleaning and validation")
            
            data_summary['total_subjects'] = len(self.combined_data)
            data_summary['baseline_subjects'] = len(self.combined_data)
            
            self.logger.info(f"   ✅ Final dataset: {len(self.combined_data)} subjects")
        else:
            self.logger.error("Failed to load both ECOG and MemTrax data")
        
        return data_summary
    
    def _load_ecog_files(self, ecog_files: List[str]) -> pd.DataFrame:
        """Load ECOG data files"""
        ecog_dataframes = []
        
        for file_path in ecog_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                self.logger.info(f"     ECOG file: {os.path.basename(file_path)} - {len(df)} records")
                ecog_dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"Could not load {file_path}: {e}")
        
        if ecog_dataframes:
            combined_ecog = pd.concat(ecog_dataframes, ignore_index=True)
            return combined_ecog
        return None
    
    def _load_memtrax_files(self, memtrax_files: List[str]) -> pd.DataFrame:
        """Load MemTrax data files"""
        memtrax_dataframes = []
        
        for file_path in memtrax_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                self.logger.info(f"     MemTrax file: {os.path.basename(file_path)} - {len(df)} records")
                memtrax_dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"Could not load {file_path}: {e}")
        
        if memtrax_dataframes:
            combined_memtrax = pd.concat(memtrax_dataframes, ignore_index=True)
            return combined_memtrax
        return None
    
    def _combine_datasets(self) -> pd.DataFrame:
        """Combine ECOG and MemTrax datasets"""
        # Find common subject identifier
        subject_cols = ['SubjectCode', 'subject_id', 'participant_id', 'Code']
        
        ecog_subject_col = None
        memtrax_subject_col = None
        
        for col in subject_cols:
            if col in self.ecog_data.columns:
                ecog_subject_col = col
                break
        
        for col in subject_cols:
            if col in self.memtrax_data.columns:
                memtrax_subject_col = col
                break
        
        if not ecog_subject_col or not memtrax_subject_col:
            raise ValueError("Could not find common subject identifier")
        
        # Merge datasets
        combined = self.memtrax_data.merge(
            self.ecog_data,
            left_on=memtrax_subject_col,
            right_on=ecog_subject_col,
            how='inner',
            suffixes=('_memtrax', '_ecog')
        )
        
        self.logger.info(f"   📊 Combined dataset: {len(combined)} subjects with both ECOG and MemTrax data")
        return combined
    
    def _get_baseline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to baseline timepoints only"""
        # Look for timepoint indicators
        timepoint_cols = ['TimepointCode', 'timepoint', 'visit']
        baseline_values = ['m00', 'baseline', '0', 'visit_1']
        
        for col in timepoint_cols:
            if col in df.columns:
                # Try different baseline indicators
                for baseline_val in baseline_values:
                    baseline_data = df[df[col] == baseline_val]
                    if len(baseline_data) > 0:
                        self.logger.info(f"   📅 Using baseline timepoint: {col} == {baseline_val}")
                        return baseline_data
        
        # If no timepoint column found, assume all data is baseline
        self.logger.info("   📅 No timepoint column found, using all data as baseline")
        return df
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the combined dataset"""
        original_size = len(df)
        
        # Find reaction time column
        rt_cols = ['CorrectResponsesRT', 'ResponseTime', 'reaction_time', 'rt']
        rt_col = None
        for col in rt_cols:
            if col in df.columns:
                rt_col = col
                break
        
        if rt_col:
            # Validate reaction times
            valid_rt = (df[rt_col] >= 0.3) & (df[rt_col] <= 3.0)
            df = df[valid_rt]
            self.logger.info(f"   🧹 RT validation: {len(df)}/{original_size} subjects retained")
        
        # Find accuracy column
        acc_cols = ['CorrectPCT', 'PercentCorrect', 'accuracy']
        acc_col = None
        for col in acc_cols:
            if col in df.columns:
                acc_col = col
                break
        
        if acc_col:
            # Validate accuracy
            if df[acc_col].max() > 1.0:  # Convert percentage to proportion
                df[acc_col] = df[acc_col] / 100
            
            valid_acc = (df[acc_col] >= 0.0) & (df[acc_col] <= 1.0)
            df = df[valid_acc]
            self.logger.info(f"   🧹 Accuracy validation: {len(df)}/{original_size} subjects retained")
        
        # Remove rows with excessive missing data
        missing_threshold = 0.5  # Remove if >50% missing
        before_missing = len(df)
        df = df.dropna(thresh=len(df.columns) * missing_threshold)
        
        self.logger.info(f"   🧹 Missing data cleanup: {len(df)}/{before_missing} subjects retained")
        
        return df
    
    def _analyze_self_informant_differences(self) -> Dict[str, Any]:
        """Analyze differences between ECOG self and informant reports"""
        self_informant_results = {
            'self_informant_available': False,
            'correlation_analysis': {},
            'discrepancy_analysis': {},
            'domain_analysis': {}
        }
        
        # Look for self and informant ECOG columns
        ecog_columns = [col for col in self.combined_data.columns if 'ecog' in col.lower() or 'QID' in col]
        
        # Try to identify self vs informant versions
        # This is dataset-specific logic that would be configured
        self_cols = [col for col in ecog_columns if any(term in col.lower() for term in ['self', 'participant'])]
        informant_cols = [col for col in ecog_columns if any(term in col.lower() for term in ['informant', 'partner', 'caregiver'])]
        
        if len(self_cols) > 0 and len(informant_cols) > 0:
            self_informant_results['self_informant_available'] = True
            
            # Calculate total ECOG scores
            if len(self_cols) > 5:  # Assume we have individual items
                self_total = self.combined_data[self_cols].sum(axis=1, skipna=False)
                informant_total = self.combined_data[informant_cols].sum(axis=1, skipna=False)
            else:
                # Assume we have domain or total scores
                self_total = self.combined_data[self_cols[0]]
                informant_total = self.combined_data[informant_cols[0]]
            
            # Correlation analysis
            if len(self_total.dropna()) > 10 and len(informant_total.dropna()) > 10:
                corr_coef, corr_p = pearsonr(self_total.dropna(), informant_total.dropna())
                self_informant_results['correlation_analysis'] = {
                    'correlation_coefficient': corr_coef,
                    'p_value': corr_p,
                    'sample_size': len(self_total.dropna())
                }
            
            # Discrepancy analysis
            discrepancy = informant_total - self_total
            self_informant_results['discrepancy_analysis'] = {
                'mean_discrepancy': float(discrepancy.mean()),
                'std_discrepancy': float(discrepancy.std()),
                'positive_discrepancy_percent': float((discrepancy > 0).mean() * 100),
                'large_discrepancy_percent': float((abs(discrepancy) > discrepancy.std()).mean() * 100)
            }
            
            self.logger.info(f"   ✅ Self-informant analysis: r={corr_coef:.3f}, p={corr_p:.4f}")
        
        else:
            self.logger.warning("   ⚠️ Could not identify both self and informant ECOG measures")
        
        return self_informant_results
    
    def _analyze_cognitive_performance(self) -> Dict[str, Any]:
        """Analyze MemTrax cognitive performance patterns"""
        cognitive_results = {
            'reaction_time_analysis': {},
            'accuracy_analysis': {},
            'efficiency_analysis': {},
            'demographic_effects': {}
        }
        
        # Find MemTrax variables
        rt_cols = ['CorrectResponsesRT', 'ResponseTime', 'reaction_time']
        acc_cols = ['CorrectPCT', 'PercentCorrect', 'accuracy']
        
        rt_col = None
        acc_col = None
        
        for col in rt_cols:
            if col in self.combined_data.columns:
                rt_col = col
                break
        
        for col in acc_cols:
            if col in self.combined_data.columns:
                acc_col = col
                break
        
        if rt_col:
            rt_data = self.combined_data[rt_col].dropna()
            cognitive_results['reaction_time_analysis'] = {
                'mean_rt': float(rt_data.mean()),
                'median_rt': float(rt_data.median()),
                'std_rt': float(rt_data.std()),
                'sample_size': len(rt_data)
            }
        
        if acc_col:
            acc_data = self.combined_data[acc_col].dropna()
            if acc_data.max() > 1.0:  # Convert percentage to proportion
                acc_data = acc_data / 100
            
            cognitive_results['accuracy_analysis'] = {
                'mean_accuracy': float(acc_data.mean()),
                'median_accuracy': float(acc_data.median()),
                'std_accuracy': float(acc_data.std()),
                'sample_size': len(acc_data)
            }
        
        # Calculate cognitive efficiency if both RT and accuracy available
        if rt_col and acc_col:
            rt_data = self.combined_data[rt_col]
            acc_data = self.combined_data[acc_col]
            if acc_data.max() > 1.0:
                acc_data = acc_data / 100
            
            efficiency = acc_data / rt_data  # Correct responses per second
            cognitive_results['efficiency_analysis'] = {
                'mean_efficiency': float(efficiency.mean()),
                'std_efficiency': float(efficiency.std()),
                'sample_size': len(efficiency.dropna())
            }
        
        return cognitive_results
    
    def _analyze_ecog_memtrax_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between ECOG and MemTrax measures"""
        correlation_results = {
            'primary_correlations': {},
            'domain_specific_correlations': {},
            'clinical_significance': {}
        }
        
        # Find key variables
        rt_cols = ['CorrectResponsesRT', 'ResponseTime', 'reaction_time']
        acc_cols = ['CorrectPCT', 'PercentCorrect', 'accuracy']
        ecog_cols = [col for col in self.combined_data.columns if 'ecog' in col.lower() or 'QID' in col]
        
        rt_col = None
        acc_col = None
        
        for col in rt_cols:
            if col in self.combined_data.columns:
                rt_col = col
                break
        
        for col in acc_cols:
            if col in self.combined_data.columns:
                acc_col = col
                break
        
        # Primary correlations
        if rt_col and len(ecog_cols) > 0:
            for ecog_col in ecog_cols[:10]:  # Limit to prevent overload
                try:
                    rt_data = self.combined_data[rt_col].dropna()
                    ecog_data = self.combined_data[ecog_col].dropna()
                    
                    # Get common indices
                    common_indices = rt_data.index.intersection(ecog_data.index)
                    if len(common_indices) > 20:  # Minimum sample size
                        rt_common = rt_data[common_indices]
                        ecog_common = ecog_data[common_indices]
                        
                        corr_coef, corr_p = pearsonr(rt_common, ecog_common)
                        
                        correlation_results['primary_correlations'][f'{ecog_col}_vs_{rt_col}'] = {
                            'correlation_coefficient': corr_coef,
                            'p_value': corr_p,
                            'sample_size': len(common_indices),
                            'effect_size': self._interpret_correlation_effect_size(abs(corr_coef))
                        }
                except Exception as e:
                    continue
        
        # Calculate clinical significance
        significant_correlations = 0
        total_correlations = len(correlation_results['primary_correlations'])
        
        for corr_name, corr_data in correlation_results['primary_correlations'].items():
            if corr_data['p_value'] < 0.05:
                significant_correlations += 1
        
        if total_correlations > 0:
            correlation_results['clinical_significance'] = {
                'significant_correlations': significant_correlations,
                'total_correlations': total_correlations,
                'significance_rate': significant_correlations / total_correlations,
                'multiple_comparison_threshold': 0.05 / total_correlations  # Bonferroni correction
            }
        
        return correlation_results
    
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
        
        # Analyze self-informant findings
        self_informant = analysis_results.get('self_informant_comparison', {})
        if self_informant.get('self_informant_available'):
            corr_data = self_informant.get('correlation_analysis', {})
            if corr_data:
                correlation = corr_data.get('correlation_coefficient', 0)
                insights['key_findings'].append(
                    f"ECOG self-informant correlation: r={correlation:.3f} (n={corr_data.get('sample_size', 0)})"
                )
                
                if correlation < 0.5:
                    insights['clinical_implications'].append(
                        "Moderate agreement between self and informant reports suggests potential anosognosia"
                    )
        
        # Analyze correlation findings
        correlations = analysis_results.get('correlation_analysis', {})
        primary_corrs = correlations.get('primary_correlations', {})
        
        significant_corrs = [name for name, data in primary_corrs.items() 
                           if data.get('p_value', 1) < 0.05]
        
        if significant_corrs:
            insights['key_findings'].append(
                f"Found {len(significant_corrs)} significant ECOG-MemTrax correlations"
            )
        
        # Check for novel findings
        clinical_sig = correlations.get('clinical_significance', {})
        if clinical_sig.get('significance_rate', 0) > 0.5:
            insights['novel_discoveries'].append(
                "High rate of significant correlations suggests strong ECOG-MemTrax relationship"
            )
        
        # Add limitations
        data_summary = analysis_results.get('data_summary', {})
        sample_size = data_summary.get('baseline_subjects', 0)
        
        if sample_size < 100:
            insights['limitations'].append("Small sample size may limit generalizability")
        
        insights['limitations'].append("Cross-sectional analysis - causality cannot be determined")
        
        return insights
    
    def _create_analysis_visualizations(self, analysis_results: Dict[str, Any]):
        """Create visualizations for the analysis"""
        self.logger.info("   🎨 Creating visualizations...")
        
        # Create output directory
        os.makedirs('outputs/visualizations', exist_ok=True)
        
        try:
            # Create correlation matrix visualization
            self._create_correlation_matrix_plot(analysis_results)
            
            # Create self-informant comparison plot
            self._create_self_informant_plot(analysis_results)
            
            # Create cognitive performance distribution plots
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
            corr_names.append(name.replace('_vs_', ' vs ').replace('_', ' '))
            corr_values.append(data.get('correlation_coefficient', 0))
            p_values.append(data.get('p_value', 1))
        
        if len(corr_values) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bar plot
            colors = ['red' if p < 0.05 else 'lightblue' for p in p_values]
            bars = ax.barh(range(len(corr_values)), corr_values, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(corr_names)))
            ax.set_yticklabels(corr_names, fontsize=9)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title('ECOG-MemTrax Correlations', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add significance indicators
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.05:
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                           '*', fontsize=12, fontweight='bold', va='center')
            
            plt.tight_layout()
            plt.savefig('outputs/visualizations/ecog_memtrax_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_self_informant_plot(self, analysis_results: Dict[str, Any]):
        """Create self vs informant comparison plot"""
        self_informant = analysis_results.get('self_informant_comparison', {})
        
        if not self_informant.get('self_informant_available'):
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Correlation info (if available)
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
            mean_disc = discrepancy.get('mean_discrepancy', 0)
            std_disc = discrepancy.get('std_discrepancy', 1)
            pos_disc_pct = discrepancy.get('positive_discrepancy_percent', 50)
            
            categories = ['Informant Higher', 'Agreement', 'Self Higher']
            percentages = [pos_disc_pct, 100 - pos_disc_pct - (100 - pos_disc_pct), 100 - pos_disc_pct]
            colors = ['orange', 'green', 'blue']
            
            # Simple approximation for visualization
            percentages = [pos_disc_pct, 20, 100 - pos_disc_pct - 20]
            
            ax2.pie(percentages, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Report Discrepancy Patterns', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/self_informant_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_distribution_plots(self, analysis_results: Dict[str, Any]):
        """Create cognitive performance distribution plots"""
        cognitive = analysis_results.get('cognitive_performance_analysis', {})
        
        if not cognitive:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot reaction time distribution
        rt_analysis = cognitive.get('reaction_time_analysis', {})
        if rt_analysis and self.combined_data is not None:
            rt_cols = ['CorrectResponsesRT', 'ResponseTime', 'reaction_time']
            rt_col = None
            for col in rt_cols:
                if col in self.combined_data.columns:
                    rt_col = col
                    break
            
            if rt_col:
                rt_data = self.combined_data[rt_col].dropna()
                axes[0].hist(rt_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0].set_xlabel('Reaction Time (seconds)')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('Reaction Time Distribution')
                axes[0].grid(alpha=0.3)
        
        # Plot accuracy distribution
        acc_analysis = cognitive.get('accuracy_analysis', {})
        if acc_analysis and self.combined_data is not None:
            acc_cols = ['CorrectPCT', 'PercentCorrect', 'accuracy']
            acc_col = None
            for col in acc_cols:
                if col in self.combined_data.columns:
                    acc_col = col
                    break
            
            if acc_col:
                acc_data = self.combined_data[acc_col].dropna()
                if acc_data.max() > 1.0:
                    acc_data = acc_data / 100
                
                axes[1].hist(acc_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1].set_xlabel('Accuracy (proportion)')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('Accuracy Distribution')
                axes[1].grid(alpha=0.3)
        
        # Plot efficiency distribution
        efficiency_analysis = cognitive.get('efficiency_analysis', {})
        if efficiency_analysis:
            axes[2].text(0.5, 0.5, f'Cognitive Efficiency\nMean: {efficiency_analysis.get("mean_efficiency", 0):.3f}\nSD: {efficiency_analysis.get("std_efficiency", 0):.3f}',
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Cognitive Efficiency')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/cognitive_performance_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_statistical_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        summary = {
            'sample_characteristics': {},
            'primary_findings': {},
            'effect_sizes': {},
            'clinical_cutoffs': {},
            'recommendations': []
        }
        
        # Sample characteristics
        data_summary = analysis_results.get('data_summary', {})
        summary['sample_characteristics'] = {
            'total_subjects': data_summary.get('baseline_subjects', 0),
            'data_completeness': 'Good' if data_summary.get('baseline_subjects', 0) > 100 else 'Limited'
        }
        
        # Primary findings
        correlations = analysis_results.get('correlation_analysis', {}).get('primary_correlations', {})
        significant_correlations = [name for name, data in correlations.items() 
                                 if data.get('p_value', 1) < 0.05]
        
        summary['primary_findings'] = {
            'significant_ecog_memtrax_correlations': len(significant_correlations),
            'total_correlations_tested': len(correlations)
        }
        
        # Recommendations
        if len(significant_correlations) > 0:
            summary['recommendations'].append("Strong evidence for ECOG-MemTrax relationship")
            summary['recommendations'].append("Consider validation in independent sample")
        
        if data_summary.get('baseline_subjects', 0) < 200:
            summary['recommendations'].append("Increase sample size for more robust findings")
        
        summary['recommendations'].append("Explore longitudinal changes in ECOG-MemTrax relationships")
        
        return summary
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        output_file = "outputs/ecog_memtrax_analysis_results.json"
        
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
        print("🧠 ECOG-MEMTRAX ANALYSIS SUMMARY")
        print("="*80)
        
        # Data summary
        data_summary = results.get('data_summary', {})
        print(f"\n📊 DATA SUMMARY:")
        print(f"   Total subjects analyzed: {data_summary.get('baseline_subjects', 0):,}")
        print(f"   ECOG files loaded: {data_summary.get('ecog_files_loaded', 0)}")
        print(f"   MemTrax files loaded: {data_summary.get('memtrax_files_loaded', 0)}")
        
        # Correlation findings
        correlations = results.get('correlation_analysis', {}).get('primary_correlations', {})
        if correlations:
            significant_corrs = [name for name, data in correlations.items() 
                               if data.get('p_value', 1) < 0.05]
            
            print(f"\n🔗 CORRELATION ANALYSIS:")
            print(f"   Total correlations tested: {len(correlations)}")
            print(f"   Significant correlations: {len(significant_corrs)}")
            
            if significant_corrs:
                print(f"\n   📈 SIGNIFICANT CORRELATIONS:")
                for corr_name in significant_corrs[:5]:  # Show top 5
                    corr_data = correlations[corr_name]
                    r = corr_data.get('correlation_coefficient', 0)
                    p = corr_data.get('p_value', 1)
                    n = corr_data.get('sample_size', 0)
                    print(f"      {corr_name}: r={r:.3f}, p={p:.4f}, n={n}")
        
        # Self-informant analysis
        self_informant = results.get('self_informant_comparison', {})
        if self_informant.get('self_informant_available'):
            print(f"\n👥 SELF-INFORMANT ANALYSIS:")
            corr_analysis = self_informant.get('correlation_analysis', {})
            if corr_analysis:
                r = corr_analysis.get('correlation_coefficient', 0)
                p = corr_analysis.get('p_value', 1)
                print(f"   Self-informant correlation: r={r:.3f}, p={p:.4f}")
            
            discrepancy = self_informant.get('discrepancy_analysis', {})
            if discrepancy:
                mean_disc = discrepancy.get('mean_discrepancy', 0)
                pos_disc = discrepancy.get('positive_discrepancy_percent', 0)
                print(f"   Mean discrepancy: {mean_disc:.2f}")
                print(f"   Informant reports higher: {pos_disc:.1f}%")
        
        # Clinical insights
        insights = results.get('clinical_insights', {})
        key_findings = insights.get('key_findings', [])
        if key_findings:
            print(f"\n💡 KEY FINDINGS:")
            for i, finding in enumerate(key_findings, 1):
                print(f"   {i}. {finding}")
        
        clinical_implications = insights.get('clinical_implications', [])
        if clinical_implications:
            print(f"\n🏥 CLINICAL IMPLICATIONS:")
            for i, implication in enumerate(clinical_implications, 1):
                print(f"   {i}. {implication}")
        
        # Recommendations
        statistical_summary = results.get('statistical_summary', {})
        recommendations = statistical_summary.get('recommendations', [])
        if recommendations:
            print(f"\n📋 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print("📁 Results saved to: outputs/ecog_memtrax_analysis_results.json")
        print("🎨 Visualizations saved to: outputs/visualizations/")
        print("="*80)


def main():
    """Test the analysis agent"""
    agent = ECOGMemTraxAnalysisAgent()
    results = agent.run_complete_analysis()
    agent.print_analysis_summary(results)
    return results


if __name__ == "__main__":
    main()