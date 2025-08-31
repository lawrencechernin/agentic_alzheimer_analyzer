#!/usr/bin/env python3
"""
Data Discovery Agent
====================

Autonomous agent for discovering and characterizing Alzheimer's datasets.
Uses AI to understand data structure, map variables, and identify research opportunities.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import yaml
import re
from collections import defaultdict

class DataDiscoveryAgent:
    """
    Autonomous agent for intelligent dataset discovery and characterization
    
    Capabilities:
    - Auto-detect file structures and formats
    - Map variables using data dictionary
    - Identify Alzheimer's-specific measures
    - Assess data quality and completeness
    - Suggest analysis strategies
    - Generate dataset characterization reports
    """
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 data_dict_path: str = "config/data_dictionary.json"):
        self.config_path = config_path
        self.data_dict_path = data_dict_path
        
        # Setup logging first
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        self.data_dictionary = self._load_data_dictionary()
        
        # Initialize results storage
        self.discovered_files = {}
        self.variable_mappings = {}
        self.data_quality_assessment = {}
        self.research_opportunities = []
        
        self.logger.info("🔍 Data Discovery Agent initialized")
    
    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _load_data_dictionary(self) -> Dict[str, Any]:
        """Load data dictionary for variable mapping"""
        try:
            with open(self.data_dict_path, 'r') as f:
                data_dict = json.load(f)
            return data_dict
        except Exception as e:
            self.logger.error(f"Error loading data dictionary: {e}")
            return {}
    
    def discover_dataset(self) -> Dict[str, Any]:
        """
        Main discovery method - analyzes entire dataset
        
        Returns:
            Comprehensive dataset characterization
        """
        self.logger.info("🚀 Starting autonomous dataset discovery...")
        
        discovery_results = {
            'discovery_timestamp': datetime.now().isoformat(),
            'dataset_info': {},
            'files_discovered': {},
            'variable_mappings': {},
            'data_quality': {},
            'research_opportunities': [],
            'analysis_recommendations': []
        }
        
        # Step 1: Discover files
        self.logger.info("Step 1: File discovery")
        files_info = self._discover_files()
        discovery_results['files_discovered'] = files_info
        
        # Step 2: Analyze file contents
        self.logger.info("Step 2: Content analysis")
        content_analysis = self._analyze_file_contents(files_info)
        discovery_results['dataset_info'] = content_analysis
        
        # Step 3: Map variables to data dictionary
        self.logger.info("Step 3: Variable mapping")
        variable_mappings = self._map_variables(content_analysis)
        discovery_results['variable_mappings'] = variable_mappings
        
        # Step 4: Assess data quality
        self.logger.info("Step 4: Data quality assessment")
        quality_assessment = self._assess_data_quality(files_info, variable_mappings)
        discovery_results['data_quality'] = quality_assessment
        
        # Step 5: Identify research opportunities
        self.logger.info("Step 5: Research opportunity identification")
        research_opportunities = self._identify_research_opportunities(variable_mappings, quality_assessment)
        discovery_results['research_opportunities'] = research_opportunities
        
        # Step 6: Generate analysis recommendations
        self.logger.info("Step 6: Analysis recommendations")
        analysis_recommendations = self._generate_analysis_recommendations(discovery_results)
        discovery_results['analysis_recommendations'] = analysis_recommendations
        
        # Save discovery results
        self._save_discovery_results(discovery_results)
        
        self.logger.info("✅ Dataset discovery complete!")
        return discovery_results
    
    def _discover_files(self) -> Dict[str, Any]:
        """Discover all relevant data files"""
        files_info = {
            'total_files_found': 0,
            'files_by_type': {},
            'file_details': {}
        }
        
        # Get data paths from configuration
        data_paths = self.config.get('dataset', {}).get('data_paths', ['./'])
        file_patterns = self.config.get('dataset', {}).get('file_patterns', {})
        
        all_files = []
        
        # Search each data path
        for data_path in data_paths:
            if os.path.exists(data_path):
                self.logger.info(f"   Searching: {data_path}")
                
                # Find all relevant files
                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        if file.endswith(('.csv', '.xlsx', '.json', '.parquet')):
                            file_path = os.path.join(root, file)
                            all_files.append(file_path)
        
        files_info['total_files_found'] = len(all_files)
        self.logger.info(f"   📁 Found {len(all_files)} data files")
        
        # Categorize files by type using patterns
        for category, patterns in file_patterns.items():
            matching_files = []
            
            for file_path in all_files:
                file_name = os.path.basename(file_path).lower()
                
                for pattern in patterns:
                    # Convert glob pattern to regex
                    regex_pattern = pattern.replace('*', '.*').lower()
                    if re.search(regex_pattern, file_name):
                        matching_files.append(file_path)
                        break
            
            if matching_files:
                files_info['files_by_type'][category] = matching_files
                self.logger.info(f"   📊 {category}: {len(matching_files)} files")
        
        # Get basic file details
        for file_path in all_files[:20]:  # Limit for performance
            try:
                stat = os.stat(file_path)
                files_info['file_details'][file_path] = {
                    'size_mb': round(stat.st_size / (1024*1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'extension': os.path.splitext(file_path)[1]
                }
            except Exception as e:
                self.logger.warning(f"Could not get details for {file_path}: {e}")
        
        return files_info
    
    def _analyze_file_contents(self, files_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contents of discovered files"""
        content_analysis = {
            'files_analyzed': 0,
            'total_subjects': 0,
            'total_records': 0,
            'file_summaries': {},
            'common_variables': [],
            'longitudinal_indicators': []
        }
        
        files_to_analyze = []
        
        # Prioritize files by type for analysis
        priority_types = ['cognitive_data', 'ecog_data', 'demographic_data', 'medical_data']
        
        for file_type in priority_types:
            if file_type in files_info['files_by_type']:
                files_to_analyze.extend(files_info['files_by_type'][file_type][:3])  # Max 3 per type
        
        # Analyze each file
        for file_path in files_to_analyze:
            try:
                self.logger.info(f"   📊 Analyzing: {os.path.basename(file_path)}")
                
                # Read file sample
                if file_path.endswith('.csv'):
                    df_sample = pd.read_csv(file_path, nrows=1000, low_memory=False)  # Sample first 1000 rows
                elif file_path.endswith('.xlsx'):
                    df_sample = pd.read_excel(file_path, nrows=1000)
                else:
                    continue
                
                # Basic file analysis
                file_summary = {
                    'total_rows': len(df_sample),
                    'total_columns': len(df_sample.columns),
                    'columns': list(df_sample.columns),
                    'dtypes': df_sample.dtypes.to_dict(),
                    'missing_data': df_sample.isnull().sum().to_dict(),
                    'sample_data': df_sample.head(3).to_dict()
                }
                
                # Try to count unique subjects
                subject_columns = ['SubjectCode', 'subject_id', 'participant_id', 'ID', 'Code']
                for col in subject_columns:
                    if col in df_sample.columns:
                        unique_subjects = df_sample[col].nunique()
                        file_summary['unique_subjects'] = unique_subjects
                        content_analysis['total_subjects'] = max(content_analysis['total_subjects'], unique_subjects)
                        break
                
                # Check for longitudinal indicators
                longitudinal_indicators = ['TimepointCode', 'timepoint', 'visit', 'DaysAfterBaseline', 'wave']
                for col in longitudinal_indicators:
                    if col in df_sample.columns:
                        unique_timepoints = df_sample[col].nunique()
                        file_summary['longitudinal'] = True
                        file_summary['timepoints'] = unique_timepoints
                        if col not in content_analysis['longitudinal_indicators']:
                            content_analysis['longitudinal_indicators'].append(col)
                        break
                else:
                    file_summary['longitudinal'] = False
                
                content_analysis['file_summaries'][os.path.basename(file_path)] = file_summary
                content_analysis['files_analyzed'] += 1
                content_analysis['total_records'] += len(df_sample)
                
                # Track common variables across files
                for col in df_sample.columns:
                    if col not in content_analysis['common_variables']:
                        content_analysis['common_variables'].append(col)
                
            except Exception as e:
                self.logger.warning(f"Could not analyze {file_path}: {e}")
        
        self.logger.info(f"   ✅ Analyzed {content_analysis['files_analyzed']} files")
        return content_analysis
    
    def _map_variables(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Map discovered variables to data dictionary concepts"""
        variable_mappings = {
            'mapped_variables': {},
            'unmapped_variables': [],
            'confidence_scores': {},
            'alzheimer_specific_vars': {}
        }
        
        # Get all variables from data dictionary
        var_mappings = self.data_dictionary.get('variable_mappings', {})
        
        # Check each discovered variable
        all_variables = content_analysis['common_variables']
        
        for variable in all_variables:
            mapped = False
            best_match = None
            best_confidence = 0
            
            # Search through data dictionary categories
            for category, category_vars in var_mappings.items():
                for concept_name, concept_info in category_vars.items():
                    possible_names = concept_info.get('possible_names', [])
                    
                    # Exact match
                    if variable in possible_names:
                        variable_mappings['mapped_variables'][variable] = {
                            'concept': concept_name,
                            'category': category,
                            'confidence': 1.0,
                            'description': concept_info.get('description', ''),
                            'clinical_significance': concept_info.get('clinical_significance', '')
                        }
                        mapped = True
                        break
                    
                    # Fuzzy match
                    for possible_name in possible_names:
                        similarity = self._calculate_similarity(variable.lower(), possible_name.lower())
                        if similarity > 0.8 and similarity > best_confidence:
                            best_match = {
                                'concept': concept_name,
                                'category': category,
                                'confidence': similarity,
                                'description': concept_info.get('description', ''),
                                'clinical_significance': concept_info.get('clinical_significance', '')
                            }
                            best_confidence = similarity
                
                if mapped:
                    break
            
            # If no exact match, use best fuzzy match
            if not mapped and best_match:
                variable_mappings['mapped_variables'][variable] = best_match
                mapped = True
            
            if not mapped:
                variable_mappings['unmapped_variables'].append(variable)
        
        # Identify Alzheimer's-specific variables
        alzheimer_categories = ['cognitive_assessments', 'ecog_assessments', 'medical_history']
        
        for var, mapping in variable_mappings['mapped_variables'].items():
            if mapping['category'] in alzheimer_categories:
                variable_mappings['alzheimer_specific_vars'][var] = mapping
        
        self.logger.info(f"   ✅ Mapped {len(variable_mappings['mapped_variables'])} variables")
        self.logger.info(f"   🧠 Found {len(variable_mappings['alzheimer_specific_vars'])} Alzheimer's-specific variables")
        
        return variable_mappings
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simple implementation)"""
        # Simple character overlap similarity
        if str1 == str2:
            return 1.0
        
        # Check if one string contains the other
        if str1 in str2 or str2 in str1:
            return 0.9
        
        # Character overlap
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        overlap = len(set1.intersection(set2))
        total_chars = len(set1.union(set2))
        
        return overlap / total_chars if total_chars > 0 else 0.0
    
    def _assess_data_quality(self, files_info: Dict[str, Any], 
                           variable_mappings: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality"""
        quality_assessment = {
            'overall_score': 0,
            'completeness_score': 0,
            'consistency_score': 0,
            'quality_issues': [],
            'recommendations': []
        }
        
        # Check for key variables
        required_vars = ['cognitive_assessments', 'demographics', 'identifiers']
        available_categories = set()
        
        for var, mapping in variable_mappings['mapped_variables'].items():
            available_categories.add(mapping['category'])
        
        missing_categories = set(required_vars) - available_categories
        if missing_categories:
            quality_assessment['quality_issues'].append(f"Missing key categories: {missing_categories}")
        
        # Calculate completeness score
        total_required = len(required_vars)
        available_required = len(set(required_vars).intersection(available_categories))
        quality_assessment['completeness_score'] = available_required / total_required
        
        # Check for longitudinal data
        longitudinal_available = len(quality_assessment.get('longitudinal_indicators', [])) > 0
        if longitudinal_available:
            quality_assessment['recommendations'].append("Longitudinal analysis possible")
        else:
            quality_assessment['recommendations'].append("Cross-sectional analysis only")
        
        # Overall quality score
        quality_assessment['overall_score'] = quality_assessment['completeness_score']
        
        return quality_assessment
    
    def _identify_research_opportunities(self, variable_mappings: Dict[str, Any], 
                                       quality_assessment: Dict[str, Any]) -> List[str]:
        """Identify potential research opportunities"""
        opportunities = []
        
        # Check for ECOG-MemTrax opportunity (main experiment)
        has_ecog = any('ecog' in var.lower() for var in variable_mappings['mapped_variables'])
        has_memtrax = any('memtrax' in var.lower() or 'reaction_time' in mapping['concept'].lower() 
                         for var, mapping in variable_mappings['mapped_variables'].items())
        
        if has_ecog and has_memtrax:
            opportunities.append("ECOG-MemTrax correlation analysis (PRIMARY EXPERIMENT)")
            opportunities.append("Self-report vs informant discrepancy analysis")
            opportunities.append("Multi-modal cognitive assessment validation")
        
        # Check for longitudinal opportunities
        if quality_assessment.get('longitudinal_indicators'):
            opportunities.append("Longitudinal cognitive trajectory modeling")
            opportunities.append("Change-point detection analysis")
            opportunities.append("Progression prediction modeling")
        
        # Check for biomarker opportunities
        medical_vars = [var for var, mapping in variable_mappings['mapped_variables'].items() 
                       if mapping['category'] == 'medical_history']
        
        if len(medical_vars) > 2:
            opportunities.append("Biomarker candidate identification")
            opportunities.append("Risk stratification analysis")
        
        # Check for demographic stratification
        demographic_vars = [var for var, mapping in variable_mappings['mapped_variables'].items() 
                           if mapping['category'] == 'demographics']
        
        if 'age' in [m['concept'] for m in variable_mappings['mapped_variables'].values()]:
            opportunities.append("Age-stratified analysis")
        
        if 'gender' in [m['concept'] for m in variable_mappings['mapped_variables'].values()]:
            opportunities.append("Gender differences in self-awareness")
        
        return opportunities
    
    def _generate_analysis_recommendations(self, discovery_results: Dict[str, Any]) -> List[str]:
        """Generate specific analysis recommendations"""
        recommendations = []
        
        # Based on experiment configuration
        experiment_objectives = self.config.get('experiment', {}).get('primary_objectives', [])
        
        for objective in experiment_objectives:
            if 'ECOG' in objective and 'MemTrax' in objective:
                recommendations.append("Execute ECOG-MemTrax correlation matrix analysis")
                recommendations.append("Compare self vs informant ECOG patterns")
                recommendations.append("Validate cognitive performance relationships")
        
        # Based on data availability
        mapped_vars = discovery_results['variable_mappings']['mapped_variables']
        
        if any('memory' in var.lower() for var in mapped_vars):
            recommendations.append("Focus on memory domain-specific analyses")
        
        if any('executive' in var.lower() for var in mapped_vars):
            recommendations.append("Include executive function domain analysis")
        
        # Statistical recommendations
        recommendations.append("Use Pearson correlations for continuous variables")
        recommendations.append("Apply Bonferroni correction for multiple comparisons")
        recommendations.append("Include effect size calculations (Cohen's d)")
        
        return recommendations
    
    def _save_discovery_results(self, results: Dict[str, Any]):
        """Save discovery results to file"""
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "dataset_discovery_results.json")
        
        try:
            # Convert any non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"📁 Discovery results saved to: {output_file}")
            
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
    
    def print_discovery_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of discovery results"""
        print("\n" + "="*80)
        print("🔍 AUTONOMOUS DATASET DISCOVERY SUMMARY")
        print("="*80)
        
        # Dataset overview
        dataset_info = results.get('dataset_info', {})
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Files analyzed: {dataset_info.get('files_analyzed', 0)}")
        print(f"   Total subjects: {dataset_info.get('total_subjects', 0):,}")
        print(f"   Total records: {dataset_info.get('total_records', 0):,}")
        
        # Variable mapping
        var_mappings = results.get('variable_mappings', {})
        print(f"\n🗺️ VARIABLE MAPPING:")
        print(f"   Mapped variables: {len(var_mappings.get('mapped_variables', {}))}")
        print(f"   Alzheimer's-specific: {len(var_mappings.get('alzheimer_specific_vars', {}))}")
        print(f"   Unmapped variables: {len(var_mappings.get('unmapped_variables', []))}")
        
        # Key variables found
        alzheimer_vars = var_mappings.get('alzheimer_specific_vars', {})
        if alzheimer_vars:
            print(f"\n🧠 KEY ALZHEIMER'S VARIABLES FOUND:")
            for var, mapping in list(alzheimer_vars.items())[:10]:  # Show first 10
                print(f"   {var} → {mapping['concept']} ({mapping['category']})")
        
        # Research opportunities
        opportunities = results.get('research_opportunities', [])
        if opportunities:
            print(f"\n🎯 RESEARCH OPPORTUNITIES IDENTIFIED:")
            for i, opp in enumerate(opportunities, 1):
                print(f"   {i}. {opp}")
        
        # Analysis recommendations
        recommendations = results.get('analysis_recommendations', [])
        if recommendations:
            print(f"\n💡 ANALYSIS RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Data quality
        quality = results.get('data_quality', {})
        print(f"\n📈 DATA QUALITY ASSESSMENT:")
        print(f"   Overall score: {quality.get('overall_score', 0):.2f}/1.0")
        print(f"   Completeness: {quality.get('completeness_score', 0):.2f}/1.0")
        
        if quality.get('quality_issues'):
            print(f"\n⚠️ QUALITY ISSUES:")
            for issue in quality['quality_issues']:
                print(f"   • {issue}")
        
        print("="*80)
        print("✅ Discovery complete - Ready for analysis!")
        print("="*80)


def main():
    """Test the discovery agent"""
    agent = DataDiscoveryAgent()
    results = agent.discover_dataset()
    agent.print_discovery_summary(results)
    return results


if __name__ == "__main__":
    main()