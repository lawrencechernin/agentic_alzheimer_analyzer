#!/usr/bin/env python3
"""
Population Bias Detection for Cognitive Datasets
Based on learnings from BHR analysis showing extreme selection bias effects
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

def assess_population_bias(df: pd.DataFrame, 
                          age_col: str = 'Age_Baseline',
                          edu_col: str = 'YearsEducationUS_Converted') -> Dict:
    """
    Assess population selection bias in cognitive datasets.
    
    Returns dict with bias indicators and expected adjustments.
    """
    results = {
        'has_bias': False,
        'bias_severity': 'low',
        'expected_auc_adjustment': 0.0,
        'warnings': []
    }
    
    # Check age distribution
    if age_col in df.columns:
        age = pd.to_numeric(df[age_col], errors='coerce')
        age_clean = age.dropna()
        
        if len(age_clean) > 0:
            pct_over_65 = (age_clean >= 65).mean() * 100
            pct_over_75 = (age_clean >= 75).mean() * 100
            
            # Expected: 65+ ~20%, 75+ ~7% in general population
            if pct_over_65 > 30:  # Over-representation of elderly
                results['warnings'].append(f"High elderly representation: {pct_over_65:.1f}% are 65+")
            elif pct_over_65 < 10:  # Under-representation
                results['warnings'].append(f"Low elderly representation: {pct_over_65:.1f}% are 65+ (expect ~20%)")
                results['has_bias'] = True
    
    # Check education distribution  
    if edu_col in df.columns:
        edu = pd.to_numeric(df[edu_col], errors='coerce')
        edu_clean = edu.dropna()
        
        if len(edu_clean) > 0:
            pct_college = (edu_clean >= 16).mean() * 100
            pct_graduate = (edu_clean >= 17).mean() * 100
            
            # Expected: college+ ~32%, graduate ~12% in US
            if pct_college > 60:  # Extreme education bias like BHR
                results['warnings'].append(f"Extreme education bias: {pct_college:.1f}% have college+ (expect ~32%)")
                results['has_bias'] = True
                results['bias_severity'] = 'high'
                
                # Based on BHR analysis: 0.798 in biased → ~0.85 in clinical
                results['expected_auc_adjustment'] = 0.05 + (pct_college - 32) * 0.001
    
    return results


def detect_cognitive_reserve_effects(df: pd.DataFrame,
                                    outcome_col: str,
                                    performance_cols: list,
                                    edu_col: str = 'YearsEducationUS_Converted') -> Dict:
    """
    Detect cognitive reserve masking effects.
    
    Returns indicators of discordance between self-report and performance.
    """
    results = {
        'reserve_effect_detected': False,
        'over_reporters_pct': 0.0,
        'hidden_impairment_pct': 0.0
    }
    
    if edu_col not in df.columns or outcome_col not in df.columns:
        return results
    
    edu = pd.to_numeric(df[edu_col], errors='coerce')
    high_edu = edu >= 16
    low_edu = edu < 12
    
    # Check for education-performance discordance
    if len(performance_cols) > 0 and any(col in df.columns for col in performance_cols):
        # Create composite performance score
        perf_data = df[performance_cols].select_dtypes(include=[np.number])
        if perf_data.shape[1] > 0:
            # Lower performance = higher impairment
            perf_composite = perf_data.mean(axis=1)
            perf_impaired = perf_composite > perf_composite.quantile(0.75)
            
            outcome = df[outcome_col].astype(bool)
            
            # Over-reporters: report impairment but perform well
            over_reporters = outcome & ~perf_impaired
            if high_edu.sum() > 10:
                over_report_rate = over_reporters[high_edu].mean() * 100
                results['over_reporters_pct'] = over_report_rate
                
                if over_report_rate > 5:
                    results['reserve_effect_detected'] = True
            
            # Hidden impairment: perform poorly but don't report
            hidden = perf_impaired & ~outcome
            if low_edu.sum() > 10:
                hidden_rate = hidden[low_edu].mean() * 100
                results['hidden_impairment_pct'] = hidden_rate
    
    return results


def calculate_expected_mci_prevalence(df: pd.DataFrame, 
                                     age_col: str = 'Age_Baseline') -> Tuple[float, float]:
    """
    Calculate expected MCI prevalence based on age distribution.
    
    Returns (expected_prevalence, observed_prevalence)
    """
    # Age-specific MCI prevalence from literature
    age_prevalence = {
        (45, 55): 0.03,
        (55, 65): 0.067,
        (65, 75): 0.131,
        (75, 85): 0.207,
        (85, 120): 0.376
    }
    
    if age_col not in df.columns:
        return (0.10, np.nan)  # Default 10% if no age data
    
    age = pd.to_numeric(df[age_col], errors='coerce')
    total_expected = 0
    total_count = 0
    
    for (min_age, max_age), prevalence in age_prevalence.items():
        mask = (age >= min_age) & (age < max_age)
        count = mask.sum()
        total_expected += count * prevalence
        total_count += count
    
    if total_count > 0:
        expected = total_expected / total_count
    else:
        expected = 0.10
    
    # Get observed if MCI column exists
    observed = np.nan
    for col in ['MCI', 'AnyCogImpairment', 'MedHx_MCI']:
        if col in df.columns:
            observed = df[col].mean()
            break
    
    return (expected, observed)


def adjust_performance_for_bias(auc: float, bias_assessment: Dict) -> Dict:
    """
    Adjust model performance expectations based on population bias.
    
    Returns adjusted metrics and interpretation.
    """
    results = {
        'reported_auc': auc,
        'estimated_clinical_auc': auc,
        'interpretation': '',
        'confidence': 'moderate'
    }
    
    if bias_assessment.get('has_bias'):
        adjustment = bias_assessment.get('expected_auc_adjustment', 0)
        results['estimated_clinical_auc'] = min(auc + adjustment, 0.99)
        
        if bias_assessment.get('bias_severity') == 'high':
            results['interpretation'] = (
                f"Due to extreme selection bias (highly educated cohort), "
                f"this model would likely achieve AUC ~{results['estimated_clinical_auc']:.2f} "
                f"in typical clinical populations"
            )
            results['confidence'] = 'high' if adjustment > 0.03 else 'moderate'
        else:
            results['interpretation'] = (
                f"Moderate selection bias detected. "
                f"Real-world performance may be slightly higher."
            )
    else:
        results['interpretation'] = "Population appears representative"
    
    return results 