#!/usr/bin/env python3
"""
BHR MemTrax Best Snapshot - AUC 0.798 Configuration
====================================================
This script captures the exact configuration that achieved AUC=0.798,
just below the clinical threshold of 0.80.

Key Components:
- Composite cognitive impairment target (OR of multiple QIDs)
- Ashford filtering with accuracy >= 0.65
- RT winsorization [0.4, 2.0]
- Advanced sequence features (fatigue, reliability, slopes)
- Demographics with interactions and splines
- ECOG/SP/ADL residuals (global and per-domain)
- Stacked ensemble (Logistic + HistGB + XGBoost)
- Calibration and threshold optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Document the winning configuration
WINNING_CONFIG = {
    "model": "StackingClassifier",
    "base_models": [
        "LogisticRegression (MI k=15)",
        "HistGradientBoosting (lr=0.1, leaves=31)",
        "XGBClassifier (scale_pos_weight)"
    ],
    "preprocessing": {
        "ashford_accuracy_threshold": 0.65,
        "rt_winsorization": [0.4, 2.0],
        "quality_filters": ["Status == 'Collected'", "0.5 <= RT <= 2.5"]
    },
    "features": {
        "memtrax_core": [
            "CognitiveScore_mean", "CorrectPCT_mean", "CorrectResponsesRT_mean",
            "CorrectPCT_std", "CorrectResponsesRT_std", "CorrectResponsesRT_cv"
        ],
        "sequence_features": [
            "seq_fatigue_effect", "seq_first_third_mean", "seq_last_third_mean",
            "long_reliability_change", "long_n_timepoints", "long_rt_slope"
        ],
        "demographics": [
            "Age_Baseline", "YearsEducationUS_Converted", "Gender_Numeric",
            "Age_squared", "Education_squared", "Age_Education_interaction",
            "CognitiveReserve_Proxy", "age_rt_interaction", "age_variability_interaction"
        ],
        "ecog_residuals": [
            "ECOG_mean", "ECOG_Memory_mean", "ECOG_Language_mean",
            "ECOG_Visuospatial_mean", "ECOG_Executive_mean",
            "SP_ECOG_mean", "SP_ADL_mean"
        ],
        "splines": ["Age and Education natural cubic splines (3 knots)"]
    },
    "target": {
        "type": "composite",
        "description": "Any Cognitive Impairment (OR of QIDs: 3, 5, 36, 216, 250)",
        "prevalence": "~11-12%"
    },
    "results": {
        "AUC": 0.798,
        "PR_AUC": 0.249,
        "demographics_only_AUC": 0.620,
        "incremental_value": 0.178
    }
}

def save_configuration():
    """Save the winning configuration with timestamp"""
    config_path = Path("bhr_memtrax_results") / "best_config_snapshot.json"
    config_path.parent.mkdir(exist_ok=True)
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "configuration": WINNING_CONFIG,
        "notes": [
            "This configuration achieved AUC=0.798, just 0.002 below clinical threshold",
            "Key success factors: composite target, ECOG residuals, stacking ensemble",
            "Winsorization and quality filtering were critical for noise reduction",
            "Spline features captured non-linear age/education relationships"
        ]
    }
    
    with open(config_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"📸 Configuration snapshot saved to: {config_path}")
    print(f"🎯 AUC: {WINNING_CONFIG['results']['AUC']}")
    print(f"📊 Incremental value over demographics: +{WINNING_CONFIG['results']['incremental_value']:.3f}")
    
    return config_path

if __name__ == "__main__":
    save_configuration()
    print("\n" + "="*60)
    print("WINNING CONFIGURATION SUMMARY")
    print("="*60)
    for category, details in WINNING_CONFIG.items():
        print(f"\n{category.upper()}:")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")
        elif isinstance(details, list):
            for item in details:
                print(f"  - {item}")
        else:
            print(f"  {details}") 