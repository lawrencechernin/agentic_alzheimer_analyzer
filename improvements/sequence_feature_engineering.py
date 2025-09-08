#!/usr/bin/env python3
"""
Sequence Feature Engineering
- Computes sequence/fatigue/reliability features from per-test ReactionTimes strings
- Aggregates to subject level using simple concatenation and robust filtering
"""
from typing import List
import numpy as np
import pandas as pd


def compute_sequence_features(df: pd.DataFrame,
                              subject_col: str = 'SubjectCode',
                              reaction_col: str = 'ReactionTimes') -> pd.DataFrame:
    """Compute subject-level sequence features from ReactionTimes strings.

    Returns a dataframe with columns:
    - seq_first_third_mean, seq_last_third_mean, seq_fatigue_effect
    - seq_mean_rt, seq_median_rt
    - long_reliability_change (std of all RTs)
    - long_n_timepoints (count of tests)
    - long_rt_slope (slope over timepoints index if dates missing)
    """
    features: List[dict] = []

    # Pre-extract list per row for speed
    def parse_rts(rt_str: str) -> List[float]:
        try:
            arr = [float(x.strip()) for x in str(rt_str).split(',') if x.strip()]
            return [v for v in arr if 0.2 < v < 2.5]
        except Exception:
            return []

    df_local = df[[subject_col, reaction_col, 'CorrectResponsesRT']].copy()
    if 'DaysAfterBaseline' in df.columns:
        df_local['DaysAfterBaseline'] = pd.to_numeric(df['DaysAfterBaseline'], errors='coerce')

    for subject, g in df_local.groupby(subject_col):
        all_rts: List[float] = []
        for rt_s in g[reaction_col].tolist():
            all_rts.extend(parse_rts(rt_s))
        seq_first_third_mean = 0.0
        seq_last_third_mean = 0.0
        seq_fatigue_effect = 0.0
        seq_mean_rt = np.nan
        seq_median_rt = np.nan
        long_reliability_change = 0.0
        if len(all_rts) >= 6:
            n = len(all_rts)
            k = max(1, n // 3)
            seq_first_third_mean = float(np.mean(all_rts[:k]))
            seq_last_third_mean = float(np.mean(all_rts[-k:]))
            seq_fatigue_effect = float(seq_last_third_mean - seq_first_third_mean)
            seq_mean_rt = float(np.mean(all_rts))
            seq_median_rt = float(np.median(all_rts))
            if len(all_rts) >= 2:
                long_reliability_change = float(np.std(all_rts))
        long_n_timepoints = int(len(g))
        # slope using time if available else index
        rt_series = pd.to_numeric(g['CorrectResponsesRT'], errors='coerce')
        if 'DaysAfterBaseline' in df_local.columns and g['DaysAfterBaseline'].notna().any():
            x = g['DaysAfterBaseline'].fillna(method='ffill').fillna(0).values
        else:
            x = np.arange(len(rt_series))
        try:
            if len(rt_series.dropna()) >= 2:
                slope = float(np.polyfit(x[:len(rt_series)], rt_series.fillna(method='ffill').fillna(rt_series.median()).values, 1)[0])
            else:
                slope = 0.0
        except Exception:
            slope = 0.0
        features.append({
            subject_col: subject,
            'seq_first_third_mean': seq_first_third_mean,
            'seq_last_third_mean': seq_last_third_mean,
            'seq_fatigue_effect': seq_fatigue_effect,
            'seq_mean_rt': seq_mean_rt if not np.isnan(seq_mean_rt) else 0.0,
            'seq_median_rt': seq_median_rt if not np.isnan(seq_median_rt) else 0.0,
            'long_reliability_change': long_reliability_change,
            'long_n_timepoints': long_n_timepoints,
            'long_rt_slope': slope
        })
    return pd.DataFrame(features) 