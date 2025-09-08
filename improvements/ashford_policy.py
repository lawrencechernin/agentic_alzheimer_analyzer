#!/usr/bin/env python3
"""
Ashford Policy Filter
- Applies MemTrax test quality filters (status, accuracy, RT bounds)
- Default thresholds: accuracy ≥ 0.65, RT in [0.5, 2.5]
"""
import pandas as pd


def apply_ashford(memtrax_df: pd.DataFrame,
                  accuracy_threshold: float = 0.65,
                  rt_min: float = 0.5,
                  rt_max: float = 2.5,
                  status_col: str = 'Status') -> pd.DataFrame:
    df = memtrax_df.copy()
    cond = (
        (df.get(status_col, 'Collected') == 'Collected') &
        (pd.to_numeric(df['CorrectPCT'], errors='coerce') >= accuracy_threshold) &
        (pd.to_numeric(df['CorrectResponsesRT'], errors='coerce') >= rt_min) &
        (pd.to_numeric(df['CorrectResponsesRT'], errors='coerce') <= rt_max)
    )
    df = df[cond].copy()
    return df 