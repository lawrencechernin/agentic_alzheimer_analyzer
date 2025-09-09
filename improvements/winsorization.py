#!/usr/bin/env python3
from typing import Optional
import numpy as np
import pandas as pd


def winsorize_reaction_times(df: pd.DataFrame, reaction_col: str = 'ReactionTimes', low: float = 0.4, high: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    if reaction_col not in out.columns:
        return out

    def _clip_str(rt_str: Optional[str]) -> str:
        parts = []
        for x in str(rt_str).split(','):
            x = x.strip()
            if not x:
                continue
            try:
                v = float(x)
                if np.isfinite(v):
                    v = min(max(v, low), high)
                    parts.append(f"{v:.3f}")
            except Exception:
                continue
        return ','.join(parts)

    out[reaction_col] = out[reaction_col].apply(_clip_str)
    return out 