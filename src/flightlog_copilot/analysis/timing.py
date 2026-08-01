"""Sampling quality and logging anomaly metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def analyze_timing(
    full_frame: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    dropout_multiplier: float = 3.0,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "full_rows": int(len(full_frame)),
        "analysis_rows": int(len(analysis_frame)),
        "dropout_multiplier": float(dropout_multiplier),
        "dropout_intervals": [],
    }
    if "timestamp" not in analysis_frame:
        result["reason"] = "timestamp 파라미터가 없습니다."
        return result

    full_time = pd.to_numeric(full_frame.get("timestamp", pd.Series(dtype=float)), errors="coerce")
    analysis_time = pd.to_numeric(analysis_frame["timestamp"], errors="coerce")
    result["duplicate_timestamp_count"] = int(full_time.duplicated().sum())
    result["reverse_timestamp_count"] = int((full_time.diff() < 0).sum())
    clean = analysis_time.dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if len(clean) < 2:
        result["reason"] = "유효하고 서로 다른 timestamp가 2개 미만입니다."
        return result

    intervals = clean.diff().dropna()
    intervals = intervals[intervals > 0]
    if intervals.empty:
        result["reason"] = "양수 sampling interval이 없습니다."
        return result
    mean_dt = float(intervals.mean())
    median_dt = float(intervals.median())
    std_dt = float(intervals.std(ddof=0))
    threshold = median_dt * float(dropout_multiplier)
    dropouts = intervals[intervals > threshold]
    result.update(
        {
            "available": True,
            "duration_s": float(clean.iloc[-1] - clean.iloc[0]),
            "mean_sampling_interval_s": mean_dt,
            "median_sampling_interval_s": median_dt,
            "sampling_interval_std_s": std_dt,
            "estimated_sampling_frequency_hz": 1.0 / median_dt if median_dt > 0 else None,
            "sampling_jitter_ratio": std_dt / median_dt if median_dt > 0 else None,
            "dropout_threshold_s": threshold,
            "dropout_intervals": [
                {"start_s": float(clean.iloc[index - 1]), "end_s": float(clean.iloc[index]), "gap_s": float(value)}
                for index, value in dropouts.items()
            ],
            "sampling_intervals_s": intervals.astype(float).tolist(),
        }
    )
    return result
