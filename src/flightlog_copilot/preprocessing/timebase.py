"""Timestamp cleanup shared by analysis modules."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def sorted_unique_time(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    if "timestamp" not in frame:
        return frame.copy(), 0, 0
    output = frame.copy()
    output["timestamp"] = pd.to_numeric(output["timestamp"], errors="coerce")
    output = output.dropna(subset=["timestamp"])
    reverse_count = int((output["timestamp"].diff() < 0).sum())
    duplicate_count = int(output["timestamp"].duplicated().sum())
    output = output.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="first")
    return output.reset_index(drop=True), duplicate_count, reverse_count
