"""Strict JSON rendering without NumPy objects or NaN values."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


def sanitize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return sanitize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if value is pd.NA or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    return value


def render_json_report(result: Any) -> str:
    return json.dumps(sanitize_for_json(result), ensure_ascii=False, indent=2, allow_nan=False)
