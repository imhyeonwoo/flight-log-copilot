"""Detect and select contiguous Armed and AltHold intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class FlightSegment:
    kind: str
    start_s: float
    end_s: float
    rows: int
    start_index: int
    end_index: int

    def to_dict(self) -> dict:
        return asdict(self)


def detect_segments(frame: pd.DataFrame) -> Dict[str, List[FlightSegment]]:
    result: Dict[str, List[FlightSegment]] = {"armed": [], "althold": []}
    if "timestamp" not in frame:
        return result
    if "armed" in frame:
        result["armed"] = _segments_from_mask(frame, frame["armed"].fillna(False).astype(bool), "armed")
    if "althold_active" in frame:
        mask = frame["althold_active"].fillna(False).astype(bool)
        if "armed" in frame:
            mask &= frame["armed"].fillna(False).astype(bool)
        result["althold"] = _segments_from_mask(frame, mask, "althold")
    return result


def _segments_from_mask(frame: pd.DataFrame, mask: pd.Series, kind: str) -> List[FlightSegment]:
    segments: List[FlightSegment] = []
    active_start: Optional[int] = None
    for position, active in enumerate(mask.tolist() + [False]):
        if active and active_start is None:
            active_start = position
        elif not active and active_start is not None:
            end = position - 1
            segments.append(
                FlightSegment(
                    kind=kind,
                    start_s=float(frame.iloc[active_start]["timestamp"]),
                    end_s=float(frame.iloc[end]["timestamp"]),
                    rows=end - active_start + 1,
                    start_index=active_start,
                    end_index=end,
                )
            )
            active_start = None
    return segments


def select_time_range(frame: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    if "timestamp" not in frame:
        return frame.copy()
    lower, upper = sorted((float(start_s), float(end_s)))
    return frame[(frame["timestamp"] >= lower) & (frame["timestamp"] <= upper)].copy()
