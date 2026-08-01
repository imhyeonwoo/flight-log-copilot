"""Robust, inspectable CSV loading and column profiling."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CSVLoadResult:
    data: pd.DataFrame
    encoding: str
    delimiter: str
    duplicate_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CSVLoadError(ValueError):
    """Raised when uploaded bytes cannot be interpreted as a usable CSV."""


def _decode(content: bytes) -> tuple[str, str]:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(str(exc))
    raise CSVLoadError("CSV 텍스트 인코딩을 판별하지 못했습니다: " + "; ".join(errors))


def _detect_delimiter(text: str) -> str:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(content: bytes, filename: str = "uploaded.csv") -> CSVLoadResult:
    if not content or not content.strip():
        raise CSVLoadError("빈 CSV 파일입니다.")
    text, encoding = _decode(content)
    delimiter = _detect_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise CSVLoadError("빈 CSV 파일입니다.") from exc
    header = [cell.strip() for cell in header]
    if not header or all(not cell for cell in header):
        raise CSVLoadError("CSV 헤더가 없습니다.")
    if all(_looks_numeric(cell) for cell in header):
        raise CSVLoadError("첫 행이 모두 숫자여서 헤더가 없는 CSV로 판단했습니다.")
    duplicate_columns = sorted({name for name in header if name and header.count(name) > 1})
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            encoding_errors="strict",
            skipinitialspace=True,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise CSVLoadError(f"CSV를 읽지 못했습니다: {exc}") from exc
    if frame.empty:
        raise CSVLoadError("CSV에 데이터 행이 없습니다.")
    warnings = []
    if duplicate_columns:
        warnings.append("중복 컬럼명: " + ", ".join(duplicate_columns))
    if len(frame.columns) == 1:
        warnings.append("컬럼이 하나뿐입니다. 구분자 자동 감지 결과를 확인하십시오.")
    return CSVLoadResult(frame, encoding, delimiter, duplicate_columns, warnings)


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def infer_semantic_type(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "unknown"
    lowered = {str(value).strip().lower() for value in values.head(500)}
    if lowered and lowered.issubset({"true", "false", "0", "1", "yes", "no", "on", "off"}):
        return "boolean"
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return "numeric"
    return "string"


def profile_columns(frame: pd.DataFrame, sample_rows: int = 5) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    for index, column in enumerate(frame.columns):
        series = frame[column]
        semantic = infer_semantic_type(series)
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric_values = numeric.dropna()
        rows.append(
            {
                "순서": index + 1,
                "컬럼명": str(column),
                "pandas dtype": str(series.dtype),
                "추정 타입": semantic,
                "유효 개수": int(series.notna().sum()),
                "결측 비율": float(series.isna().mean()),
                "최솟값": _safe_stat(numeric_values, "min"),
                "최댓값": _safe_stat(numeric_values, "max"),
                "평균": _safe_stat(numeric_values, "mean"),
                "표준편차": _safe_stat(numeric_values, "std"),
                "고유값 개수": int(series.nunique(dropna=True)),
                "앞부분 샘플": series.head(sample_rows).tolist(),
            }
        )
    return pd.DataFrame(rows)


def _safe_stat(series: pd.Series, operation: str) -> Optional[float]:
    if series.empty:
        return None
    value = getattr(series, operation)()
    return float(value) if pd.notna(value) else None
