"""Transparent score accumulation utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional


def apply_rule(
    hypothesis: Dict[str, Any],
    condition: Optional[bool],
    delta: int,
    rule: str,
    evidence: str,
    counter_evidence: str = "",
    observed: Any = None,
    threshold: Any = None,
) -> None:
    if condition is None:
        hypothesis["score_details"].append(
            {"rule": rule, "delta": 0, "result": "계산 불가", "observed": observed, "threshold": threshold}
        )
        return
    applied_delta = int(delta if condition else (-max(1, abs(delta) // 4) if counter_evidence else 0))
    hypothesis["score"] = max(0, min(100, int(hypothesis["score"]) + applied_delta))
    if condition and evidence:
        hypothesis["evidence"].append(evidence)
    elif not condition and counter_evidence:
        hypothesis["counter_evidence"].append(counter_evidence)
    hypothesis["score_details"].append(
        {
            "rule": rule,
            "delta": applied_delta,
            "result": "충족" if condition else "미충족",
            "observed": observed,
            "threshold": threshold,
        }
    )
