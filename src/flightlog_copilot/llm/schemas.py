"""Strict schema separating rule-based results from unverified AI suggestions."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HypothesisReview(StrictModel):
    hypothesis_id: str
    priority: int = Field(ge=1)
    explanation: str
    supporting_evidence: List[str]
    counter_evidence: List[str]
    uncertainties: List[str]


class RecommendedExperiment(StrictModel):
    title: str
    purpose: str
    variables_to_change: List[str]
    variables_to_log: List[str]
    expected_observation: str
    safety_notes: List[str]


class CopilotResponse(StrictModel):
    summary: str
    hypothesis_review: List[HypothesisReview]
    additional_hypotheses: List[str]
    recommended_experiments: List[RecommendedExperiment]
    missing_data_requests: List[str]
