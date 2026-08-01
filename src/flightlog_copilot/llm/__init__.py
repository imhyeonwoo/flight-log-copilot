"""Optional OpenAI structured-output explainer."""

from flightlog_copilot.llm.client import LLMAnalysisError, analyze_with_openai
from flightlog_copilot.llm.schemas import CopilotResponse

__all__ = ["LLMAnalysisError", "analyze_with_openai", "CopilotResponse"]
