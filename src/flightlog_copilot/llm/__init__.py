"""Provider-aware optional AI structured-output explainer."""

from flightlog_copilot.llm.base import LLMAnalysisError, error_message, safe_list_models
from flightlog_copilot.llm.client import analyze_with_openai
from flightlog_copilot.llm.factory import create_llm_client
from flightlog_copilot.llm.settings import AiSettings, ProviderSettings
from flightlog_copilot.llm.schemas import CopilotResponse

__all__ = [
    "AiSettings",
    "CopilotResponse",
    "LLMAnalysisError",
    "ProviderSettings",
    "analyze_with_openai",
    "create_llm_client",
    "error_message",
    "safe_list_models",
]
