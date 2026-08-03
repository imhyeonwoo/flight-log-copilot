"""Provider identifiers and maintainable model-picker metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Tuple


AiProvider = Literal["openai", "openai_compatible"]
ApiKeyMode = Literal["api_key", "no_key", "bearer"]


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    description: str


# Keep picker defaults in this module so model IDs can be updated without
# touching Streamlit UI or client code. Direct model-ID entry is always available.
DEFAULT_OPENAI_MODELS: Tuple[ModelOption, ...] = (
    ModelOption("gpt-5.6-terra", "GPT-5.6 Terra", "Balanced cost, latency, and quality."),
    ModelOption("gpt-5.6-sol", "GPT-5.6 Sol", "Quality-first model for difficult analysis."),
    ModelOption("gpt-5.6-luna", "GPT-5.6 Luna", "Efficient option for lighter workloads."),
    ModelOption("gpt-5.6", "GPT-5.6 alias", "Alias that follows the GPT-5.6 flagship route."),
)

DEFAULT_OPENAI_MODEL = DEFAULT_OPENAI_MODELS[0].id
SUPPORTED_PROVIDERS: Tuple[AiProvider, ...] = ("openai", "openai_compatible")


def is_likely_openai_text_model(model_id: str) -> bool:
    """Conservatively filter model-list results for text generation."""
    value = str(model_id).lower()
    return bool(re.match(r"^(gpt-|chatgpt-|o[1-9](?:-|$))", value))
