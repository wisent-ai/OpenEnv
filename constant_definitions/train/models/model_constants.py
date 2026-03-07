"""Model registry -- aggregates all provider-specific model constants."""

from constant_definitions.train.models.local_constants import (
    GEMMA_3_27B,
    LLAMA_3_1_8B,
    LLAMA_3_2_1B,
    LOCAL_MODELS,
    MISTRAL_SMALL_3_24B,
    PHI_4_REASONING,
    QWEN_3_5_9B,
    QWEN_3_5_27B,
)
from constant_definitions.train.models.openai_constants import (
    GPT_5_4,
    GPT_OSS_20B,
    OPENAI_API_MODELS,
    OPENAI_LOCAL_MODELS,
    OPENAI_MODELS,
)
from constant_definitions.train.models.anthropic_constants import (
    ANTHROPIC_MODELS,
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
)

# ---------------------------------------------------------------------------
# Short-name registry
# ---------------------------------------------------------------------------

# Maps human-readable short names to full model identifiers.
# Used by experiment scripts to select models by name.
MODELS = {
    # Open-weight -- Meta
    "llama3.2-1b": LLAMA_3_2_1B,
    "llama3.1-8b": LLAMA_3_1_8B,
    # Open-weight -- Qwen
    "qwen3.5-9b": QWEN_3_5_9B,
    "qwen3.5-27b": QWEN_3_5_27B,
    # Open-weight -- Google
    "gemma3-27b": GEMMA_3_27B,
    # Open-weight -- Microsoft
    "phi4-reasoning": PHI_4_REASONING,
    # Open-weight -- Mistral
    "mistral-small-24b": MISTRAL_SMALL_3_24B,
    # Open-weight -- OpenAI
    "gpt-oss-20b": GPT_OSS_20B,
    # API -- OpenAI
    "gpt-5.4": GPT_5_4,
    # API -- Anthropic
    "claude-opus": CLAUDE_OPUS,
    "claude-sonnet": CLAUDE_SONNET,
    "claude-haiku": CLAUDE_HAIKU,
}

# ---------------------------------------------------------------------------
# Groupings
# ---------------------------------------------------------------------------

# All open-weight models that can be run and trained locally
ALL_LOCAL_MODELS = LOCAL_MODELS + OPENAI_LOCAL_MODELS

# Models evaluated via API only (no local weights)
API_MODELS = OPENAI_API_MODELS + ANTHROPIC_MODELS
