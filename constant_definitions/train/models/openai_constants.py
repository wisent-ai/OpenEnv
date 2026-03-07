"""OpenAI model identifiers for evaluation."""

# ---------------------------------------------------------------------------
# OpenAI API models
# ---------------------------------------------------------------------------

GPT_5_4 = "gpt-5.4"

# ---------------------------------------------------------------------------
# OpenAI open-weight models (Apache 2.0)
# ---------------------------------------------------------------------------

GPT_OSS_20B = "openai/gpt-oss-20b"

# API-only models
OPENAI_API_MODELS = (GPT_5_4,)

# Open-weight models run locally
OPENAI_LOCAL_MODELS = (GPT_OSS_20B,)

# All OpenAI models
OPENAI_MODELS = OPENAI_API_MODELS + OPENAI_LOCAL_MODELS
