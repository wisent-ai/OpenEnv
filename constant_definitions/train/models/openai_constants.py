"""OpenAI model identifiers for evaluation."""

# ---------------------------------------------------------------------------
# OpenAI API models
# ---------------------------------------------------------------------------

GPT_5_4 = "gpt-5.4"
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
O3 = "o3"
O3_MINI = "o3-mini"
O4_MINI = "o4-mini"

# ---------------------------------------------------------------------------
# OpenAI open-weight models (Apache 2.0)
# ---------------------------------------------------------------------------

GPT_OSS_20B = "openai/gpt-oss-20b"

# API-only models
OPENAI_API_MODELS = (GPT_4O_MINI, GPT_4O, GPT_5_4, O3_MINI, O3, O4_MINI)

# Open-weight models run locally
OPENAI_LOCAL_MODELS = (GPT_OSS_20B,)

# All OpenAI models
OPENAI_MODELS = OPENAI_API_MODELS + OPENAI_LOCAL_MODELS
