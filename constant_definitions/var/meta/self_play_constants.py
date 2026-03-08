"""Constants for self-play multi-agent training."""

# Opponent update frequency (steps between opponent refresh)
SELF_PLAY_OPPONENT_UPDATE_INTERVAL = 50

# Maximum frozen checkpoints kept in the opponent pool
SELF_PLAY_POOL_MAX_SIZE = 5

# Self-play reward weights (numerator / denominator pairs)
SELF_PLAY_EXPLOIT_WEIGHT_NUMERATOR = 3
SELF_PLAY_EXPLOIT_WEIGHT_DENOMINATOR = 10

SELF_PLAY_COOP_WEIGHT_NUMERATOR = 3
SELF_PLAY_COOP_WEIGHT_DENOMINATOR = 10

SELF_PLAY_PARETO_WEIGHT_NUMERATOR = 2
SELF_PLAY_PARETO_WEIGHT_DENOMINATOR = 10

SELF_PLAY_FAIRNESS_WEIGHT_NUMERATOR = 1
SELF_PLAY_FAIRNESS_WEIGHT_DENOMINATOR = 10

SELF_PLAY_ADAPT_WEIGHT_NUMERATOR = 1
SELF_PLAY_ADAPT_WEIGHT_DENOMINATOR = 10

# Training defaults
SELF_PLAY_DEFAULT_EPISODES_PER_STEP = 16
SELF_PLAY_DEFAULT_MAX_STEPS = 500
SELF_PLAY_CHECKPOINT_PREFIX = "self_play_step"
SELF_PLAY_WARMUP_EPISODES = 32

# Opponent strategy label used in trajectory metadata
SELF_PLAY_OPPONENT_LABEL = "agent"

# Anthropic OAuth constants for self-play integration
ANTHROPIC_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
ANTHROPIC_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_OAUTH_BETA_HEADER = "oauth-2025-04-20"
ANTHROPIC_OAUTH_MAX_TOKENS = 5

# OpenAI OAuth constants for self-play integration
OPENAI_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"

# Supabase constants for credential storage
SUPABASE_OAUTH_TABLE = "oauth_credentials"
SUPABASE_PROVIDER_ANTHROPIC = "anthropic"
SUPABASE_PROVIDER_OPENAI = "openai"
