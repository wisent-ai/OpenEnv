"""FastAPI application factory for Kant."""
from openenv.core.env_server.http_server import create_app
from env.models import GameAction, GameObservation
from env.environment import KantEnvironment
from constant_definitions.game_constants import MAX_CONCURRENT_ENVS

app = create_app(
    KantEnvironment,
    GameAction,
    GameObservation,
    env_name="kant",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
