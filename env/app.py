"""FastAPI application factory for Machiavelli."""
from openenv.core.env_server.http_server import create_app
from env.models import GameAction, GameObservation
from env.environment import MachiavelliEnvironment
from constant_definitions.game_constants import MAX_CONCURRENT_ENVS

app = create_app(
    MachiavelliEnvironment,
    GameAction,
    GameObservation,
    env_name="machiaveli",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
