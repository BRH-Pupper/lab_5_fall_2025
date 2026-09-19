from . import ppo
from .base import Hooks, TrainResult

# config `algo:` -> train(env, cfg, out_dir, hooks) -> TrainResult
ALGOS = {"ppo": ppo.train}


def get_algo(name: str):
    if name not in ALGOS:
        raise ValueError(f"Unknown algo '{name}'. Available: {list(ALGOS)}")
    return ALGOS[name]


__all__ = ["ALGOS", "get_algo", "Hooks", "TrainResult"]
