"""PPO via brax.training (runs fully on GPU with the MJX env).

Every brax import lives in this module; train.py only sees Hooks/TrainResult.
"""

import functools
import os

import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as brax_ppo
from flax import linen

from .base import Hooks, TrainResult
from .brax_export import export_policy_json

ACTIVATIONS = {"elu": linen.elu, "relu": linen.relu, "tanh": linen.tanh}


def _scalars(metrics: dict) -> dict:
    """Device arrays -> floats. Non-scalars are averaged rather than raising mid-run."""
    return {k: float(np.asarray(v).mean()) for k, v in metrics.items()}


def train(env, cfg: dict, out_dir: str, hooks: Hooks = None) -> TrainResult:
    hooks = hooks or Hooks()
    p = cfg["policy"]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(p["hidden_sizes"]),
        value_hidden_layer_sizes=tuple(p["value_hidden_sizes"]),
        activation=ACTIVATIONS[p["activation"]],
    )

    def progress_fn(step, metrics):
        hooks.on_progress(int(step), _scalars(metrics))

    def policy_params_fn(step, make_policy, params):
        step = int(step)
        ckpt = os.path.join(out_dir, f"params_{step}.pkl")  # no ".json": see base.py
        model.save_params(ckpt, params)
        hooks.on_checkpoint(step, ckpt)
        if hooks.wants_rollout(step):  # exactly once per checkpoint
            policy = make_policy(params, deterministic=True)
            hooks.on_rollout_policy(step, lambda obs, rng: policy(obs, rng)[0])

    _, params, metrics = brax_ppo.train(
        environment=env,
        seed=cfg["seed"],
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        **cfg["ppo"],
    )

    return TrainResult(
        params=params,
        export_json=lambda path: export_policy_json(params, cfg["env"], cfg["policy"], cfg["export"], path),
        metrics=_scalars(metrics),
    )
