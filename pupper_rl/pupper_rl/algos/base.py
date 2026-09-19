"""Algorithm interface: keeps RL-library code out of train.py.

An algorithm module exposes

    train(env, cfg, out_dir, hooks) -> TrainResult

and owns everything library-specific: networks, the training loop, checkpoint
files, and the export to the RTNeural JSON the robot loads. It reports back
only through `hooks`, which carry plain python types and callables, so train.py
never imports brax / sb3 / torch.

Rules for algorithm authors:
  - Call `hooks.wants_rollout(step)` AT MOST ONCE per checkpoint; train.py may
    keep a counter behind it (render cadence).
  - Cast metric values to float before calling `hooks.on_progress`.
  - Never put ".json" in a checkpoint file or artifact name: the robot-side
    download_latest_policy.py picks the first artifact whose name contains
    ".json" and would grab a checkpoint instead of the policy.
"""

from dataclasses import dataclass, field
from typing import Callable


def _noop(*args, **kwargs):
    return None


@dataclass
class Hooks:
    """Callbacks an algorithm calls. All optional; defaults do nothing."""

    on_progress: Callable[[int, dict], None] = _noop
    """(step, {metric name: float}) -- once per evaluation."""

    on_checkpoint: Callable[[int, str], None] = _noop
    """(step, path) -- after the algorithm wrote a checkpoint file."""

    on_rollout_policy: Callable[[int, Callable], None] = _noop
    """(step, policy) where policy(obs, rng) -> action, deterministic. For rendering."""

    wants_rollout: Callable[[int], bool] = lambda step: False
    """Asked before building a rollout policy, so the algorithm can skip the work."""


@dataclass
class TrainResult:
    """What an algorithm hands back to train.py."""

    params: object                       # opaque, library-specific
    export_json: Callable[[str], str]    # (path) -> path, writes the RTNeural JSON
    metrics: dict = field(default_factory=dict)
