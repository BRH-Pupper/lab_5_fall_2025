"""W&B helpers. Naming matches lab_5_fall_2025/download_latest_policy.py:

- project: "pupperv3-mjx-rl"
- run name must end in "-<number>" (wandb default names do, e.g. "swift-river-42");
  deploy.py asks for that number.
- policy artifact name must contain ".json" AND equal the file's basename,
  because the downloader does `art.download()` then copies `<artifact base name>`.
"""

import os
from datetime import datetime

import wandb


def init_wandb(cfg: dict):
    w = cfg["wandb"]
    if not w["enabled"] or w["mode"] == "disabled":
        return None
    run = wandb.init(
        project=w["project"],
        entity=w["entity"],
        mode=w["mode"],
        config={k: v for k, v in cfg.items() if not k.startswith("_")},
        save_code=True,
    )
    if not run.name.split("-")[-1].isdigit():
        print(f"WARNING: run name '{run.name}' has no '-<number>' suffix; deploy.py cannot find it.")
    return run


_START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_tag(run) -> str:
    return run.name if run else _START_TIME


def policy_filename(run, max_reward: float) -> str:
    return f"policy_{run_tag(run)}_max_reward_{max_reward:0.2f}.json"


def log_policy(run, json_path: str):
    """Upload exported policy JSON; artifact name == file basename (contains .json)."""
    if run is None:
        return
    wandb.log_model(path=json_path, name=os.path.basename(json_path))


def log_checkpoint(run, ckpt_path: str, iteration: int):
    if run is None:
        return
    # name must NOT contain ".json" or the downloader could pick it instead of the policy
    name = f"ppo_params_{run.name}"
    inner = "params" + os.path.splitext(ckpt_path)[1]  # keep the algorithm's own extension
    assert ".json" not in name and ".json" not in inner, f"checkpoint artifact must not look like a policy: {name}/{inner}"
    art = wandb.Artifact(name=name, type="model", metadata={"step": iteration})
    art.add_file(ckpt_path, name=inner)
    run.log_artifact(art, aliases=[f"step-{iteration}", "latest"])
