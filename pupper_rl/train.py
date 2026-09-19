#!/usr/bin/env python3
"""Train a Pupper v3 policy and push it to W&B.

No RL-library imports live here: the algorithm module owns the training loop,
checkpoints and the policy export (see pupper_rl/algos/base.py).

Usage:
    python train.py                                    # headless (default)
    python train.py --head                             # viewer window at each eval
    python train.py --render video                     # mp4 per eval, logged to W&B
    python train.py ppo.learning_rate=1e-4 ppo.num_envs=1024
    python train.py wandb.mode=disabled                # local smoke test

Deploy on robot afterwards:
    python3 ../deploy.py   -> enter the number at the end of the run name
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs/ppo_default.yaml"))
    parser.add_argument("--render", choices=["headless", "human", "video"], default=None,
                        help="override render.mode")
    parser.add_argument("--head", dest="render", action="store_const", const="human",
                        help="shorthand for --render human")
    parser.add_argument("--headless", dest="render", action="store_const", const="headless")
    parser.add_argument("overrides", nargs="*", help="dotted key=value config overrides")
    return parser.parse_args()


def main():
    args = parse_args()

    # GL backend must be chosen before mujoco is imported
    from pupper_rl.utils.config import load_config, resolve_path
    cfg = load_config(args.config, args.overrides)
    if args.render:
        cfg["render"]["mode"] = args.render
    render_mode = cfg["render"]["mode"]
    if render_mode == "video":
        os.environ.setdefault("MUJOCO_GL", "egl")     # offscreen
    elif render_mode == "human":
        os.environ.setdefault("MUJOCO_GL", "glfw")    # window
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")  # MJX speedup

    import mujoco
    import wandb

    from pupper_rl.algos import Hooks, get_algo
    from pupper_rl.envs.pupper_mjx_env import PupperMJXEnv
    from pupper_rl.utils import render as rnd
    from pupper_rl.utils import wandb_utils as wu

    run = wu.init_wandb(cfg)
    out_dir = os.path.join(cfg["logging"]["output_root"], wu.run_tag(run))
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = cfg["env"]
    model_xml = resolve_path(cfg, env_cfg["model_xml"])
    env = PupperMJXEnv(env_cfg, model_xml, resolve_path(cfg, env_cfg["reward_file"]))
    mj_model = mujoco.MjModel.from_xml_path(str(model_xml)) if render_mode != "headless" else None

    best = {"reward": float("-inf")}
    eval_count = {"n": 0}
    r_cfg = cfg["render"]

    # ---------------- hooks handed to the algorithm ----------------
    def on_progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        best["reward"] = max(best["reward"], r)
        print(f"step {step:,}  eval/episode_reward {r:.3f}")
        if run:
            wandb.log(metrics, step=step)

    def on_checkpoint(step, path):
        wu.log_checkpoint(run, path, step)

    def wants_rollout(step):
        eval_count["n"] += 1  # called once per checkpoint by the algorithm
        return render_mode != "headless" and eval_count["n"] % r_cfg["every_evals"] == 0

    def on_rollout_policy(step, policy):
        traj = rnd.rollout_qpos(env, policy, r_cfg["episode_steps"], seed=cfg["seed"])
        fps = 1.0 / env_cfg["ctrl_dt"]
        if render_mode == "human":
            rnd.view_human(mj_model, traj, fps)
        else:
            path = os.path.join(out_dir, f"rollout_{step}.mp4")
            rnd.render_video(mj_model, traj, path, fps, r_cfg["width"], r_cfg["height"], r_cfg["camera"])
            if run:
                wandb.log({"rollout": wandb.Video(path, fps=int(fps), format="mp4")}, step=step)

    hooks = Hooks(on_progress=on_progress, on_checkpoint=on_checkpoint,
                  on_rollout_policy=on_rollout_policy, wants_rollout=wants_rollout)

    result = get_algo(cfg["algo"])(env, cfg, out_dir, hooks)

    # Final export -> RTNeural JSON -> W&B (picked up by download_latest_policy.py)
    json_path = result.export_json(os.path.join(out_dir, wu.policy_filename(run, best["reward"])))
    wu.log_policy(run, json_path)
    print(f"Policy exported: {json_path}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
