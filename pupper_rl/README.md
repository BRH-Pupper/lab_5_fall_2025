# pupper_rl

Basic PPO template: Pupper v3 in MJX (GPU) + brax PPO → W&B → robot via `lab_5_fall_2025/deploy.py`.

Built for **ROS Jazzy** (the distro the robot and this workspace target): run it on a Jazzy
system — the `pupper-jazzy` distrobox or a native install. `eval.py --backend ros` launches the
workspace directly and will not work on Humble.

## Layout

```
pupper_rl/
├── train.py                     # entry: config → wandb → env → PPO → export → upload
├── eval.py                      # inference: 10 episodes in the ROS sim, logged to W&B
├── configs/ppo_default.yaml     # all settings; override with dotted key=value
├── requirements.txt             # pinned to pupperv3-monorepo/ai/rl/uv.lock
└── pupper_rl/
    ├── envs/
    │   ├── pupper_mjx_env.py    # brax PipelineEnv (mjx backend), obs == neural_controller.cpp
    │   └── reward_adapter.py    # loads ../reward.py read-only
    ├── eval/
    │   ├── ros_backend.py       # launch/teardown, start-pose keyframe swap (orchestrator)
    │   └── ros_agent.py         # runs under SYSTEM python: activation, /cmd_vel, recording
    ├── algos/                   # all RL-library code lives here
    │   ├── __init__.py          # ALGOS registry
    │   ├── base.py              # Hooks / TrainResult: the train.py <-> algo contract
    │   ├── ppo.py               # brax PPO: loop, checkpoints, export
    │   └── brax_export.py       # brax params → RTNeural JSON (obs norm baked in)
    └── utils/
        ├── config.py            # yaml + CLI overrides
        ├── render.py            # rollout playback: viewer window or mp4
        └── wandb_utils.py       # init, naming, artifact upload
```

## Run

```bash
pip install -r requirements.txt        # python 3.12, NVIDIA GPU
wandb login
python train.py
python train.py wandb.mode=disabled ppo.num_timesteps=20000 ppo.num_envs=64 \
    ppo.batch_size=16 ppo.num_minibatches=4 ppo.num_evals=2   # smoke test
```

## Rendering

```bash
python train.py                  # headless (default, fastest)
python train.py --head           # viewer window at each eval, blocks training
python train.py --render video   # mp4 per eval -> outputs/<run>/ and W&B
```
Tune under `render:` in the config (`every_evals`, `episode_steps`, `width`,
`height`, `camera`). `--head` needs a display; `video` needs `imageio[ffmpeg]`
and uses `MUJOCO_GL=egl`. Training itself always runs headless on the GPU;
rendering replays one deterministic rollout on CPU MuJoCo.

## Config overrides

`train.py` takes any number of bare `key=value` args after the flags. Keys are
dotted paths into `configs/ppo_default.yaml`; values are parsed as YAML
(numbers, bools, null, lists). Unknown keys are created, not rejected.

```bash
python train.py ppo.num_envs=1024 ppo.learning_rate=1e-4
python train.py policy.hidden_sizes=[512,256,128] env.reward_scales.termination=-50
python train.py wandb.mode=disabled render.mode=video
```

## Adding an algorithm

`train.py` imports no RL library. An algorithm module implements

```python
train(env, cfg, out_dir, hooks) -> TrainResult
```

and owns its networks, training loop, checkpoint files and policy export. It
talks back only through `Hooks` (see `algos/base.py`):
`on_progress(step, {metric: float})`, `on_checkpoint(step, path)`,
`wants_rollout(step) -> bool`, `on_rollout_policy(step, policy(obs, rng) -> action)`.
Register it in `ALGOS` and set `algo:` in the config; `cfg["ppo"]`, `cfg["policy"]`
and `cfg["export"]` are read inside the algorithm, so the config stays a thin
wrapper. Checkpoint names must never contain ".json" (the robot-side downloader
picks policies by that substring).

## Evaluation (`eval.py`)

```bash
python eval.py --backend ros                  # prompts for a run number, like deploy.py
python eval.py --backend ros --run 42
python eval.py --backend ros --skip-download --episodes 2 wandb.mode=disabled
```

Model retrieval is deploy.py's flow: prompt a W&B run number → `../download_latest_policy.py`
→ `test_policy.json`, so the evaluated file is what the robot would load. The policy then runs
as RTNeural inside the `neural_controller` C++ controller (`neural_controller_test`, the one
whose `model_path` is `test_policy.json`).

Per episode: a randomized start pose (reuses `env.init_randomization`) is written as a sibling
MuJoCo file and the install symlink is pointed at it; `ros2 launch neural_controller launch.py
sim:=True teleop:=False` comes up; the controller is activated through
`/controller_manager/switch_controller`; `/cmd_vel` is driven with the episode's command preset;
`/<controller>/observation` is recorded; then everything is torn down and the symlink restored.

W&B: the eval does not create a run of its own. It identifies the run that produced the
deployed policy — from `--run <N>`, else by finding which `policy_<run>_max_reward_*.json` in
the model dir is byte-identical to `test_policy.json` — resumes it, and writes
`ros_eval/<metric>` into that run's summary plus a `ros_eval/episodes` table. The prefix is the
backend name because brax's evaluator already owns `eval/*` on the training run. If the run
cannot be identified, nothing is logged and it says so; episode JSONs are in `outputs/` either
way.

Environment notes:
- **Run this on ROS Jazzy** — the system the workspace was built for (`pupper-jazzy` distrobox
  or a native Jazzy install). On a Humble system the launch dies with
  `GLIBCXX_3.4.32 not found` and every spawner exits 1. Every ROS command goes through
  `eval.ros_crun` (`distrobox-setup/crun.sh`), which drops inherited Humble paths, sources
  Jazzy + the workspace install and selects cyclonedds, so it does not matter what the calling
  shell had sourced.
- `rclpy` lives in the system python, not this venv, so the ROS side is a separate script
  (`eval/ros_agent.py`) run under that interpreter; it returns metrics as JSON.
- Metrics are limited to what ROS topics expose: survival, tilt (from projected gravity),
  yaw-rate tracking, action rate, joint deviation. **No torso height and no body linear
  velocity** — the sim publishes no ground-truth pose, so reward terms cannot be recomputed.
- `--backend mjx` is not implemented yet.

## W&B / deploy contract (from download_latest_policy.py)

- project `pupperv3-mjx-rl`, entity = logged-in user
- run name ends in `-<N>` (wandb default) → `deploy.py` asks for `N`
- policy artifact name = `policy_<run.name>_max_reward_<r>.json` = file basename
- checkpoints logged as `ppo_params_<run.name>` (no ".json" so downloader skips them)

## Reward

`reward.py` is loaded read-only. A function `reward_<name>` is used when
`env.reward_scales` has `<name>`; it is called **by argument name**:

1. quantities the env publishes each step (`PupperMJXEnv._reward_inputs`):
   `pipeline_state, x, xd, commands, torques, joint_angles, joint_vel,
   velocities, last_joint_vel, act, last_act, dt, default_pose, done, step,
   air_time, first_contact, contact_filt, feet_site_id, lower_leg_body_id,
   geom_ids`
2. else `env.reward_params` in the config (`tracking_sigma`, `command_threshold`,
   `step_threshold`, `minimum_airtime`, `desired_world_z_in_body_frame`, ...)
3. else the function's own default

An argument matching none of those raises at env construction with the list of
available names — never inside jit. New reward term needing new state → add it
to `_reward_inputs` (env) or `reward_params` (config), never to `reward.py`.

Foot contact is `foot site z < env.foot_contact_height`; air time and
`first_contact` are tracked in `state.info`.
