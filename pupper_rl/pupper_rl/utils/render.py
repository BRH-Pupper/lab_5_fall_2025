"""Rollout + rendering helpers (MJX rollout on GPU, playback on CPU MuJoCo).

Modes:
  headless -> nothing rendered (default, fastest)
  human    -> mujoco.viewer window, plays the rollout in real time
  video    -> offscreen render to mp4, logged to W&B

Needs a GL backend: `human` uses glfw, `video` uses MUJOCO_GL=egl (set in train.py).
"""

import time

import jax
import mujoco
import numpy as np


def rollout_qpos(env, policy, steps: int, seed: int = 0) -> list[np.ndarray]:
    """Run the unwrapped env (no auto-reset) and return the qpos trajectory.

    `policy(obs, rng) -> action`, as handed over by Hooks.on_rollout_policy.
    """
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    state = reset(key)

    traj = [np.asarray(state.pipeline_state.qpos)]
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action = policy(state.obs, key)
        state = step(state, action)
        traj.append(np.asarray(state.pipeline_state.qpos))
        if bool(state.done):
            break
    return traj


def _playback_data(mj_model, qpos):
    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)
    return data


def view_human(mj_model, traj, fps: float):
    """Blocking playback in a viewer window."""
    import mujoco.viewer

    data = mujoco.MjData(mj_model)
    dt = 1.0 / fps
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        for qpos in traj:
            if not viewer.is_running():
                break
            t0 = time.time()
            data.qpos[:] = qpos
            mujoco.mj_forward(mj_model, data)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))


def render_video(mj_model, traj, path: str, fps: float, width: int, height: int, camera=-1) -> str:
    import imageio.v2 as imageio

    frames = []
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        for qpos in traj:
            renderer.update_scene(_playback_data(mj_model, qpos), camera=camera)
            frames.append(renderer.render())
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path
