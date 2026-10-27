"""ROS 2 sim evaluation backend: same stack deploy.py launches.

The policy runs as RTNeural inside the `neural_controller` C++ controller, so this
backend measures the deployed artifact, not the training graph.

Start-pose randomization: the sim's MuJoCo file is hardcoded in
mujoco_hardware_interface.cpp as
    <pupper_v3_description share>/description/mujoco_xml/pupper_v3_complete.xml
and colcon installed that path as a SYMLINK into the source tree. So each episode
writes a sibling copy with a randomized start pose and repoints the install symlink at
it. The original source file is never modified, and the symlink is always restored
(context manager + finally in run_episodes).

The pose goes on <body name="base_link">, NOT in the <key name="home"> keyframe. The
sim does `d = mj_makeData(m)` (mujoco_core_interactive.hpp loadmodel) and never calls
mj_resetDataKeyframe, so qpos starts at qpos0 and the keyframe is dead config. For the
free joint, qpos0 IS base_link's pos/quat, so patching those is what the sim honors.

Consequence: torso pose only. A hinge's qpos0 is its `ref`, and moving `ref` moves the
geometric zero, which would desync sim joint angles from the robot's -- so joints spawn
at 0 and velocities at 0 regardless. Little is lost: no controller is active for the
first seconds of a launch, so ctrl sits at 0 and the legs are driven straight long
before neural_controller_test activates. Joint and velocity randomization would need
mj_resetDataKeyframe in the sim.
"""

import atexit
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..utils.config import init_ranges

AGENT = Path(__file__).parent / "ros_agent.py"
MODEL_REL = "description/mujoco_xml/pupper_v3_complete.xml"
INSTALL_REL = "install/pupper_v3_description/share/pupper_v3_description"


def _euler_to_quat(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


class RosSim:
    def __init__(self, monorepo: Path, env_cfg: dict, eval_cfg: dict, out_dir: Path):
        self.monorepo = Path(monorepo)
        self.ws = self.monorepo / "ros2_ws"
        self.install_model = self.ws / INSTALL_REL / MODEL_REL
        self.src_model = self.ws / "src/pupper_v3_description" / MODEL_REL
        self.env_cfg = env_cfg
        self.eval_cfg = eval_cfg
        self.out_dir = Path(out_dir)
        self._mj = None   # lazily loaded sim model for the ground-clearance solve
        self.crun = Path(eval_cfg["ros_crun"]).expanduser()
        if not self.crun.is_file():
            raise FileNotFoundError(f"container launcher not found: {self.crun}")
        if not self.src_model.is_file():
            raise FileNotFoundError(f"sim model not found: {self.src_model}")
        if not self.install_model.is_symlink():
            raise RuntimeError(
                f"{self.install_model} is not a symlink; this backend expects a "
                "symlink-install workspace (colcon build --symlink-install)")
        # a run killed mid-episode leaves the symlink pointing at a deleted temp file
        if "eval_tmp" in os.readlink(self.install_model) or not self.install_model.exists():
            print(f"[ros] repairing stale sim model symlink -> {self.src_model}")
            self.install_model.unlink()
            self.install_model.symlink_to(self.src_model)
        for stale in self.src_model.parent.glob("pupper_v3_complete.eval_tmp_*.xml"):
            stale.unlink()

    # ---------------- start pose ----------------
    def _clearance_lift(self, quat, z) -> tuple:
        """How far to raise the torso so the lowest collision geom clears the ground.

        Same solve as PupperMJXEnv._initial_state: a tilted pose buries the feet in the
        floor, and dropping from a safe height instead would destroy the torso_z
        distribution. Run on the real sim model (the one mujoco_hardware_interface.cpp
        loads) with plain mujoco rather than mjx. Joints are left at qpos0 because that
        is what the sim spawns -- straight legs, so the floor sits higher than training's
        crouched default pose.
        """
        import mujoco   # only the ROS backend needs it at eval time

        if self._mj is None:
            model = mujoco.MjModel.from_xml_path(str(self.src_model))
            # conaffinity > 0 and bodyid > 0: collidable, and not the static floor
            geoms = np.flatnonzero(model.geom_conaffinity * (model.geom_bodyid > 0))
            self._mj = (model, mujoco.MjData(model), geoms,
                        model.geom_size[geoms].max(axis=1))
        model, data, geoms, half_extent = self._mj

        # qpos0, not the keyframe: that is what mj_makeData gives the sim, joints included
        data.qpos[:] = model.qpos0
        data.qpos[:3] = [0.0, 0.0, z]
        data.qpos[3:7] = quat
        mujoco.mj_forward(model, data)
        lowest = float(np.min(data.geom_xpos[geoms, 2] - half_extent))
        return max(0.0, self.env_cfg["init_randomization"]["ground_clearance"] - lowest), lowest

    def _randomized_xml(self, rng: np.random.Generator, index: int) -> Path:
        """Sibling copy of the sim model with a randomized base_link transform."""
        r = self.env_cfg["init_randomization"]
        z = rng.uniform(*r["torso_z"])
        tilt = init_ranges(r)      # same parser the env uses, so the two cannot drift
        roll = rng.uniform(*tilt["roll"])
        pitch = rng.uniform(*tilt["pitch"])
        yaw = rng.uniform(*tilt["yaw"])
        quat = _euler_to_quat(roll, pitch, yaw)
        lift, lowest = self._clearance_lift(quat, z)
        z += lift

        pos_attr = f'pos="0 0 {z:.6f}"'
        quat_attr = f'quat="{" ".join(f"{v:.6f}" for v in quat)}"'
        text = self.src_model.read_text()
        text, n = re.subn(r'(<body name="base_link")\s+pos="[^"]*"',
                          rf'\1 {pos_attr}', text, count=1)
        # base_link carries no quat of its own, so this inserts one; the optional group
        # also replaces it if the model ever gains one.
        text, m = re.subn(rf'(<body name="base_link"\s+{re.escape(pos_attr)})(\s+quat="[^"]*")?',
                          rf'\1 {quat_attr}', text, count=1)
        if n != 1 or m != 1:
            raise RuntimeError(f"could not patch base_link transform in {self.src_model}")

        # must sit beside the original so meshdir="../meshes/stl/" still resolves
        path = self.src_model.with_name(f"pupper_v3_complete.eval_tmp_{index}.xml")
        path.write_text(text)
        (self.out_dir / f"start_pose_{index}.json").write_text(json.dumps(
            {"torso_z": z, "roll": roll, "pitch": pitch, "yaw": yaw,
             "roll_deg": np.degrees(roll), "pitch_deg": np.degrees(pitch), "yaw_deg": np.degrees(yaw),
             "clearance_lift": lift, "lowest_geom_z": lowest,
             "joints": "not randomized: the sim spawns qpos0, joint angles would need "
                       "mj_resetDataKeyframe in pupperv3_mujoco_sim"}, indent=2))
        return path

    def _restore_symlink(self, target):
        if self.install_model.is_symlink() or self.install_model.exists():
            self.install_model.unlink()
        self.install_model.symlink_to(target)

    @contextmanager
    def start_pose(self, rng, index):
        """Point the install symlink at a randomized model, then always restore it."""
        if not self.eval_cfg["randomize_start_pose"]:
            yield None
            return
        tmp = self._randomized_xml(rng, index)
        target = os.readlink(self.install_model)
        restore = atexit.register(self._restore_symlink, target)   # also covers SIGTERM paths
        try:
            self._restore_symlink(tmp)
            yield tmp
        finally:
            atexit.unregister(restore)
            self._restore_symlink(target)
            tmp.unlink(missing_ok=True)

    # ---------------- launch ----------------
    def _ros_cmd(self, command: str) -> list:
        """Run a command against the Jazzy workspace.

        Assumes we are already on the Jazzy system the workspace was built for
        (distrobox or native install). crun.sh does the environment work: drops any
        inherited Humble paths, sources Jazzy plus the workspace install, and selects
        cyclonedds -- so the command sees the graph deploy.py launches no matter what
        the calling shell had sourced.
        """
        return ["bash", str(self.crun), *shlex.split(command)]

    def _kill_launch(self, pid_file: Path):
        """Signal the launch by the PID it recorded.

        `ros2 launch` sits under the timeout/bash wrappers, so the pid returned by
        Popen is not the launch. A leftover launch tree means two controller managers
        on one DDS graph, which makes every switch_controller call ambiguous. Killing
        the recorded PID hits it exactly, with none of the collateral risk of a pkill
        pattern that could match another session's run.
        """
        if not pid_file.exists():
            return
        pid = pid_file.read_text().strip()
        # the pid itself, not the group: the wrapper bash is not a process-group leader,
        # so `kill -INT -<pid>` finds no group. `timeout` forwards INT to `ros2 launch`,
        # which then shuts its own nodes down.
        for sig in ("INT", "KILL"):
            subprocess.run(["bash", "-c", f"kill -{sig} {pid} 2>/dev/null; true"],
                           capture_output=True)
            time.sleep(8 if sig == "INT" else 1)
        pid_file.unlink(missing_ok=True)

    @contextmanager
    def launched(self, index: int):
        log = open(self.out_dir / f"ros_launch_{index}.log", "w")
        pid_file = self.out_dir / f"launch_{index}.pid"
        # setsid + recorded pgid for teardown; timeout so the launch also dies on its own
        budget = int(self.eval_cfg["ros_episode_seconds"] + self.eval_cfg["ros_launch_budget"])
        proc = subprocess.Popen(
            self._ros_cmd(
                f"bash -c 'timeout -s INT {budget} "
                "ros2 launch neural_controller launch.py sim:=True teleop:=False & "
                f"echo $! > {pid_file}; wait $!'"),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield proc
        finally:
            self._kill_launch(pid_file)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            log.close()

    # ---------------- one episode ----------------
    def episode(self, index: int, command, rng) -> dict:
        result_path = self.out_dir / f"episode_{index}.json"
        with self.start_pose(rng, index), self.launched(index):
            agent = subprocess.run(
                self._ros_cmd(
                    f"python3 {AGENT} --controller {self.eval_cfg['ros_controller']} "
                    f"--cmd {command[0]} {command[1]} {command[2]} "
                    f"--duration {self.eval_cfg['ros_episode_seconds']} "
                    f"--out {result_path}"),
                capture_output=True, text=True, timeout=self.eval_cfg["ros_episode_seconds"] + 180)
        if not result_path.exists():
            return {"error": "agent produced no result", "stderr": agent.stderr[-2000:]}
        return json.loads(result_path.read_text())


def run_episodes(monorepo, env_cfg, eval_cfg, out_dir, commands, seed=0) -> list:
    sim = RosSim(monorepo, env_cfg, eval_cfg, out_dir)
    results = []
    for i, command in enumerate(commands):
        rng = np.random.default_rng(seed + i)
        print(f"[ros] episode {i + 1}/{len(commands)}  cmd {command}")
        episode = sim.episode(i, command, rng)
        episode["episode"] = i
        results.append(episode)
        print(f"[ros]   {episode.get('error') or {k: episode[k] for k in ('survived', 'tilt_deg_max', 'frames') if k in episode}}")
    return results
