"""ROS 2 sim evaluation backend: same stack deploy.py launches.

The policy runs as RTNeural inside the `neural_controller` C++ controller, so this
backend measures the deployed artifact, not the training graph.

Start-pose randomization: the sim's MuJoCo file is hardcoded in
mujoco_hardware_interface.cpp as
    <pupper_v3_description share>/description/mujoco_xml/pupper_v3_complete.xml
and colcon installed that path as a SYMLINK into the source tree. So each episode
writes a sibling copy with a randomized <key name="home" qpos=...> and repoints the
install symlink at it. The original source file is never modified, and the symlink is
always restored (context manager + finally in run_episodes).
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
    def _randomized_xml(self, rng: np.random.Generator, index: int) -> Path:
        """Sibling copy of the sim model with a randomized home keyframe."""
        r = self.env_cfg["init_randomization"]
        default = np.array(self.env_cfg["default_joint_pos"])
        lower = np.array(self.env_cfg["joint_lower_limits"])
        upper = np.array(self.env_cfg["joint_upper_limits"])

        joints = np.clip(default + rng.uniform(-r["joint_pos"], r["joint_pos"], 12), lower, upper)
        z = rng.uniform(*r["torso_z"]) + self.eval_cfg["ros_drop_margin"]  # no clearance solver here
        roll, pitch = rng.uniform(-r["roll_pitch"], r["roll_pitch"], 2)
        yaw = rng.uniform(-r["yaw"], r["yaw"])
        quat = _euler_to_quat(roll, pitch, yaw)

        qpos = " ".join(f"{v:.6f}" for v in [0.0, 0.0, z, *quat, *joints])
        ctrl = " ".join(f"{v:.6f}" for v in joints)
        text = self.src_model.read_text()
        text, n = re.subn(r'qpos="[^"]*"', f'qpos="{qpos}"', text, count=1)
        text, m = re.subn(r'ctrl="[^"]*"', f'ctrl="{ctrl}"', text, count=1)
        if n != 1 or m != 1:
            raise RuntimeError(f"could not patch keyframe in {self.src_model}")

        # must sit beside the original so meshdir="../meshes/stl/" still resolves
        path = self.src_model.with_name(f"pupper_v3_complete.eval_tmp_{index}.xml")
        path.write_text(text)
        (self.out_dir / f"start_pose_{index}.json").write_text(json.dumps(
            {"torso_z": z, "roll": roll, "pitch": pitch, "yaw": yaw, "joints": joints.tolist()}, indent=2))
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
    def _in_container(self, command: str) -> list:
        """Run inside the distrobox the workspace was built in.

        The host has ROS Humble, the workspace is Jazzy and its binaries need a newer
        libstdc++, so launching on the host dies with GLIBCXX errors and the spawners
        never come up. crun.sh strips Humble from the environment, sources Jazzy +
        the workspace, and selects cyclonedds.
        """
        parts = shlex.split(command)
        if not self.eval_cfg["ros_container"]:
            return ["bash", str(self.crun), *parts]
        return ["distrobox", "enter", self.eval_cfg["ros_container"], "--",
                "bash", str(self.crun), *parts]

    def _kill_launch(self, pid_file: Path):
        """Signal the in-container launch by the PID it recorded.

        `distrobox enter -- cmd` runs the command in the container: killing the host
        side process group leaves the in-container `ros2 launch` tree alive, and two
        controller managers on one DDS graph make every switch_controller call
        ambiguous. Killing by recorded PID avoids pkill patterns that could match
        unrelated sessions.
        """
        if not pid_file.exists():
            return
        pid = pid_file.read_text().strip()
        # signal the pid itself: the in-container bash is not a process-group leader, so
        # `kill -INT -<pid>` finds no group. `timeout` forwards INT to `ros2 launch`,
        # which then shuts its own nodes down.
        for sig in ("INT", "KILL"):
            subprocess.run(self._in_container(f"bash -c 'kill -{sig} {pid} 2>/dev/null; true'"),
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
            self._in_container(
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
                self._in_container(
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
