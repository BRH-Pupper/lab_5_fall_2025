"""Brax/MJX env: Pupper v3 velocity tracking on GPU.

Observation layout mirrors neural_controller.cpp exactly (36 per frame):
  [0:3]   base angular velocity (gyro, body frame)
  [3:6]   projected gravity (body frame)
  [6:9]   commands vx, vy, yaw_rate
  [9:12]  desired world z in body frame (flat: [0, 0, 1])
  [12:24] joint_pos - default_joint_pos
  [24:36] last action
History is stacked newest-first (robot rotates buffer right each step).

Reward terms come from lab_5_fall_2025/reward.py and are called by argument
name; everything they can ask for is published in `_reward_inputs`.
"""

import jax
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp

from ..utils.config import init_ranges, worst_tilt
from .reward_adapter import RewardAdapter

SINGLE_OBS = 36
NUM_JOINTS = 12
FOOT_SITES = ["leg_front_r_3_foot_site", "leg_front_l_3_foot_site",
              "leg_back_r_3_foot_site", "leg_back_l_3_foot_site"]
LOWER_LEG_BODIES = ["leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3"]


def _euler_to_quat(rpy: jax.Array) -> jax.Array:
    """roll-pitch-yaw (ZYX) -> MuJoCo quaternion [w, x, y, z]."""
    cr, cp, cy = jp.cos(rpy / 2)
    sr, sp, sy = jp.sin(rpy / 2)
    return jp.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


class PupperMJXEnv(PipelineEnv):
    def __init__(self, env_cfg: dict, model_xml, reward_file):
        e = env_cfg
        mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
        n_frames = max(1, round(e["ctrl_dt"] / mj_model.opt.timestep))
        super().__init__(mjcf.load_model(mj_model), backend="mjx", n_frames=n_frames)

        self.history = e["observation_history"]
        self.action_scale = e["action_scale"]
        self.terminal_body_z = e["terminal_body_z"]
        self.cos_terminal_angle = float(np.cos(e["terminal_body_angle"]))
        self.foot_contact_height = e["foot_contact_height"]
        self.default_pos = jp.array(e["default_joint_pos"])
        self.lower = jp.array(e["joint_lower_limits"])
        self.upper = jp.array(e["joint_upper_limits"])
        r = e["command_ranges"]
        self.cmd_low = jp.array([r["lin_vel_x"][0], r["lin_vel_y"][0], r["ang_vel_yaw"][0]])
        self.cmd_high = jp.array([r["lin_vel_x"][1], r["lin_vel_y"][1], r["ang_vel_yaw"][1]])

        self._imu_site = mj_model.site("body_imu_site").id
        self._gyro_adr = mj_model.sensor_adr[mj_model.sensor("body_gyro").id]
        self._feet_site_id = np.array([mj_model.site(s).id for s in FOOT_SITES])
        self._lower_leg_body_id = np.array([mj_model.body(b).id for b in LOWER_LEG_BODIES])
        self._collision_geom_ids = np.array(
            [g for body in e["collision_bodies"]
             for g in range(mj_model.body(body).geomadr[0],
                            mj_model.body(body).geomadr[0] + mj_model.body(body).geomnum[0])],
            dtype=int,
        )
        home = mj_model.key_qpos[0] if mj_model.nkey else mj_model.qpos0
        self._init_q = jp.array(home).at[7:7 + NUM_JOINTS].set(self.default_pos)

        # start-pose randomization (all collision geoms are used for the ground-clearance lift)
        self._collision_geoms = np.flatnonzero(mj_model.geom_conaffinity * (mj_model.geom_bodyid > 0))
        self._collision_half_extent = jp.array(mj_model.geom_size[self._collision_geoms].max(axis=1))
        self.rand = dict(e["init_randomization"])
        self.tilt = None
        if self.rand["enabled"]:
            self.tilt = init_ranges(self.rand)
            joint_margin = float(jp.min(jp.minimum(self.default_pos - self.lower, self.upper - self.default_pos)))
            assert self.rand["joint_pos"] < joint_margin, (
                f"joint_pos {self.rand['joint_pos']} exceeds the tightest joint margin {joint_margin:.3f} rad")
            # roll and pitch compound, so check the tilt they can reach together, not
            # each axis on its own: a reset past terminal_body_angle is `done` on step 1.
            reach = worst_tilt(self.tilt)
            assert reach < 0.8 * e["terminal_body_angle"], (
                f"roll_deg {self.rand.get('roll_deg', self.rand.get('roll_pitch_deg'))} + "
                f"pitch_deg {self.rand.get('pitch_deg', self.rand.get('roll_pitch_deg'))} reach "
                f"{np.degrees(reach):.1f} deg ({reach:.3f} rad) of body tilt, too close to "
                f"terminal_body_angle {e['terminal_body_angle']} rad "
                f"({np.degrees(e['terminal_body_angle']):.1f} deg)")
            z_lo, z_hi = self.rand["torso_z"]
            assert z_lo > e["terminal_body_z"] + 0.01, f"torso_z lower bound {z_lo} too close to terminal_body_z"
            assert z_hi < 0.30, f"torso_z upper bound {z_hi} is a drop test, not a pose"

        self.reward_fn = RewardAdapter(reward_file, e["reward_scales"], e["reward_params"])
        self.reward_fn.bind(set(self._probe_inputs()))  # fail fast on unbindable arguments

        # Height Check flag
        self._hhc_done = False
        self.half_height = e["half_height"]
        self.full_height = e["full_height"]

    # ---------------- core API ----------------
    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        ps = self._initial_state(rng)

        info = {
            "rng": rng,
            "command": jax.random.uniform(key, (3,), minval=self.cmd_low, maxval=self.cmd_high),
            "last_act": jp.zeros(NUM_JOINTS),
            "last_joint_vel": jp.zeros(NUM_JOINTS),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "step": jp.zeros((), dtype=jp.int32),
            "obs_history": jp.zeros(SINGLE_OBS * self.history),
            "h_height": jp.zeros((), dtype=bool),
        }
        # fill the whole history with the first frame instead of a fake upright one
        info["obs_history"] = jp.tile(self._single_obs(ps, info), self.history)
        obs = info["obs_history"]
        metrics = {f"reward/{k}": jp.zeros(()) for k in self.reward_fn.terms}
        return State(ps, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ---------------- start pose ----------------
    def _initial_state(self, rng: jax.Array):
        """Nominal pose, or a randomized one lifted clear of the ground and settled."""
        if not self.rand["enabled"]:
            return self.pipeline_init(self._init_q, jp.zeros(self.sys.nv))

        r = self.rand
        k_jp, k_jv, k_rp, k_yaw, k_z, k_lv, k_av = jax.random.split(rng, 7)

        joints = jp.clip(
            self.default_pos
            + jax.random.uniform(k_jp, (NUM_JOINTS,), minval=-r["joint_pos"], maxval=r["joint_pos"]),
            self.lower, self.upper)
        rpy = jp.concatenate([
            jax.random.uniform(k_rp, (2,),
                               minval=jp.array([self.tilt["roll"][0], self.tilt["pitch"][0]]),
                               maxval=jp.array([self.tilt["roll"][1], self.tilt["pitch"][1]])),
            jax.random.uniform(k_yaw, (1,), minval=self.tilt["yaw"][0], maxval=self.tilt["yaw"][1]),
        ])
        z = jax.random.uniform(k_z, (), minval=r["torso_z"][0], maxval=r["torso_z"][1])

        q = self._init_q.at[7:7 + NUM_JOINTS].set(joints).at[2].set(z).at[3:7].set(_euler_to_quat(rpy))
        qd = jp.concatenate([
            jax.random.uniform(k_lv, (3,), minval=-r["lin_vel"], maxval=r["lin_vel"]),
            jax.random.uniform(k_av, (3,), minval=-r["ang_vel"], maxval=r["ang_vel"]),
            jax.random.uniform(k_jv, (NUM_JOINTS,), minval=-r["joint_vel"], maxval=r["joint_vel"]),
        ])

        # lift so the lowest collision geom clears the ground (tilt can bury the feet)
        ps = self.pipeline_init(q, qd)
        lowest = jp.min(ps.geom_xpos[self._collision_geoms, 2] - self._collision_half_extent)
        lift = jp.maximum(0.0, r["ground_clearance"] - lowest)
        ps = self.pipeline_init(q.at[2].add(lift), qd)

        # hold the default pose for a moment so episodes start standing, not falling
        for _ in range(r["settle_steps"]):
            ps = self.pipeline_step(ps, self.default_pos)
        return ps

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        target = jp.clip(self.default_pos + self.action_scale * action, self.lower, self.upper)
        ps = self.pipeline_step(state.pipeline_state, target)  # position actuators in MJCF

        # foot contacts and air time (feet sites near the ground count as contact)
        contact = ps.site_xpos[self._feet_site_id][:, 2] < self.foot_contact_height
        contact_filt = jp.logical_or(contact, state.info["last_contact"])
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt
        air_time = state.info["feet_air_time"] + self.dt


        #Height Checks
        h = ps.qpos[2]
        latched = state.info["h_height"]

        hh_done = (h >= self.half_height) & ~latched      # True on exactly one step
        fh_done = h >= self.full_height
        done = jp.logical_or(self._fallen(ps), fh_done)

        inputs = self._reward_inputs(ps, state.info, action, done, contact_filt, first_contact, air_time, hh_done, fh_done)
        reward, terms = self.reward_fn(inputs)

        info = {
            **state.info,
            "last_act": action,
            "last_joint_vel": ps.qvel[6:6 + NUM_JOINTS],
            "feet_air_time": air_time * ~contact_filt,
            "last_contact": contact,
            "step": state.info["step"] + 1,
            "h_height": (latched | hh_done) & ~done,
        }
        obs = self._get_obs(ps, info)
        info["obs_history"] = obs
        # keep keys added by brax wrappers (pytree structure must match reset)
        metrics = {**state.metrics, **{f"reward/{k}": v for k, v in terms.items()}}
        return state.replace(pipeline_state=ps, obs=obs, reward=reward,
                             done=done.astype(jp.float32), metrics=metrics, info=info)

    # ---------------- reward plumbing ----------------
    def _reward_inputs(self, ps, info, action, done, contact_filt, first_contact, air_time, hh_done, fh_done) -> dict:
        """Everything reward.py functions may ask for, keyed by argument name."""
        return {
            "pipeline_state": ps,
            "x": ps.x,
            "xd": ps.xd,
            "commands": info["command"],
            "torques": ps.actuator_force,
            "joint_angles": ps.qpos[7:7 + NUM_JOINTS],
            "joint_vel": ps.qvel[6:6 + NUM_JOINTS],
            "velocities": ps.qvel[6:6 + NUM_JOINTS],
            "last_joint_vel": info["last_joint_vel"],
            "act": action,
            "last_act": info["last_act"],
            "dt": self.dt,
            "default_pose": self.default_pos,
            "done": done,
            "step": info["step"],
            "air_time": air_time,
            "first_contact": first_contact,
            "contact_filt": contact_filt,
            "feet_site_id": self._feet_site_id,
            "lower_leg_body_id": self._lower_leg_body_id,
            "geom_ids": self._collision_geom_ids,
            "hh_done": hh_done,
            "fh_done": fh_done,
        }

    def _probe_inputs(self) -> dict:
        """Real (unused) inputs built once at construction, so bind() sees true names."""
        info = {
            "command": jp.zeros(3),
            "last_joint_vel": jp.zeros(NUM_JOINTS),
            "last_act": jp.zeros(NUM_JOINTS),
            "step": jp.zeros((), dtype=jp.int32),
        }
        return self._reward_inputs(
            self.pipeline_init(self._init_q, jp.zeros(self.sys.nv)), info,
            jp.zeros(NUM_JOINTS), jp.zeros((), dtype=bool), jp.zeros(4, dtype=bool),
            jp.zeros(4), jp.zeros(4),jp.zeros((), dtype=bool), jp.zeros((), dtype=bool),
        )

    # ---------------- helpers ----------------
    def _body_rot(self, ps):
        return ps.site_xmat[self._imu_site].reshape(3, 3)  # body->world

    def _gyro(self, ps):
        return ps.sensordata[self._gyro_adr:self._gyro_adr + 3]

    def _projected_gravity(self, ps):
        return self._body_rot(ps).T @ jp.array([0.0, 0.0, -1.0])

    def _fallen(self, ps):
        tipped = -self._projected_gravity(ps)[2] < self.cos_terminal_angle
        return jp.logical_or(ps.qpos[2] < self.terminal_body_z, tipped)
     
    def _h_height(self, ps):
        if not self._hhc_done:
            print(ps.qpos[2])
            if (ps.qpos[2] >= 100) & (ps.qpos[2] < 200):
            # if ps.qpos[2] >= 100:
                self._hhc_done = True
                return jp.array([True])
        return False

    def _f_height(self, ps):
        if ps.qpos[2] >= 200:
            return jp.array([True])


    def _single_obs(self, ps, info):
        """One 36-value observation frame, same order as neural_controller.cpp."""
        return jp.concatenate([
            self._gyro(ps),
            self._projected_gravity(ps),
            info["command"],
            self._body_rot(ps).T @ jp.array([0.0, 0.0, 1.0]),
            ps.qpos[7:7 + NUM_JOINTS] - self.default_pos,
            info["last_act"],
        ])

    def _get_obs(self, ps, info):
        # newest frame first, drop oldest
        return jp.concatenate([self._single_obs(ps, info), info["obs_history"][:-SINGLE_OBS]])
