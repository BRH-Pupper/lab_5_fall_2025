#!/usr/bin/env python3
"""One ROS-side evaluation episode. Runs under the SYSTEM python (rclpy), not the venv.

Activates the neural controller, drives /cmd_vel with a fixed command, records the
controller's observation stream, and writes metrics as JSON.

Usage (inside a sourced ROS shell):
    python3 ros_agent.py --controller neural_controller_test --cmd 0.3 0 0 \
        --duration 10 --out episode.json
"""

import argparse
import json
import math
import time

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

SINGLE_OBS = 36  # one observation frame, see neural_controller.cpp


class EpisodeRunner(Node):
    def __init__(self, args):
        super().__init__("pupper_eval_agent")
        self.args = args
        self.frames = []            # (t, newest observation frame)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Float32MultiArray, f"/{args.controller}/observation", self._on_obs, 10)
        self.switch = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self.list = self.create_client(ListControllers, "/controller_manager/list_controllers")

    # ---------------- plumbing ----------------
    def _on_obs(self, msg):
        self.frames.append((time.time(), list(msg.data[:SINGLE_OBS])))

    def _call(self, client, request, timeout=20.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def wait_for_manager(self, timeout=60.0):
        deadline = time.time() + timeout
        for client, name in ((self.switch, "switch_controller"), (self.list, "list_controllers")):
            while not client.wait_for_service(timeout_sec=1.0):
                if time.time() > deadline:
                    raise TimeoutError(f"/controller_manager/{name} never appeared")
                rclpy.spin_once(self, timeout_sec=0.1)

    def controller_states(self) -> dict:
        result = self._call(self.list, ListControllers.Request())
        return {c.name: c.state for c in (result.controller if result else [])}

    def controller_state(self, name):
        return self.controller_states().get(name, "missing")

    def wait_until_inactive(self, name, timeout):
        """Spawners load, then configure, each controller a few seconds after launch.

        Switching while the controller is still 'missing' aborts a STRICT switch with
        'no controller with this name exists'; switching while it is 'unconfigured'
        is rejected too. Only 'inactive' can be activated.
        """
        deadline = time.time() + timeout
        while True:
            state = self.controller_state(name)
            if state == "inactive":
                return
            if state == "active":                     # already running from a previous call
                return
            if time.time() > deadline:
                raise TimeoutError(f"controller {name} stuck in state '{state}'")
            rclpy.spin_once(self, timeout_sec=0.2)

    def set_active(self, activate, deactivate):
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        result = self._call(self.switch, req)
        return bool(result and result.ok)

    # ---------------- episode ----------------
    def run(self):
        a = self.args
        self.wait_for_manager()

        self.wait_until_inactive(a.controller, a.activation_timeout)
        # STRICT fails on any name that is not currently active, so only list those
        activated = False
        for attempt in range(a.activation_attempts):
            states = self.controller_states()
            others = [c for c in a.deactivate if c != a.controller and states.get(c) == "active"]
            if self.set_active([a.controller], others):
                activated = True
                break
            self.get_logger().warn(f"switch attempt {attempt + 1} failed, states={states}")
            rclpy.spin_once(self, timeout_sec=1.0)
        if not activated:
            return {"error": f"failed to activate {a.controller}: states={self.controller_states()}"}

        deadline = time.time() + a.activation_timeout
        while self.controller_state(a.controller) != "active":
            if time.time() > deadline:
                return {"error": f"{a.controller} never reached state 'active'"}
            rclpy.spin_once(self, timeout_sec=0.1)

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = a.cmd
        t0 = time.time()
        self.frames.clear()
        while time.time() - t0 < a.duration:
            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.02)

        self.cmd_pub.publish(Twist())                       # stop commanding
        state_end = self.controller_state(a.controller)
        if state_end == "active":
            self.set_active([], [a.controller])             # leave it inactive between episodes
        return self._metrics(state_end)

    def _metrics(self, state_end):
        """Everything observable from ROS topics: no torso height, no body linear velocity."""
        a = self.args
        if not self.frames:
            return {"error": f"no frames on /{a.controller}/observation"}

        t = [f[0] - self.frames[0][0] for f in self.frames]
        obs = [f[1] for f in self.frames]
        gyro_z = [o[2] for o in obs]
        grav = [o[3:6] for o in obs]
        joints = [o[12:24] for o in obs]
        actions = [o[24:36] for o in obs]

        tilt = [math.degrees(math.acos(max(-1.0, min(1.0, -g[2])))) for g in grav]
        yaw_err = [abs(a.cmd[2] - w) for w in gyro_z]
        rate = [sum(abs(x - y) for x, y in zip(actions[i], actions[i - 1])) for i in range(1, len(actions))]

        return {
            "command": list(a.cmd),
            "duration_s": t[-1],
            "frames": len(obs),
            "controller_state_end": state_end,
            "survived": state_end == "active",
            # first frame after the controller activates: the pose the policy actually
            # inherits, which is not the spawn pose -- the sim runs uncontrolled while
            # the spawners come up, so this is what start-pose randomization delivers.
            "tilt_deg_start": tilt[0],
            "tilt_deg_mean": sum(tilt) / len(tilt),
            "tilt_deg_max": max(tilt),
            "yaw_rate_err_mean": sum(yaw_err) / len(yaw_err),
            "action_rate_mean": (sum(rate) / len(rate)) if rate else 0.0,
            "joint_dev_max": max(max(abs(v) for v in j) for j in joints),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="neural_controller_test")
    p.add_argument("--deactivate", nargs="*", default=[
        "neural_controller", "neural_controller_three_legged",
        "neural_controller_parkour", "neural_controller_test"])
    p.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   help="x vel, y vel, yaw rate")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--activation_timeout", type=float, default=45.0)
    p.add_argument("--activation_attempts", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rclpy.init()
    runner = EpisodeRunner(args)
    try:
        metrics = runner.run()
    except Exception as e:                      # report instead of crashing the orchestrator
        metrics = {"error": f"{type(e).__name__}: {e}"}
    finally:
        runner.destroy_node()
        rclpy.shutdown()

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return 0 if "error" not in metrics else 1


if __name__ == "__main__":
    raise SystemExit(main())
