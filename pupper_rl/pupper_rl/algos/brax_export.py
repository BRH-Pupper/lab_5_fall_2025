"""Export brax PPO policy params to the RTNeural JSON used by neural_controller.

Format mirrors lab_5_fall_2025/test_policy.json:
  in_shape: [None, history*36]
  layers: [{type: dense, activation, shape: [None, out], weights: [W(in x out), b(out)]}, ...]

brax specifics handled here:
  - params = (normalizer_params, policy_params, [value_params])
  - observation normalization (obs - mean) / std is baked into the first layer
  - last layer outputs [loc, scale] (2 * act_dim); keep loc, apply tanh
    (brax NormalTanhDistribution mode == tanh(loc))
"""

import json

import numpy as np


def export_policy_json(params, env_cfg: dict, policy_cfg: dict, export_cfg: dict, path: str) -> str:
    normalizer, policy_params = params[0], params[1]
    mlp = policy_params["params"]
    names = sorted(mlp.keys(), key=lambda n: int(n.split("_")[-1]))  # hidden_0, hidden_1, ...

    mean = np.asarray(normalizer.mean)
    std = np.asarray(normalizer.std)

    layers = []
    for i, name in enumerate(names):
        W = np.asarray(mlp[name]["kernel"], dtype=np.float64)  # flax kernel: (in, out)
        b = np.asarray(mlp[name]["bias"], dtype=np.float64)
        if i == 0:
            b = b - (mean / std) @ W
            W = W / std[:, None]
        last = i == len(names) - 1
        if last:
            act_dim = W.shape[1] // 2
            W, b = W[:, :act_dim], b[:act_dim]
        layers.append({
            "type": "dense",
            "activation": "tanh" if last else policy_cfg["activation"],
            "shape": [None, int(W.shape[1])],
            "weights": [W.tolist(), b.tolist()],
        })

    out = {
        "use_imu": export_cfg["use_imu"],
        "control_orientation": export_cfg["control_orientation"],
        "observation_history": env_cfg["observation_history"],
        "action_scale": env_cfg["action_scale"],
        "kp": export_cfg["kp"],
        "kd": export_cfg["kd"],
        "default_joint_pos": env_cfg["default_joint_pos"],
        "joint_upper_limits": env_cfg["joint_upper_limits"],
        "joint_lower_limits": env_cfg["joint_lower_limits"],
        "in_shape": [None, int(mean.shape[0])],
        "layers": layers,
    }
    with open(path, "w") as f:
        json.dump(out, f)
    return path
