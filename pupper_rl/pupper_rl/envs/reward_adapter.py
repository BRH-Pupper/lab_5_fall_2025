"""Bridge between the env and lab_5_fall_2025/reward.py.

reward.py is loaded read-only via importlib -- it is NEVER modified.

Contract:
  - Every callable named `reward_<name>` / `_reward_<name>` in reward.py is a
    reward term. It is used only if `env.reward_scales` has `<name>`.
  - Terms are called by ARGUMENT NAME: each parameter is filled from the
    quantities the env publishes each step (see PupperMJXEnv._reward_inputs),
    else from `env.reward_params` in the config, else from the function's own
    default. A parameter that matches none of those raises at startup, not
    under jit.
"""

import importlib.util
import inspect
import sys
import warnings
from pathlib import Path

from jax import numpy as jp

PREFIXES = ("reward_", "_reward_")


class RewardAdapter:
    def __init__(self, reward_file: Path, scales: dict, params: dict):
        self.scales = scales
        self.params = {k: (jp.asarray(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}
        self.terms = self._load_terms(Path(reward_file))
        self.arg_names: dict[str, list[str]] = {}

    # ---------------- setup ----------------
    def _load_terms(self, path: Path) -> dict:
        spec = importlib.util.spec_from_file_location("lab_reward", path)
        module = importlib.util.module_from_spec(spec)
        prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ next to reward.py
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = prev

        terms = {}
        for attr in dir(module):
            for prefix in PREFIXES:
                if attr.startswith(prefix) and callable(getattr(module, attr)):
                    name = attr[len(prefix):]
                    if name in self.scales:
                        terms[name] = getattr(module, attr)
        missing_fns = set(self.scales) - set(terms)
        if missing_fns:
            warnings.warn(f"reward_scales has no matching function in {path}: {sorted(missing_fns)}")
        if not terms:
            warnings.warn(f"No matching reward terms found in {path}; reward will be 0.")
        return terms

    def bind(self, available: set[str]):
        """Resolve each term's parameters once, at env construction."""
        problems = []
        for name, fn in self.terms.items():
            args = []
            for pname, param in inspect.signature(fn).parameters.items():
                if pname in available or pname in self.params:
                    args.append(pname)
                elif param.default is inspect.Parameter.empty:
                    problems.append(f"  {name}: no value for argument '{pname}'")
            self.arg_names[name] = args
        if problems:
            raise TypeError(
                "Cannot bind reward terms. Add the value to env.reward_params in the config, "
                "or publish it from PupperMJXEnv._reward_inputs.\n" + "\n".join(problems)
                + f"\navailable from env: {sorted(available)}\nfrom reward_params: {sorted(self.params)}"
            )

    # ---------------- per step ----------------
    def __call__(self, quantities: dict) -> tuple[jp.ndarray, dict]:
        info = {}
        for name, fn in self.terms.items():
            kwargs = {a: quantities[a] if a in quantities else self.params[a] for a in self.arg_names[name]}
            info[name] = self.scales[name] * fn(**kwargs).astype(float)
        total = sum(info.values(), jp.zeros(()))
        return total, info
