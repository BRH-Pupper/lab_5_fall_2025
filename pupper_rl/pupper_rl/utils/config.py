"""YAML config loading with dotted CLI overrides (e.g. ppo.lr=1e-4)."""

import math
from pathlib import Path
from types import SimpleNamespace

import yaml


def _set_dotted(cfg: dict, key: str, value):
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key, yaml.safe_load(raw))
    cfg["_config_dir"] = str(Path(path).resolve().parent)
    return cfg


def to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    return d


def resolve_path(cfg: dict, p: str) -> Path:
    """Resolve a path relative to the project root (parent of configs/)."""
    p = Path(p)
    return p if p.is_absolute() else (Path(cfg["_config_dir"]).parent / p).resolve()


def as_range(value, name: str) -> tuple[float, float]:
    """`x` -> (-x, x), `[lo, hi]` -> (lo, hi).

    Start-pose knobs accept either spelling so a symmetric band stays a single number
    while an asymmetric one (tilt onto one side, say) is explicit.
    """
    if isinstance(value, (int, float)):
        lo, hi = -float(value), float(value)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be a number or [lo, hi], got {value!r}")
        lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(f"{name} range is inverted: [{lo}, {hi}]")
    return lo, hi


def init_ranges(rand: dict) -> dict[str, tuple[float, float]]:
    """Start-pose angle ranges in RADIANS, from `*_deg` config keys.

    Config states body angles in degrees because that is how they get reasoned about
    ("start it tilted 20 degrees"); everything downstream -- quaternions, projected
    gravity, terminal_body_angle -- is radians. Converting here keeps both backends on
    one definition. `roll_pitch_deg` sets both axes; an explicit axis key wins.
    """
    legacy = [k for k in ("roll", "pitch", "yaw", "roll_pitch") if k in rand]
    if legacy:
        raise KeyError(
            f"init_randomization keys {legacy} are now in degrees and renamed: use "
            "roll_deg / pitch_deg / yaw_deg (or roll_pitch_deg for both axes)")

    out = {}
    both = rand.get("roll_pitch_deg")
    for axis in ("roll", "pitch"):
        source = rand[f"{axis}_deg"] if rand.get(f"{axis}_deg") is not None else both
        if source is None:
            raise KeyError(f"init_randomization needs '{axis}_deg' (or 'roll_pitch_deg' for both)")
        lo, hi = as_range(source, f"init_randomization.{axis}_deg")
        out[axis] = (math.radians(lo), math.radians(hi))
    lo, hi = as_range(rand["yaw_deg"], "init_randomization.yaw_deg")
    out["yaw"] = (math.radians(lo), math.radians(hi))
    return out


def worst_tilt(ranges: dict) -> float:
    """Largest body tilt the roll/pitch ranges can produce, in rad.

    Takes the radian ranges init_ranges returns. Not max(|roll|, |pitch|): roll and
    pitch compound. Tilt is the angle between body +z and world +z,
    cos(tilt) = cos(roll) * cos(pitch), maximal at whichever range endpoints are
    furthest from zero.
    """
    r = max(abs(ranges["roll"][0]), abs(ranges["roll"][1]))
    p = max(abs(ranges["pitch"][0]), abs(ranges["pitch"][1]))
    return math.acos(max(-1.0, min(1.0, math.cos(r) * math.cos(p))))
