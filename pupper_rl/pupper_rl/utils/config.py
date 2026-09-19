"""YAML config loading with dotted CLI overrides (e.g. ppo.lr=1e-4)."""

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
