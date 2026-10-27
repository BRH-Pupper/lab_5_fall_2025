#!/usr/bin/env python3
"""Evaluate a trained policy and log the runs to W&B against their model.

Model retrieval mirrors ../deploy.py: it prompts for a W&B run number and calls
../download_latest_policy.py, so the evaluated file is the same test_policy.json the
robot would load.

Backends:
    ros   ros2 launch neural_controller launch.py sim:=True, policy runs as RTNeural
          in C++ (deploy stack). Start pose randomized per episode.
    mjx   not implemented yet.

Usage:
    python eval.py --backend ros                 # prompts for a run number
    python eval.py --backend ros --run 42        # non-interactive
    python eval.py --backend ros --skip-download # evaluate whatever policy is deployed
"""

import argparse
import filecmp
import importlib.util
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_SCRIPT = LAB_DIR / "download_latest_policy.py"
# how download_latest_policy.py names what it fetches: policy_<run name>_max_reward_<r>.json
POLICY_NAME = re.compile(r"^policy_(?P<run>.+)_max_reward_.*\.json$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs/ppo_default.yaml"))
    p.add_argument("--backend", choices=["ros", "mjx"], default="ros")
    p.add_argument("--run", type=int, default=None, help="W&B run number; skips the prompt")
    p.add_argument("--skip-download", action="store_true", help="use the already deployed policy")
    p.add_argument("--episodes", type=int, default=None, help="override eval.episodes")
    p.add_argument("overrides", nargs="*", help="dotted key=value config overrides")
    return p.parse_args()


def download_policy(run_number):
    """Same flow as deploy.py: optional run number, then download_latest_policy.py."""
    if run_number is None:
        print("Enter run number to download a specific policy")
        print("(Press Enter to keep the current policy)")
        try:
            raw = input("Run number: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled by user")
            return None, False
        if not raw:
            print("No run number provided. Using current policy.")
            return None, True
        run_number = int(raw)

    print(f"Downloading policy from run {run_number}...")
    result = subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT), "--run_number", str(run_number)],
                            cwd=str(LAB_DIR), check=False)
    return run_number, result.returncode == 0


def summarize(results):
    """mean/std of every numeric field, plus how many episodes errored."""
    ok = [r for r in results if "error" not in r]
    summary = {"episodes": len(results), "failed": len(results) - len(ok)}
    if not ok:
        return summary
    for key in sorted(k for k in ok[0] if isinstance(ok[0][k], (int, float, bool))):
        values = [float(r[key]) for r in ok if key in r]
        summary[f"{key}_mean"] = statistics.fmean(values)
        if len(values) > 1:
            summary[f"{key}_std"] = statistics.stdev(values)
    return summary


def _lab_module(name):
    """Import a sibling lab script read-only (it is never modified, no __pycache__)."""
    spec = importlib.util.spec_from_file_location(name, LAB_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prev
    return module


def deployed_run_name(model_name="test_policy.json"):
    """Which W&B run produced the policy that is deployed right now.

    download_latest_policy.py downloads policy_<run>_max_reward_<r>.json and copies it
    to test_policy.json, so the run name is recoverable from whichever download is
    byte-identical to the deployed copy. Identical beats newest-mtime: a stale download
    from another run sitting in the same folder would otherwise win.
    """
    model_dir = _lab_module("download_latest_policy").resolve_model_dir()
    deployed = model_dir / model_name
    candidates = [(p, m.group("run")) for p in model_dir.glob("policy_*.json")
                  if (m := POLICY_NAME.match(p.name))]
    if not candidates:
        return None
    if deployed.is_file():
        same = [name for path, name in candidates if filecmp.cmp(path, deployed, shallow=False)]
        if len(same) == 1:
            return same[0]
        if len(same) > 1:   # the same policy downloaded twice; any of them names the run
            return same[0]
    newest = max(candidates, key=lambda c: c[0].stat().st_mtime)
    print(f"[wandb] {model_name} matches no download; guessing from newest file: {newest[0].name}")
    return newest[1]


def find_run(api, entity, project, number=None, name=None):
    """The training run, by exact name or by the `-<N>` suffix download_latest_policy.py uses."""
    runs = api.runs(f"{entity}/{project}")
    if name is not None:
        matches = [r for r in runs if r.name == name]
        if matches:
            return matches[0]
        number = int(name.rsplit("-", 1)[-1]) if name.rsplit("-", 1)[-1].isdigit() else number
    if number is None:
        return None
    matches = [r for r in runs if r.name.endswith(f"-{number}")]
    return matches[0] if matches else None


def log_to_wandb(cfg, results, summary, meta):
    w = cfg["wandb"]
    if not w["enabled"] or w["mode"] == "disabled":
        print("W&B logging disabled")
        return
    import wandb

    api = wandb.Api()
    entity = w["entity"] or api.default_entity
    name = None if meta.get("source_run_number") else deployed_run_name()
    target = find_run(api, entity, w["project"], meta.get("source_run_number"), name)
    if target is None:
        print("[wandb] could not identify the run that produced this policy; "
              "nothing logged. Pass --run <N> to say which it is.")
        return

    # resume the producing run: the eval belongs to the policy, not to a run of its own.
    # No job_type and no use_artifact -- that run already owns the artifact it produced.
    # Namespace per backend: brax's own evaluator already owns `eval/*` on this run
    # (eval/episode_reward, eval/avg_episode_length, ...), so `<backend>_eval/*` keeps
    # the deployed-stack numbers separate and comparable across backends.
    prefix = f"{meta['backend']}_eval"
    artifacts = [a.name for a in target.logged_artifacts() if ".json" in a.name]
    run = wandb.init(project=w["project"], entity=entity, mode=w["mode"],
                     id=target.id, resume="allow")
    columns = sorted({k for r in results for k in r})
    table = wandb.Table(columns=columns,
                        data=[[r.get(c) for c in columns] for r in results])
    run.log({f"{prefix}/episodes": table})
    run.summary.update({
        **{f"{prefix}/{k}": v for k, v in summary.items()},
        f"{prefix}/meta": {**meta, "run": target.name,
                           "policy_artifact": artifacts[0] if artifacts else None,
                           "at": datetime.now().isoformat(timespec="seconds"), "config": cfg["eval"]},
    })
    print(f"logged eval into {target.name}: {run.url}")
    run.finish()


def main():
    args = parse_args()
    from pupper_rl.utils.config import load_config, resolve_path
    cfg = load_config(args.config, args.overrides)
    if args.episodes:
        cfg["eval"]["episodes"] = args.episodes

    if args.backend == "mjx":
        print("mjx backend not implemented yet")
        return 2

    run_number = args.run if args.skip_download else None
    if not args.skip_download:
        run_number, ok = download_policy(args.run)
        if not ok:
            print("Policy download failed; aborting.")
            return 1

    out_dir = Path(cfg["logging"]["output_root"]) / f"eval_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    e_cfg = cfg["eval"]
    commands = [list(c) for c in e_cfg["commands"]][:e_cfg["episodes"]]
    monorepo = resolve_path(cfg, e_cfg["monorepo"])

    from pupper_rl.eval import ros_backend
    results = ros_backend.run_episodes(monorepo, cfg["env"], e_cfg, out_dir, commands, seed=cfg["seed"])

    summary = summarize(results)
    print("\n".join(f"{k}: {v}" for k, v in summary.items()))
    log_to_wandb(cfg, results, summary, {
        "backend": "ros",
        "source_run_number": run_number,
        "out_dir": str(out_dir),
    })
    print(f"\nartifacts in {out_dir}")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
