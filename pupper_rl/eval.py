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
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_SCRIPT = LAB_DIR / "download_latest_policy.py"


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


def log_to_wandb(cfg, results, summary, meta):
    w = cfg["wandb"]
    if not w["enabled"] or w["mode"] == "disabled":
        print("W&B logging disabled")
        return
    import wandb

    run = wandb.init(project=w["project"], entity=w["entity"], mode=w["mode"],
                     job_type="eval", config={**meta, "eval": cfg["eval"]})
    if meta.get("source_artifact"):
        try:
            run.use_artifact(meta["source_artifact"])   # model -> eval lineage edge
        except Exception as e:
            print(f"could not link artifact {meta['source_artifact']}: {e}")

    columns = sorted({k for r in results for k in r})
    table = wandb.Table(columns=columns,
                        data=[[r.get(c) for c in columns] for r in results])
    wandb.log({"eval/episodes": table, **{f"eval/{k}": v for k, v in summary.items()}})
    print(f"logged to {run.url}")
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

    run_number = None
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
        "source_artifact": e_cfg["source_artifact"] or None,
        "out_dir": str(out_dir),
    })
    print(f"\nartifacts in {out_dir}")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
