#!/usr/bin/env python3
"""
Deployment and rebuild script for the neural controller.

This script:
1. Copies configuration files from lab_5_fall_2025 to the pupperv3-monorepo
2. Rebuilds the ROS2 workspace using build.sh
3. Prompts for Weights & Biases (wandb) authentication

Usage:
    python3 rebuild_neural_controller.py              # Deploy, rebuild, and setup wandb
    python3 rebuild_neural_controller.py --dry-run    # Show what would be done without doing it
    python3 rebuild_neural_controller.py --no-build   # Deploy only, skip rebuild and wandb
    python3 rebuild_neural_controller.py --monorepo /path/to/pupperv3-monorepo

Paths are resolved relative to this script, so the lab folder can live anywhere.
The monorepo is looked up in this order: --monorepo, $PUPPER_MONOREPO,
/home/pi/pupperv3-monorepo (robot), a sibling of this folder (dev machine).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# This script lives in the lab folder; everything is resolved from there
LAB_DIR = Path(__file__).resolve().parent

# Where pupperv3-monorepo may live, most specific first
MONOREPO_CANDIDATES = [
    Path("/home/pi/pupperv3-monorepo"),   # robot
    LAB_DIR.parent / "pupperv3-monorepo",  # dev machine (sibling of the lab folder)
]


def resolve_monorepo(monorepo=None):
    """Pick the monorepo root: --monorepo > $PUPPER_MONOREPO > robot path > sibling folder."""
    explicit = monorepo or os.environ.get("PUPPER_MONOREPO")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else (LAB_DIR / path).resolve()

    for candidate in MONOREPO_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    return MONOREPO_CANDIDATES[-1].resolve()  # report the dev-machine path in the error


def build_file_mappings(monorepo):
    """Source (lab folder) -> destination (monorepo) for every deployed file."""
    launch_dir = monorepo / "ros2_ws/src/neural_controller/launch"
    return [
        {
            "source": LAB_DIR / "config.yaml",
            "destination": launch_dir / "config.yaml",
            "description": "Neural controller configuration file"
        },
        {
            "source": LAB_DIR / "launch.py",
            "destination": launch_dir / "launch.py",
            "description": "Neural controller launch file"
        },
        {
            "source": LAB_DIR / "estop_controller.cpp",
            "destination": monorepo / "ros2_ws/src/joy_utils/src/estop_controller.cpp",
            "description": "Emergency stop controller C++ source file"
        },
        {
            "source": LAB_DIR / "parkour_policy.json",
            "destination": launch_dir / "parkour_policy.json",
            "description": "Parkour neural network policy file"
        },
        {
            "source": LAB_DIR / "test_policy.json",
            "destination": launch_dir / "test_policy.json",
            "description": "Place Holder test policy file"
        },
    ]


def create_backup(file_path):
    """Create a backup of an existing file"""
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        shutil.copy2(file_path, backup_path)
        return backup_path
    return None


def deploy_files(monorepo, dry_run=False):
    """Deploy all files to their destinations"""
    
    print("=" * 70)
    print("Neural Controller Deployment & Rebuild Script")
    print("=" * 70)
    print(f"📁 Lab folder: {LAB_DIR}")
    print(f"📁 Monorepo:   {monorepo}")
    
    if not monorepo.is_dir():
        print(f"\n❌ ERROR: Monorepo not found: {monorepo}")
        print("   Pass --monorepo /path/to/pupperv3-monorepo or set $PUPPER_MONOREPO")
        return False
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified\n")
    
    success_count = 0
    error_count = 0
    
    for mapping in build_file_mappings(monorepo):
        source = mapping["source"]
        destination = mapping["destination"]
        description = mapping["description"]
        
        print(f"\n📄 {description}")
        print(f"   Source:      {source}")
        print(f"   Destination: {destination}")
        
        # Check if source exists
        if not source.exists():
            print(f"   ❌ ERROR: Source file does not exist!")
            error_count += 1
            continue
        
        # Check if destination directory exists
        if not destination.parent.exists():
            print(f"   ❌ ERROR: Destination directory does not exist!")
            error_count += 1
            continue
        
        if dry_run:
            if destination.exists():
                print(f"   ℹ️  Would replace existing file (backup would be created)")
            else:
                print(f"   ℹ️  Would create new file")
            success_count += 1
            continue
        
        try:
            # Create backup if destination exists
            backup_path = create_backup(destination)
            if backup_path:
                print(f"   💾 Backup created: {backup_path.name}")
            
            # Copy the file
            shutil.copy2(source, destination)
            print(f"   ✅ Successfully deployed!")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            error_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Deployment Summary")
    print("=" * 70)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors:     {error_count}")
    
    if dry_run:
        print("\nℹ️  This was a dry run. Run without --dry-run to actually deploy.")
    
    return error_count == 0


def rebuild_workspace(monorepo, dry_run=False):
    """Rebuild the ROS2 workspace"""
    
    print("\n" + "=" * 70)
    print("Rebuilding ROS2 Workspace")
    print("=" * 70)
    
    ros2_ws = monorepo / "ros2_ws"
    build_script = ros2_ws / "build.sh"
    
    if not build_script.exists():
        print(f"❌ ERROR: Build script not found: {build_script}")
        return False
    
    if dry_run:
        print(f"ℹ️  Would execute: {build_script}")
        return True
    
    print(f"🔨 Running build script: {build_script}")
    print(f"📁 Working directory: {ros2_ws}")
    print()
    
    try:
        # Run build.sh from the ros2_ws directory
        result = subprocess.run(
            ["bash", str(build_script)],
            cwd=str(ros2_ws),
            check=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Build completed successfully!")
            return True
        else:
            print(f"\n❌ Build failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR during build: {e}")
        return False


def wandb_login():
    """Prompt for Weights & Biases authentication"""
    
    print("\n" + "=" * 70)
    print("Weights & Biases (wandb) Authentication")
    print("=" * 70)
    
    try:
        # Check if wandb is installed
        check_wandb = subprocess.run(
            ["python3", "-c", "import wandb"],
            capture_output=True,
            text=True
        )
        
        if check_wandb.returncode != 0:
            print("⚠️  wandb is not installed. Attempting to install...")
            print()
            
            # Try to install wandb
            install_result = subprocess.run(
                ["pip3", "install", "wandb"],
                check=False
            )
            
            if install_result.returncode != 0:
                print("\n❌ Failed to install wandb automatically")
                print("   You can install it manually with: pip3 install wandb")
                return True
            
            print("\n✅ wandb installed successfully!")
        
        print("\n🔐 Please authenticate your wandb account")
        print("   This will allow logging of training runs and experiments")
        print("   Visit https://wandb.ai/authorize to get your API key")
        print()
        
        # Run wandb login which will prompt for API key
        result = subprocess.run(
            ["wandb", "login"],
            check=False
        )
        
        if result.returncode == 0:
            print("\n✅ wandb authentication successful!")
            return True
        else:
            print("\n⚠️  wandb authentication was skipped or failed")
            print("   You can run 'wandb login' manually later")
            return True  # Don't fail the entire script for this
            
    except Exception as e:
        print(f"\n⚠️  Could not complete wandb authentication: {e}")
        print("   You can run 'wandb login' manually later")
        return True  # Don't fail the entire script for this


def _arg_value(flag):
    """Read `--flag value` or `--flag=value` from sys.argv; None if absent."""
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def main():
    """Main entry point"""
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    skip_build = "--no-build" in sys.argv
    monorepo = resolve_monorepo(_arg_value("--monorepo"))
    
    # Step 1: Deploy files
    deploy_success = deploy_files(monorepo, dry_run)
    
    if not deploy_success:
        print("\n❌ Deployment failed. Aborting.")
        return 1
    
    # Step 2: Rebuild workspace (unless skipped or dry run)
    if skip_build:
        print("\nℹ️  Skipping rebuild (--no-build flag)")
        return 0
    
    if not dry_run:
        rebuild_success = rebuild_workspace(monorepo, dry_run)
        if not rebuild_success:
            print("\n❌ Rebuild failed.")
            return 1
        
        # Step 3: wandb authentication after successful rebuild
        wandb_login()
    else:
        rebuild_workspace(monorepo, dry_run)
    
    print("\n" + "=" * 70)
    print("✅ All operations completed successfully!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

