#!/usr/bin/env python3
"""
Real-time Training Monitor for GPU Curriculum Training
Monitors progress of both FULL and CLASSIC curriculum runs
"""

import os
import time
import glob
from pathlib import Path
import pandas as pd
from datetime import datetime

def check_training_progress(runs_dir="./runs_20250824_203859"):
    """Monitor both FULL and CLASSIC curriculum training progress."""
    print(f"🎯 GPU Training Monitor - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    full_dir = Path(runs_dir) / "FULL_curriculum"
    classic_dir = Path(runs_dir) / "CLASSIC_curriculum"

    # Check FULL curriculum
    print("📊 FULL CURRICULUM (Box2D):")
    if full_dir.exists():
        check_curriculum_status(full_dir, "FULL")
    else:
        print("   📂 Directory not created yet...")

    print()

    # Check CLASSIC curriculum
    print("📊 CLASSIC CURRICULUM (Classic Control):")
    if classic_dir.exists():
        check_curriculum_status(classic_dir, "CLASSIC")
    else:
        print("   📂 Directory not created yet...")

    print("=" * 70)

def check_curriculum_status(curriculum_dir, name):
    """Check status of a specific curriculum."""
    log_files = list(curriculum_dir.glob("**/curriculum_progress.csv"))

    if not log_files:
        # Check for environment-specific directories
        env_dirs = [d for d in curriculum_dir.iterdir() if d.is_dir()]
        if env_dirs:
            print(f"   🏗️  Training in progress... ({len(env_dirs)} env dirs created)")
            for env_dir in sorted(env_dirs):
                phases = check_environment_phases(env_dir)
                if phases:
                    print(f"   📁 {env_dir.name}: {phases}")
        else:
            print("   🚀 Starting up...")
        return

    # Read latest progress
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    try:
        df = pd.read_csv(latest_log)
        if len(df) > 0:
            latest = df.iloc[-1]
            env_id = latest.get('env_id', 'Unknown')
            generation = latest.get('generation', 0)
            mean_score = latest.get('mean_score', 0.0)
            threshold = latest.get('threshold', 0.0)
            solved = latest.get('solved', False)

            status = "✅ SOLVED" if solved else "🎮 Training"
            progress = (mean_score / threshold * 100) if threshold > 0 else 0

            print(f"   {status}: {env_id}")
            print(f"   📈 Gen {generation} | Score: {mean_score:.1f}/{threshold:.1f} ({progress:.1f}%)")

            # Show all environments in curriculum
            envs = df['env_id'].unique()
            solved_envs = df[df['solved'] == True]['env_id'].unique()
            print(f"   🎯 Progress: {len(solved_envs)}/{len(envs)} environments solved")
        else:
            print("   📝 Log file empty...")
    except Exception as e:
        print(f"   ❌ Error reading log: {e}")

def check_environment_phases(env_dir):
    """Check what phase an environment training is in."""
    phases = []

    # Check data collection
    data_dirs = list(env_dir.glob("**/random_data"))
    if data_dirs:
        for data_dir in data_dirs:
            npz_files = list(data_dir.glob("*.npz"))
            if npz_files:
                phases.append("✅ Data")
            else:
                phases.append("🔄 Data")

    # Check VAE training
    vae_files = list(env_dir.glob("**/vae.pt"))
    if vae_files:
        phases.append("✅ VAE")
    else:
        phases.append("🔄 VAE")

    # Check MDN-RNN training
    rnn_files = list(env_dir.glob("**/mdnrnn.pt"))
    if rnn_files:
        phases.append("✅ RNN")
    else:
        phases.append("🔄 RNN")

    # Check controller training
    controller_files = list(env_dir.glob("**/best_controller.pt"))
    if controller_files:
        phases.append("✅ Controller")
    else:
        phases.append("🔄 Controller")

    return " | ".join(phases) if phases else None

if __name__ == "__main__":
    import sys
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs_20250824_203859"

    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            check_training_progress(runs_dir)
            print(f"\n🔄 Refreshing every 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped.")
