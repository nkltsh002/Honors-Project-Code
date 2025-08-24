import os
import glob
import json
from datetime import datetime

def check_training_progress():
    print(f"=== Training Progress Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # Check both run directories
    full_dir = "runs\\full_curriculum_20250824_192155"
    classic_dir = "runs\\classic_curriculum_20250824_192155"

    for run_name, run_dir in [("FULL", full_dir), ("CLASSIC", classic_dir)]:
        print(f"{run_name} Curriculum ({run_dir}):")

        if not os.path.exists(run_dir):
            print(f"  ❌ Directory not found: {run_dir}")
            continue

        # Look for progress files
        csv_files = glob.glob(os.path.join(run_dir, "**", "*.csv"), recursive=True)
        json_files = glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
        npz_files = glob.glob(os.path.join(run_dir, "**", "*.npz"), recursive=True)

        print(f"  📊 CSV files: {len(csv_files)}")
        print(f"  📄 JSON files: {len(json_files)}")
        print(f"  💾 NPZ files: {len(npz_files)}")

        # Check for curriculum results
        results_file = os.path.join(run_dir, "curriculum_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"  ✅ Results found: {results.get('tasks_completed', 0)}/{results.get('total_tasks', 4)} tasks completed")
                print(f"  ⏱️ Training time: {results.get('total_time_hours', 0):.2f} hours")
            except:
                print(f"  ⚠️ Results file exists but couldn't be read")
        else:
            print(f"  ⏳ Training in progress...")

        # List environment directories
        env_dirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
        if env_dirs:
            print(f"  🎮 Environments: {', '.join(env_dirs)}")

        print()

if __name__ == "__main__":
    check_training_progress()
