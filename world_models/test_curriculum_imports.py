#!/usr/bin/env python3
"""
Test script to check curriculum_trainer_visual imports and basic functionality
"""

import sys
import os
sys.path.insert(0, os.getcwd())

print("Testing curriculum trainer imports...")

try:
    print("Importing basic libraries...")
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    import gymnasium as gym
    print(f"✅ Gymnasium {gym.__version__}")
    
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
    
    print("Importing World Models components...")
    from models.vae import ConvVAE
    print("✅ ConvVAE")
    
    from models.mdnrnn import MDNRNN
    print("✅ MDNRNN")
    
    from models.controller import Controller, CMAESController
    print("✅ Controller and CMAESController")
    
    print("Testing curriculum_trainer_visual script...")
    
    # Import the main script
    import curriculum_trainer_visual as ctv
    print("✅ curriculum_trainer_visual imported successfully")
    
    # Test argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-generations', type=int, default=5)
    parser.add_argument('--episodes-per-eval', type=int, default=2)
    parser.add_argument('--checkpoint-dir', default='./test_runs')
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--record-video', type=bool, default=False)
    
    args = parser.parse_args(['--device', 'cpu', '--max-generations', '2', '--episodes-per-eval', '1'])
    print("✅ Argument parsing works")
    
    # Test config creation
    config = ctv.TrainingConfig(
        device=args.device,
        max_generations=args.max_generations,
        episodes_per_eval=args.episodes_per_eval,
        checkpoint_dir=args.checkpoint_dir,
        visualize=args.visualize,
        record_video=args.record_video
    )
    print("✅ TrainingConfig created")
    
    print("\n🎉 All tests passed! The script should work.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
