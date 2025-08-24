#!/usr/bin/env python3
"""
Minimal test to verify curriculum trainer can start
"""

import sys
import os
sys.path.insert(0, os.getcwd())

print("[TEST] Starting minimal curriculum test...")

try:
    print("[TEST] Testing imports...")
    import torch
    print(f"[TEST] PyTorch: {torch.__version__}")
    
    import gymnasium as gym
    print("[TEST] Gymnasium: OK")
    
    from models.vae import ConvVAE
    from models.mdnrnn import MDNRNN
    from models.controller import Controller
    print("[TEST] World Models imports: OK")
    
    # Import main script
    print("[TEST] Importing curriculum trainer...")
    import curriculum_trainer_visual as ctv
    print("[TEST] Import successful!")
    
    # Test config creation
    config = ctv.TrainingConfig(device='cpu', max_generations=1, episodes_per_eval=1)
    print(f"[TEST] Config created: {config.device}, {config.max_generations} gens")
    
    # Test argument parsing
    import argparse
    parser = ctv.parse_args.__code__.co_names  # Check if parse_args exists
    print("[TEST] Argument parser exists")
    
    print("[TEST] ✓ All tests passed!")
    print("[TEST] The curriculum trainer should work properly.")
    
except Exception as e:
    print(f"[TEST] ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
