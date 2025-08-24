#!/usr/bin/env python3
"""
Step-by-step test of curriculum trainer components
"""

import sys
import os

print("=== Curriculum Trainer Debug Test ===")

# Test 1: Basic Python
print("1. Python version:", sys.version)

# Test 2: Path setup
sys.path.insert(0, os.getcwd())
print("2. Working directory:", os.getcwd())

# Test 3: Basic imports
try:
    import torch
    print("3. PyTorch version:", torch.__version__)
except Exception as e:
    print("3. PyTorch import FAILED:", e)
    sys.exit(1)

try:
    import gymnasium as gym
    print("4. Gymnasium imported successfully")
except Exception as e:
    print("4. Gymnasium import FAILED:", e)
    sys.exit(1)

# Test 4: World Models imports
try:
    from models.vae import ConvVAE
    print("5. ConvVAE imported successfully")
except Exception as e:
    print("5. ConvVAE import FAILED:", e)
    sys.exit(1)

try:
    from models.mdnrnn import MDNRNN
    print("6. MDNRNN imported successfully")
except Exception as e:
    print("6. MDNRNN import FAILED:", e)
    sys.exit(1)

try:
    from models.controller import Controller
    print("7. Controller imported successfully")
except Exception as e:
    print("7. Controller import FAILED:", e)
    sys.exit(1)

# Test 5: Environment creation
try:
    env = gym.make("PongNoFrameskip-v4")
    print("8. Environment created successfully:", env.spec.id)
    env.close()
except Exception as e:
    print("8. Environment creation FAILED:", e)
    print("   Trying alternative environment...")
    try:
        env = gym.make("CartPole-v1")
        print("8. Alternative environment works:", env.spec.id)
        env.close()
    except Exception as e2:
        print("8. All environment tests FAILED:", e2)

# Test 6: Try to import curriculum trainer module
try:
    print("9. Testing curriculum trainer import...")
    import curriculum_trainer_visual as ctv
    print("9. curriculum_trainer_visual imported successfully")
    
    # Test config creation
    config = ctv.TrainingConfig()
    print("10. TrainingConfig created successfully")
    
    print("\n=== ALL TESTS PASSED ===")
    print("The curriculum trainer should work!")
    
except Exception as e:
    print("9. curriculum_trainer_visual import FAILED:", e)
    import traceback
    traceback.print_exc()

print("\n=== TEST COMPLETE ===")
