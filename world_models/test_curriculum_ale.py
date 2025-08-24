#!/usr/bin/env python3
"""
Test ALE import in curriculum trainer context
"""

import sys
import os

# Add the current directory to Python path (same as curriculum trainer)
sys.path.insert(0, os.getcwd())

try:
    import gymnasium as gym
    import ale_py
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    print("✅ ALE import and registration successful!")
    print(f"ALE-py version: {ale_py.__version__}")
    
    # Test creating an ALE environment
    env = gym.make("ALE/Pong-v5")
    print(f"✅ ALE/Pong-v5 environment created successfully!")
    
    obs, info = env.reset()
    print(f"✅ Environment reset successful! Observation shape: {obs.shape}")
    
    env.close()
    print("✅ Environment closed successfully!")
    
    print("\n🎉 ALE environments are ready for curriculum trainer!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
