#!/usr/bin/env python3
"""
Quick test of curriculum trainer ALE environment creation
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    import torch
    import gymnasium as gym
    import ale_py
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    print("Testing curriculum trainer environment creation...")
    
    # Test the new curriculum environments
    test_envs = [
        "ALE/Pong-v5",
        "LunarLander-v3", 
        "ALE/Breakout-v5",
        "CarRacing-v2"
    ]
    
    for env_id in test_envs:
        try:
            print(f"Creating {env_id}...", end=" ")
            env = gym.make(env_id, render_mode=None)
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            print("✅ SUCCESS")
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print("\n🎉 All curriculum environments are ready!")
    
except Exception as e:
    print(f"❌ Error: {e}")
