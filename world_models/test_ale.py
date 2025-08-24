#!/usr/bin/env python3
"""
Test script to verify ALE environments are working
"""

import gymnasium as gym
import ale_py  # Import ale_py to register ALE environments

def test_ale_environments():
    """Test ALE environment creation"""
    environments = [
        "ALE/Pong-v5",
        "ALE/Breakout-v5"
    ]
    
    print("Testing ALE environments...")
    print(f"ALE-py version: {ale_py.__version__}")
    
    # Register ALE environments explicitly
    gym.register_envs(ale_py)
    
    for env_id in environments:
        try:
            print(f"Creating {env_id}...", end=" ")
            env = gym.make(env_id)
            print("✅ SUCCESS")
            
            # Test basic functionality
            obs, info = env.reset()
            print(f"  - Observation shape: {obs.shape}")
            print(f"  - Action space: {env.action_space}")
            
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  - Step completed: reward={reward}")
            
            env.close()
            print(f"  - Environment closed successfully")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            
        print()

if __name__ == "__main__":
    test_ale_environments()
    print("ALE environment test completed!")
