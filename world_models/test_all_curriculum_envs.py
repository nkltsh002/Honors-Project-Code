#!/usr/bin/env python3
"""
Test all curriculum environments after fixing Box2D and CarRacing version issues
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    import gymnasium as gym
    import ale_py
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    print("🧪 Testing All Curriculum Environments")
    print("="*50)
    
    # Test the complete curriculum environments
    test_envs = [
        ("ALE/Pong-v5", "Atari Pong with ALE"),
        ("LunarLander-v3", "Box2D Lunar Lander"),
        ("ALE/Breakout-v5", "Atari Breakout with ALE"),
        ("CarRacing-v3", "Box2D Car Racing (updated from v2)")
    ]
    
    success_count = 0
    
    for env_id, description in test_envs:
        try:
            print(f"\n🔍 Testing {env_id} ({description})")
            print(f"   Creating environment...", end=" ")
            
            env = gym.make(env_id, render_mode=None)
            print("✅ Created")
            
            print(f"   Resetting environment...", end=" ")
            obs, info = env.reset()
            print(f"✅ Reset (obs shape: {obs.shape})")
            
            print(f"   Testing action space...", end=" ")
            action = env.action_space.sample()
            print(f"✅ Action space: {env.action_space}")
            
            print(f"   Taking a step...", end=" ")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✅ Step completed (reward: {reward})")
            
            print(f"   Closing environment...", end=" ")
            env.close()
            print("✅ Closed")
            
            print(f"   ✅ {env_id} - ALL TESTS PASSED")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            print(f"   ❌ {env_id} - TESTS FAILED")
    
    print("\n" + "="*50)
    print(f"🎯 FINAL RESULTS: {success_count}/{len(test_envs)} environments working")
    
    if success_count == len(test_envs):
        print("🎉 ALL CURRICULUM ENVIRONMENTS ARE READY!")
        print("\nReady to run:")
        print("• Full curriculum: python curriculum_trainer_visual.py --device cpu --max-generations 200 --visualize True --record-video True")
        print("• Quick test: python curriculum_trainer_visual.py --device cpu --quick True --visualize True")
    else:
        print("⚠️  Some environments still have issues. Please check the error messages above.")
    
except Exception as e:
    print(f"❌ Critical error during testing: {e}")
    import traceback
    traceback.print_exc()
