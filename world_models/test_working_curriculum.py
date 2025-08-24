#!/usr/bin/env python3
"""
Test the updated curriculum with working environments
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
    
    print("🧪 Testing Updated Curriculum Environments (Working Alternatives)")
    print("="*60)
    
    # Test the updated curriculum environments
    test_envs = [
        ("ALE/Pong-v5", "Atari Pong with ALE"),
        ("ALE/Breakout-v5", "Atari Breakout with ALE"),
        ("CartPole-v1", "Classic Control - CartPole"),
        ("Acrobot-v1", "Classic Control - Acrobot")
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
    
    print("\n" + "="*60)
    print(f"🎯 FINAL RESULTS: {success_count}/{len(test_envs)} environments working")
    
    if success_count == len(test_envs):
        print("🎉 ALL UPDATED CURRICULUM ENVIRONMENTS ARE READY!")
        print("\nWorking curriculum environments:")
        for env_id, desc in test_envs:
            print(f"  ✅ {env_id} - {desc}")
        print("\nReady to run:")
        print("• Full curriculum: python curriculum_trainer_visual.py --device cpu --max-generations 200 --visualize True --record-video True")
        print("• Quick test: python curriculum_trainer_visual.py --device cpu --quick True --visualize True")
    else:
        print("⚠️  Some environments still have issues. Please check the error messages above.")
        
    print("\n📝 NOTE: Box2D environments (LunarLander-v3, CarRacing-v3) have been")
    print("   temporarily replaced with CartPole-v1 and Acrobot-v1.")
    print("   To use Box2D environments, you'll need to solve the Box2D installation issue.")
    
except Exception as e:
    print(f"❌ Critical error during testing: {e}")
    import traceback
    traceback.print_exc()
