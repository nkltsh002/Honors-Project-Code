#!/usr/bin/env python3
"""
Simplified test version of curriculum trainer
"""

import os
import sys
import argparse
import time

# Import World Models components
sys.path.insert(0, os.getcwd())

def test_imports():
    """Test all required imports"""
    print("🔧 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import gymnasium as gym  
        print(f"✅ Gymnasium")
        
        # Test environment creation
        env = gym.make("PongNoFrameskip-v4")
        print(f"✅ Environment: {env.spec.id}")
        env.close()
        
        from models.vae import ConvVAE
        print("✅ ConvVAE")
        
        from models.mdnrnn import MDNRNN
        print("✅ MDNRNN") 
        
        from models.controller import Controller
        print("✅ Controller")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic curriculum trainer functionality"""
    print("\n🎯 Testing basic functionality...")
    
    try:
        # Test argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='cpu')
        parser.add_argument('--max-generations', type=int, default=2)
        args = parser.parse_args(['--device', 'cpu'])
        print(f"✅ Arguments parsed: device={args.device}, generations={args.max_generations}")
        
        # Test config creation
        from curriculum_trainer_visual import TrainingConfig
        config = TrainingConfig(device=args.device, max_generations=args.max_generations)
        print(f"✅ Config created: {config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Curriculum Trainer Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
        
    # Test basic functionality  
    if not test_basic_functionality():
        print("❌ Basic functionality test failed")
        return False
        
    print("\n🎉 All tests passed!")
    print("The curriculum trainer should work properly.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
