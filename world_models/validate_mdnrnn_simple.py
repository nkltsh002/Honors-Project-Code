"""Simple validation script for MDN-RNN module.

This script provides basic testing and validation for the MDN-RNN implementation.
Run this script to verify that the MDN-RNN implementation is working correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from models.mdnrnn import MDNRNN, sample_from_mdn
    MODELS_AVAILABLE = True
    print("Successfully imported MDN-RNN models")
except ImportError as e:
    print(f"Could not import models: {e}")
    MODELS_AVAILABLE = False

try:
    from tools.dream_env import DreamEnvironment, DreamAgent, DreamRollout
    DREAM_AVAILABLE = True
    print("Successfully imported dream environment")
except ImportError as e:
    print(f"Could not import dream environment: {e}")
    DREAM_AVAILABLE = False


def test_model_creation():
    """Test basic model creation and forward pass."""
    print("\nTesting Model Creation")
    print("=" * 50)
    
    if not MODELS_AVAILABLE:
        print("Models not available")
        return False
    
    try:
        # Create model
        model = MDNRNN(
            z_dim=32,
            action_dim=3,
            rnn_size=128,
            num_mixtures=5,
            predict_reward=False,
            predict_done=False
        )
        
        print("Model created successfully")
        print(f"  - Latent dim: 32")
        print(f"  - Action dim: 3")
        print(f"  - RNN size: 128")
        print(f"  - Num mixtures: 5")
        
        # Test forward pass
        batch_size, seq_len = 4, 10
        input_dim = 32 + 3  # z_dim + action_dim
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            output, hidden = model(x)
        
        print("Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output MDN params shape: {output['mdn_params'].shape}")
        print(f"  - Hidden state shapes: {[h.shape for h in hidden]}")
        
        # Check expected shapes
        expected_mdn_dim = 5 * (32 + 2)  # num_mixtures * (z_dim + 2)
        assert output['mdn_params'].shape[-1] == expected_mdn_dim
        
        print("Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"Model creation test failed: {e}")
        return False


def test_sampling():
    """Test MDN sampling with simple parameters."""
    print("\nTesting MDN Sampling")
    print("=" * 50)
    
    if not MODELS_AVAILABLE:
        print("Models not available")
        return False
    
    try:
        batch_size = 4
        num_mixtures = 3
        z_dim = 8
        
        # Create simple test parameters
        weights = torch.ones(batch_size, num_mixtures)  # Equal weights
        means = torch.randn(batch_size, num_mixtures, z_dim)
        log_stds = torch.full((batch_size, num_mixtures), -1.0)  # Small variance
        
        # Test sampling
        samples = sample_from_mdn(weights, means, log_stds, temperature=1.0)
        
        print("Sampling successful")
        print(f"  - Sample shape: {samples.shape}")
        print(f"  - Sample mean: {samples.mean().item():.4f}")
        print(f"  - Sample std: {samples.std().item():.4f}")
        
        # Check shape
        assert samples.shape == (batch_size, z_dim)
        
        # Check finite values
        assert torch.isfinite(samples).all()
        
        print("Sampling test passed!")
        return True
        
    except Exception as e:
        print(f"Sampling test failed: {e}")
        return False


def test_dream_environment():
    """Test dream environment basic functionality."""
    print("\nTesting Dream Environment")
    print("=" * 50)
    
    if not DREAM_AVAILABLE:
        print("Dream environment not available")
        return False
    
    try:
        # Create environment
        env = DreamEnvironment(
            vae_model_path="dummy.pth",
            mdnrnn_model_path="dummy.pth",
            action_space_size=3,
            max_episode_steps=20,
            device='cpu'
        )
        
        # Create agent
        agent = DreamAgent(action_dim=3, policy_type='random')
        
        print("Environment and agent created")
        
        # Test reset
        obs, info = env.reset()
        print(f"Reset successful, obs shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {i+1}: reward={reward:.3f}, done={terminated or truncated}")
            
            if terminated or truncated:
                break
        
        print("Dream environment test passed!")
        return True
        
    except Exception as e:
        print(f"Dream environment test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("\nTesting Component Integration")
    print("=" * 50)
    
    if not (MODELS_AVAILABLE and DREAM_AVAILABLE):
        print("Not all components available")
        return False
    
    try:
        # Create model
        model = MDNRNN(
            z_dim=32,
            action_dim=3,
            rnn_size=64,
            num_mixtures=3,
            predict_reward=False,
            predict_done=False
        )
        
        # Create some synthetic training data
        batch_size, seq_len = 2, 5
        input_dim = 32 + 3
        
        # Random sequences
        sequences = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        with torch.no_grad():
            outputs, hidden = model(sequences)
        
        print("Model forward pass successful")
        
        # Test dream environment rollout
        env = DreamEnvironment(
            vae_model_path="dummy.pth",
            mdnrnn_model_path="dummy.pth", 
            action_space_size=3,
            max_episode_steps=10,
            device='cpu'
        )
        
        agent = DreamAgent(action_dim=3, policy_type='random')
        
        rollout_data = DreamRollout.generate_rollout(
            env=env,
            agent=agent, 
            max_steps=10
        )
        
        stats = DreamRollout.rollout_statistics(rollout_data)
        
        print("Dream rollout successful")
        print(f"  Episode length: {stats['episode_length']}")
        print(f"  Total reward: {stats['total_reward']:.3f}")
        
        print("Integration test passed!")
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False


def run_validation():
    """Run all validation tests."""
    print("MDN-RNN Validation Script")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("MDN Sampling", test_sampling),
        ("Dream Environment", test_dream_environment),
        ("Integration", test_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        results[test_name] = test_func()
    
    # Summary
    print("\nVALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<20} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nAll tests passed! MDN-RNN implementation is working correctly.")
        print("\nYou can now:")
        print("  1. Train the MDN-RNN using your data")
        print("  2. Use the dream environment for planning")
        print("  3. Integrate with your World Models pipeline")
    else:
        print(f"\n{total-passed} test(s) failed. Please check the implementation.")
    
    return results


if __name__ == "__main__":
    results = run_validation()
    
    print(f"\nValidation complete!")
    print(f"Check the output above for detailed results.")
    
    # Optional: Show available modules
    print(f"\nModule availability:")
    print(f"  MDN-RNN models: {'Available' if MODELS_AVAILABLE else 'Not Available'}")
    print(f"  Dream environment: {'Available' if DREAM_AVAILABLE else 'Not Available'}")
