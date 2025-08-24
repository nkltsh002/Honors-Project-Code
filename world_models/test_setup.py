#!/usr/bin/env python3
"""
Test script to verify World Models implementation setup and basic functionality.
Run this after installation to ensure everything is working correctly.
"""

import torch
import gymnasium as gym
import numpy as np
from pathlib import Path

def test_torch_installation():
    """Test PyTorch installation and GPU availability."""
    print("🔧 Testing PyTorch installation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
    
    # Test basic tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(3, 2)
    z = torch.mm(x, y)
    print(f"✅ Basic tensor operations working")
    
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = torch.mm(x_gpu, y_gpu)
        print(f"✅ GPU tensor operations working")
    
    print()

def test_gymnasium_installation():
    """Test Gymnasium and environment availability."""
    print("🎮 Testing Gymnasium environments...")
    
    # Test basic environments
    test_envs = [
        "LunarLander-v2",
        "ALE/Pong-v5", 
        "ALE/Breakout-v5",
        "CarRacing-v2"
    ]
    
    for env_name in test_envs:
        try:
            env = gym.make(env_name)
            obs, info = env.reset()
            print(f"✅ {env_name}: obs_shape={obs.shape}, action_space={env.action_space}")
            env.close()
        except Exception as e:
            print(f"❌ {env_name}: {str(e)}")
    
    print()

def test_model_imports():
    """Test that all model components can be imported."""
    print("🧠 Testing model imports...")
    
    try:
        from config import get_config
        config = get_config('ALE/Pong-v5')
        print(f"✅ Configuration loaded for Pong")
        
        from models.vae import ConvVAE
        vae = ConvVAE(
            latent_size=config.vae.latent_size
        )
        print(f"✅ ConvVAE model created: {sum(p.numel() for p in vae.parameters())} parameters")
        
        from models.mdnrnn import MDNRNN
        mdnrnn = MDNRNN(
            latent_size=config.vae.latent_size,
            action_size=config.controller.action_size,
            hidden_size=config.mdnrnn.hidden_size,
            num_mixtures=config.mdnrnn.num_mixtures
        )
        print(f"✅ MDN-RNN model created: {sum(p.numel() for p in mdnrnn.parameters())} parameters")
        
        from models.controller import Controller
        controller = Controller(
            input_size=config.controller.input_size,
            action_size=config.controller.action_size,
            hidden_sizes=config.controller.hidden_sizes
        )
        print(f"✅ Controller model created: {sum(p.numel() for p in controller.parameters())} parameters")
        
        from utils.environment import EnvironmentWrapper
        print(f"✅ Environment utilities imported")
        
        from training.train_utils import TrainingLogger
        print(f"✅ Training utilities imported")
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
    
    print()

def test_data_pipeline():
    """Test basic data pipeline functionality."""
    print("💾 Testing data pipeline...")
    
    try:
        from utils.environment import FramePreprocessor
        preprocessor = FramePreprocessor()
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        processed = preprocessor.preprocess_frame(dummy_frame, 'ALE/Pong-v5')
        
        expected_shape = (64, 64, 3)
        assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
        assert processed.dtype == np.uint8, f"Expected uint8, got {processed.dtype}"
        
        print(f"✅ Frame preprocessing: {dummy_frame.shape} -> {processed.shape}")
        
    except Exception as e:
        print(f"❌ Data pipeline error: {str(e)}")
    
    print()

def test_training_setup():
    """Test training setup and basic forward pass."""
    print("🚀 Testing training setup...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from config import get_config
        from models.vae import ConvVAE
        
        config = get_config('ALE/Pong-v5')
        vae = ConvVAE(
            latent_size=config.vae.latent_size
        ).to(device)
        
        # Test forward pass
        batch_size = 4
        dummy_batch = torch.randn(batch_size, 3, 64, 64).to(device)
        
        with torch.no_grad():
            reconstructed, mu, logvar = vae(dummy_batch)
        
        print(f"✅ VAE forward pass: input={dummy_batch.shape}, output={reconstructed.shape}")
        print(f"✅ Latent representations: mu={mu.shape}, logvar={logvar.shape}")
        
        # Test basic loss computation
        reconstruction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            reconstructed, dummy_batch, reduction='mean'
        )
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        print(f"✅ Loss computation: recon_loss={reconstruction_loss:.4f}, kl_loss={kl_loss:.4f}")
        
    except Exception as e:
        print(f"❌ Training setup error: {str(e)}")
    
    print()

def test_directory_structure():
    """Test that all necessary directories can be created."""
    print("📁 Testing directory structure...")
    
    base_dir = Path("./test_world_models")
    required_dirs = [
        "checkpoints",
        "logs", 
        "data",
        "videos",
        "experiments"
    ]
    
    try:
        # Create test directories
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        
        # Clean up
        import shutil
        shutil.rmtree(base_dir, ignore_errors=True)
        print(f"✅ Directory cleanup completed")
        
    except Exception as e:
        print(f"❌ Directory structure error: {str(e)}")
    
    print()

def main():
    """Run all tests."""
    print("🧪 World Models Implementation Test Suite")
    print("=" * 50)
    
    test_torch_installation()
    test_gymnasium_installation() 
    test_model_imports()
    test_data_pipeline()
    test_training_setup()
    test_directory_structure()
    
    print("🎉 Test suite completed!")
    print("\n💡 If all tests passed, you're ready to start training:")
    print("   python train.py --env ALE/Pong-v5")
    print("\n📊 To monitor training progress:")
    print("   tensorboard --logdir ./logs")

if __name__ == "__main__":
    main()
