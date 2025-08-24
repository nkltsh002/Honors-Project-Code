"""
World Models Configuration File
Based on Ha & Schmidhuber (2018): "World Models"

This configuration file contains all hyperparameters and settings for the World Models
architecture implementation. Parameters are chosen based on the original paper with
modern optimizations and GPU-friendly batch sizes.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import os

@dataclass
class VAEConfig:
    """
    Convolutional Variational Autoencoder Configuration
    
    The VAE compresses 64x64 RGB frames into latent vectors z.
    Architecture follows the paper with minor modifications for stability.
    """
    # Input/Output dimensions
    input_size: Tuple[int, int, int] = (3, 64, 64)  # RGB 64x64 frames
    latent_size: int = 32  # Size of latent vector z (original paper uses 32)
    
    # Architecture parameters
    hidden_channels: Tuple[int, ...] = (32, 64, 128, 256)  # Encoder channels
    kernel_sizes: Tuple[int, ...] = (4, 4, 4, 4)  # Convolution kernel sizes
    strides: Tuple[int, ...] = (2, 2, 2, 2)  # Convolution strides
    
    # Training hyperparameters
    learning_rate: float = 1e-4  # Adam optimizer learning rate
    batch_size: int = 128  # Batch size for training
    beta: float = 4.0  # Beta parameter for β-VAE (controls KL divergence weight)
    reconstruction_weight: float = 1.0  # Weight for reconstruction loss
    
    # Training schedule
    num_epochs: int = 50  # Number of training epochs
    warmup_epochs: int = 5  # KL warmup epochs (gradually increase β)
    
    # Data augmentation
    random_crop: bool = True
    random_flip: bool = False  # Usually false for games like racing
    
@dataclass 
class MDNRNNConfig:
    """
    Mixture Density Network + LSTM Configuration
    
    The MDN-RNN models temporal dynamics: p(z_{t+1} | a_t, z_t, h_t)
    where h_t is the hidden state of the LSTM.
    """
    # Input dimensions
    latent_size: int = 32  # Size of latent vector z from VAE
    action_size: int = 3   # Action space size (environment dependent)
    
    # LSTM parameters
    hidden_size: int = 256  # LSTM hidden state size (original paper uses 256)
    num_layers: int = 1     # Number of LSTM layers
    
    # Mixture Density Network parameters
    num_mixtures: int = 5   # Number of Gaussian mixtures (K in paper)
    
    # Training hyperparameters
    learning_rate: float = 1e-3  # Adam optimizer learning rate
    batch_size: int = 32         # Batch size for sequence training
    sequence_length: int = 100   # Length of sequences for LSTM training
    
    # Training schedule
    num_epochs: int = 100
    gradient_clip: float = 1.0   # Gradient clipping for LSTM stability
    
    # Regularization
    dropout: float = 0.1         # Dropout rate for regularization
    
@dataclass
class ControllerConfig:
    """
    Controller Configuration
    
    The controller maps [z_t, h_t] to actions. Can be linear or small MLP.
    Trained with evolution strategies (CMA-ES) or optionally PPO.
    """
    # Input dimensions
    input_size: int = 32 + 256   # latent_size + hidden_size
    action_size: int = 3         # Environment dependent
    
    # Architecture
    hidden_sizes: Tuple[int, ...] = ()  # Empty for linear controller, (64,) for MLP
    use_bias: bool = True
    activation: str = 'tanh'     # Activation function
    
    # CMA-ES parameters
    population_size: int = 64    # CMA-ES population size
    sigma: float = 0.5           # Initial standard deviation
    max_generations: int = 100   # Maximum generations
    
    # PPO parameters (alternative training)
    ppo_learning_rate: float = 3e-4
    ppo_epochs: int = 10
    ppo_batch_size: int = 64
    ppo_clip: float = 0.2
    
@dataclass
class TrainingConfig:
    """
    Overall Training Configuration
    """
    # Data collection
    num_random_rollouts: int = 10000  # Random rollouts for VAE training
    rollout_length: int = 1000        # Maximum length per rollout
    
    # Environment settings
    environments: Tuple[str, ...] = (
        'ALE/Pong-v5',
        'LunarLander-v2', 
        'ALE/Breakout-v5',
        'CarRacing-v2'
    )
    render_mode: str = 'rgb_array'
    
    # Device and performance
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4              # DataLoader workers
    pin_memory: bool = True
    
    # Checkpointing and logging
    save_dir: str = './checkpoints'
    log_dir: str = './logs'
    save_frequency: int = 10          # Save every N epochs
    log_frequency: int = 100          # Log every N steps
    
    # Visualization
    visualize: bool = True            # Real-time visualization
    save_videos: bool = True          # Save rollout videos
    plot_losses: bool = True          # Plot training curves
    
    # Reproducibility
    seed: int = 42

@dataclass
class ExperimentConfig:
    """
    Experiment-specific configurations
    """
    # Current experiment
    experiment_name: str = 'world_models_v1'
    environment: str = 'ALE/Pong-v5'
    
    # Training phases
    train_vae: bool = True
    train_mdnrnn: bool = True
    train_controller: bool = True
    
    # Baseline comparison
    train_baseline_ppo: bool = True
    
    # Model configurations
    vae: VAEConfig = VAEConfig()
    mdnrnn: MDNRNNConfig = MDNRNNConfig()
    controller: ControllerConfig = ControllerConfig()
    training: TrainingConfig = TrainingConfig()
    
    def __post_init__(self):
        """Post-initialization setup based on environment"""
        self.setup_environment_specific_configs()
        
    def setup_environment_specific_configs(self):
        """
        Adjust configurations based on the target environment.
        Different games have different action spaces and optimal parameters.
        """
        env_configs = {
            'ALE/Pong-v5': {
                'action_size': 6,  # Atari Pong actions
                'mdnrnn_lr': 1e-3,
                'controller_sigma': 0.5,
            },
            'LunarLander-v2': {
                'action_size': 4,  # LunarLander actions
                'mdnrnn_lr': 1e-3,
                'controller_sigma': 0.3,
            },
            'ALE/Breakout-v5': {
                'action_size': 4,  # Atari Breakout actions
                'mdnrnn_lr': 1e-3,
                'controller_sigma': 0.5,
            },
            'CarRacing-v2': {
                'action_size': 3,  # Continuous actions (discretized)
                'mdnrnn_lr': 5e-4,  # Lower LR for more complex dynamics
                'controller_sigma': 0.2,  # Smaller exploration for precise control
            }
        }
        
        if self.environment in env_configs:
            config = env_configs[self.environment]
            self.mdnrnn.action_size = config['action_size']
            self.controller.action_size = config['action_size'] 
            self.controller.input_size = self.vae.latent_size + self.mdnrnn.hidden_size
            self.mdnrnn.learning_rate = config['mdnrnn_lr']
            self.controller.sigma = config['controller_sigma']

# Global configuration instance
def get_config(environment: str = 'ALE/Pong-v5', experiment_name: str = None) -> ExperimentConfig:
    """
    Get configuration for a specific environment and experiment.
    
    Args:
        environment: Target environment name
        experiment_name: Optional experiment name override
        
    Returns:
        ExperimentConfig: Complete configuration object
    """
    config = ExperimentConfig()
    config.environment = environment
    if experiment_name:
        config.experiment_name = experiment_name
        
    # Create directories
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    return config

# Environment-specific helper functions
def get_action_space_size(env_name: str) -> int:
    """Get the action space size for a given environment"""
    action_spaces = {
        'ALE/Pong-v5': 6,
        'LunarLander-v2': 4,
        'ALE/Breakout-v5': 4,
        'CarRacing-v2': 3,
    }
    return action_spaces.get(env_name, 3)

def is_atari_environment(env_name: str) -> bool:
    """Check if the environment is an Atari game"""
    return env_name.startswith('ALE/')

def requires_frame_preprocessing(env_name: str) -> bool:
    """Check if the environment requires special frame preprocessing"""
    return env_name in ['CarRacing-v2'] or is_atari_environment(env_name)
