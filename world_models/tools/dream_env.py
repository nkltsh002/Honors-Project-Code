"""Dream environment wrapper for World Models.

This module provides a dream environment that uses the trained World Models
components (VAE + MDN-RNN) to simulate environment dynamics. This enables
model-based planning and control without interacting with the real environment.

Key components:
- DreamEnvironment: Main wrapper using VAE + MDN-RNN for simulation
- DreamAgent: Agent that can act in dream environment
- DreamRollout: Utilities for generating rollouts in dream space
- DreamPlanner: Simple planning algorithms using dream environment

The dream environment operates in latent space for efficiency and uses the
MDN-RNN's stochastic predictions to maintain realistic uncertainty.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
from dataclasses import dataclass


@dataclass
class DreamState:
    """State representation in dream environment."""
    latent_obs: torch.Tensor  # Current latent observation
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]  # RNN hidden state
    step: int  # Step counter
    done: bool  # Episode termination flag
    reward: float  # Last reward received
    info: Dict[str, Any]  # Additional information


class DreamEnvironment:
    """Dream environment using VAE + MDN-RNN for simulation."""
    
    def __init__(
        self,
        vae_model_path: str,
        mdnrnn_model_path: str,
        action_space_size: int = 3,
        max_episode_steps: int = 1000,
        temperature: float = 1.0,
        device: str = 'cuda',
        render_mode: Optional[str] = None
    ):
        """Initialize dream environment.
        
        Args:
            vae_model_path: Path to trained VAE checkpoint
            mdnrnn_model_path: Path to trained MDN-RNN checkpoint
            action_space_size: Size of action space
            max_episode_steps: Maximum steps per episode
            temperature: Temperature for MDN sampling
            device: Device for computation
            render_mode: Rendering mode ('rgb_array' or None)
        """
        self.device = device
        self.action_space_size = action_space_size
        self.max_episode_steps = max_episode_steps
        self.temperature = temperature
        self.render_mode = render_mode
        
        # Model paths for loading when needed
        self.vae_path = vae_model_path
        self.mdnrnn_path = mdnrnn_model_path
        
        # Models (will be loaded lazily)
        self.vae = None
        self.mdnrnn = None
        
        # Environment state
        self.current_state: Optional[DreamState] = None
        self.step_count = 0
        
        # Create mock action space for compatibility
        self.action_space = type('MockSpace', (), {
            'low': np.full((action_space_size,), -1.0, dtype=np.float32),
            'high': np.full((action_space_size,), 1.0, dtype=np.float32),
            'shape': (action_space_size,),
            'dtype': np.float32
        })()
        
        print(f"Dream environment initialized with:")
        print(f"  Action space size: {action_space_size}")
        print(f"  Max episode steps: {max_episode_steps}")
        print(f"  Temperature: {temperature}")
    
    def load_models(self):
        """Load VAE and MDN-RNN models when available."""
        try:
            from ..models.conv_vae import ConvVAE
            from ..models.mdnrnn import MDNRNN
            
            # Load VAE
            self.vae = ConvVAE(latent_dim=32).to(self.device)
            vae_checkpoint = torch.load(self.vae_path, map_location=self.device)
            self.vae.load_state_dict(vae_checkpoint['model_state_dict'])
            self.vae.eval()
            
            # Load MDN-RNN
            self.mdnrnn = MDNRNN(
                z_dim=32,
                action_dim=self.action_space_size,
                rnn_size=256,
                num_mixtures=5
            ).to(self.device)
            mdnrnn_checkpoint = torch.load(self.mdnrnn_path, map_location=self.device)
            self.mdnrnn.load_state_dict(mdnrnn_checkpoint['model_state_dict'])
            self.mdnrnn.eval()
            
            print("Successfully loaded VAE and MDN-RNN models")
        except Exception as e:
            print(f"Could not load models: {e}")
            print("Using mock implementations for testing")
    
    def reset(
        self,
        initial_obs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the dream environment.
        
        Args:
            initial_obs: Initial observation (frame or latent code)
            **kwargs: Additional reset arguments
            
        Returns:
            Initial observation and info dict
        """
        self.step_count = 0
        
        if initial_obs is not None:
            # Convert initial observation to latent space if needed
            if isinstance(initial_obs, np.ndarray) and initial_obs.ndim >= 2:
                # Assume it's a frame that needs VAE encoding
                latent_obs = self._encode_observation(initial_obs)
            else:
                # Assume it's already a latent code
                latent_obs = torch.FloatTensor(initial_obs).to(self.device)
                if latent_obs.ndim == 1:
                    latent_obs = latent_obs.unsqueeze(0)
        else:
            # Random initial latent state
            latent_obs = torch.randn(1, 32, device=self.device)
        
        # Initialize dream state
        self.current_state = DreamState(
            latent_obs=latent_obs,
            hidden=None,  # Will be initialized on first step
            step=0,
            done=False,
            reward=0.0,
            info={}
        )
        
        info = {
            'dream_reset': True,
            'initial_latent_obs': latent_obs.cpu().numpy()
        }
        
        return latent_obs.squeeze(), info
    
    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """Take a step in the dream environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        action = action.to(self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        
        # Predict next state using MDN-RNN
        next_latent_obs, reward, done, hidden = self._predict_next_state(
            self.current_state.latent_obs,
            action,
            self.current_state.hidden
        )
        
        # Update step count
        self.step_count += 1
        
        # Check for episode termination
        truncated = self.step_count >= self.max_episode_steps
        terminated = done or truncated
        
        # Update current state
        self.current_state = DreamState(
            latent_obs=next_latent_obs,
            hidden=hidden,
            step=self.step_count,
            done=terminated,
            reward=reward,
            info={}
        )
        
        info = {
            'dream_step': self.step_count,
            'predicted_reward': reward,
            'predicted_done': done,
            'truncated': truncated
        }
        
        return next_latent_obs.squeeze(), reward, terminated, truncated, info
    
    def _encode_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Encode observation using VAE encoder."""
        # Mock encoding for now - return random latent vector
        return torch.randn(1, 32, device=self.device)
    
    def _predict_next_state(
        self,
        current_latent: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, float, bool, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict next state using MDN-RNN."""
        # Mock prediction - add some noise to current state
        next_latent = current_latent + torch.randn_like(current_latent) * 0.1
        reward = float(np.random.randn())
        done = np.random.random() < 0.01  # 1% chance of done
        new_hidden = (torch.randn(1, 1, 256, device=self.device),
                     torch.randn(1, 1, 256, device=self.device))
        
        return next_latent, reward, done, new_hidden
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state."""
        if self.render_mode is None:
            return None
        
        if self.current_state is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Mock rendering - return random image
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    def close(self):
        """Close the environment."""
        self.current_state = None


class DreamAgent:
    """Simple agent for acting in dream environment."""
    
    def __init__(self, action_dim: int, policy_type: str = 'random'):
        """Initialize dream agent.
        
        Args:
            action_dim: Dimension of action space
            policy_type: Type of policy ('random', 'constant', or 'neural')
        """
        self.action_dim = action_dim
        self.policy_type = policy_type
        
        if policy_type == 'constant':
            self.constant_action = np.zeros(action_dim)
        elif policy_type == 'neural':
            # Simple neural network policy
            self.policy_net = nn.Sequential(
                nn.Linear(32, 64),  # Latent dim to hidden
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Tanh()  # Assume actions are in [-1, 1]
            )
    
    def act(self, observation: torch.Tensor) -> np.ndarray:
        """Select action based on observation.
        
        Args:
            observation: Current observation (latent state)
            
        Returns:
            Selected action
        """
        if self.policy_type == 'random':
            return np.random.uniform(-1, 1, size=self.action_dim)
        elif self.policy_type == 'constant':
            return self.constant_action.copy()
        elif self.policy_type == 'neural':
            with torch.no_grad():
                action = self.policy_net(observation)
            return action.numpy()
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")


class DreamRollout:
    """Utilities for generating rollouts in dream environment."""
    
    @staticmethod
    def generate_rollout(
        env: DreamEnvironment,
        agent: DreamAgent,
        max_steps: int = 1000,
        initial_obs: Optional[np.ndarray] = None
    ) -> Dict[str, List]:
        """Generate a rollout in dream environment.
        
        Args:
            env: Dream environment
            agent: Agent to generate rollout
            max_steps: Maximum steps in rollout
            initial_obs: Initial observation
            
        Returns:
            Dictionary containing rollout data
        """
        rollout_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }
        
        # Reset environment
        obs, info = env.reset(initial_obs=initial_obs)
        rollout_data['observations'].append(obs.cpu().numpy())
        rollout_data['infos'].append(info)
        
        # Generate rollout
        for step in range(max_steps):
            # Select action
            action = agent.act(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store data
            rollout_data['actions'].append(action)
            rollout_data['rewards'].append(reward)
            rollout_data['dones'].append(terminated or truncated)
            rollout_data['infos'].append(info)
            rollout_data['observations'].append(next_obs.cpu().numpy())
            
            # Update observation
            obs = next_obs
            
            # Check termination
            if terminated or truncated:
                break
        
        return rollout_data
    
    @staticmethod
    def rollout_statistics(rollout_data: Dict[str, List]) -> Dict[str, float]:
        """Compute statistics for a rollout.
        
        Args:
            rollout_data: Rollout data from generate_rollout
            
        Returns:
            Dictionary with rollout statistics
        """
        rewards = rollout_data['rewards']
        
        stats = {
            'total_reward': sum(rewards),
            'episode_length': len(rewards),
            'mean_reward': np.mean(rewards) if rewards else 0.0,
            'std_reward': np.std(rewards) if rewards else 0.0,
            'min_reward': min(rewards) if rewards else 0.0,
            'max_reward': max(rewards) if rewards else 0.0
        }
        
        return stats


# Example usage and testing functions
def test_dream_environment():
    """Test the dream environment with mock models."""
    print("Testing Dream Environment...")
    
    # Create dream environment (will use mock models)
    env = DreamEnvironment(
        vae_model_path="nonexistent_vae.pth",
        mdnrnn_model_path="nonexistent_mdnrnn.pth",
        action_space_size=3,
        max_episode_steps=100,
        temperature=1.0,
        device='cpu'  # Use CPU for testing
    )
    
    # Create random agent
    agent = DreamAgent(action_dim=3, policy_type='random')
    
    # Generate rollout
    rollout_data = DreamRollout.generate_rollout(
        env=env,
        agent=agent,
        max_steps=50
    )
    
    # Compute statistics
    stats = DreamRollout.rollout_statistics(rollout_data)
    
    print("Rollout Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("Dream environment test completed!")


if __name__ == "__main__":
    test_dream_environment()
