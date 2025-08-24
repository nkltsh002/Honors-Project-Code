"""
PPO Baseline Implementation for World Models Comparison

This module implements a baseline Proximal Policy Optimization (PPO) agent
for comparison with the World Models approach. The PPO agent operates directly
on raw environment observations without using learned world models.

Key features:
- Standard PPO implementation with Actor-Critic architecture
- CNN-based policy for visual observations
- GAE (Generalized Advantage Estimation) for advantage computation
- Vectorized environment support for parallel training
- Comprehensive logging and visualization

Based on Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
import os
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.environment import EnvironmentWrapper, FramePreprocessor
from training.train_utils import TrainingLogger

class CNNPolicy(nn.Module):
    """
    CNN-based policy network for visual observations.
    
    Processes 64x64 RGB frames and outputs action probabilities
    and state value estimates for PPO training.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        action_size: int = 6,
        hidden_size: int = 512
    ):
        """
        Initialize CNN policy.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            action_size: Size of action space
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        self.action_size = action_size
        
        # CNN feature extractor (similar to VAE encoder)
        self.cnn = nn.Sequential(
            # 64x64x3 -> 32x32x32
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 32x32x32 -> 16x16x64  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Calculate flattened CNN output size
        self.cnn_output_size = 256 * 4 * 4
        
        # Shared hidden layers
        self.shared_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_size, action_size)
        
        # Value head (critic)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if module == self.policy_head:
                    # Small initialization for policy head
                    nn.init.xavier_normal_(module.weight, gain=0.01)
                elif module == self.value_head:
                    # Standard initialization for value head
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                else:
                    nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                    
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            x: Input frames (batch_size, 3, 64, 64)
            
        Returns:
            Tuple of (action_logits, state_values)
        """
        # Normalize input to [0, 1] range
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0
        
        # CNN feature extraction
        cnn_features = self.cnn(x)
        flattened = cnn_features.view(cnn_features.size(0), -1)
        
        # Shared layers
        shared_out = self.shared_fc(flattened)
        
        # Policy and value outputs
        action_logits = self.policy_head(shared_out)
        state_values = self.value_head(shared_out)
        
        return action_logits, state_values.squeeze(-1)
    
    def get_action_and_value(
        self, 
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action probabilities, sampled action, log probability, and state value.
        
        Args:
            x: Input observations
            action: Specific action to evaluate (optional)
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(x)
        
        # Create action distribution
        action_dist = Categorical(logits=action_logits)
        
        if action is None:
            # Sample action
            action = action_dist.sample()
        
        # Compute log probability and entropy
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action, log_prob, entropy, value

class PPOBuffer:
    """
    Buffer for storing PPO rollout data and computing advantages.
    
    Implements Generalized Advantage Estimation (GAE) for stable
    advantage computation and handles batched data collection.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Initialize PPO buffer.
        
        Args:
            buffer_size: Maximum buffer size
            observation_shape: Shape of observations
            action_size: Size of action space
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage arrays
        self.observations = np.zeros((buffer_size,) + observation_shape, dtype=np.uint8)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # GAE computation arrays
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        
    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store a single step of experience"""
        assert self.ptr < self.buffer_size
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """
        Finish an episode path and compute advantages using GAE.
        
        Args:
            last_value: Value estimate for final state (0 if terminal)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute advantages using GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        advantages = np.zeros_like(deltas)
        advantage = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.gae_lambda * advantage
            advantages[t] = advantage
        
        self.advantages[path_slice] = advantages
        
        # Compute returns (advantages + values)
        self.returns[path_slice] = advantages + self.values[path_slice]
        
        self.path_start_idx = self.ptr
    
    def get(self) -> Dict[str, np.ndarray]:
        """
        Get all stored data and prepare for training.
        
        Returns:
            Dictionary of training data
        """
        assert self.ptr == self.buffer_size
        
        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
        data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns,
            'dones': self.dones
        }
        
        # Reset buffer
        self.ptr = 0
        self.path_start_idx = 0
        
        return data

class PPOAgent:
    """
    Complete PPO agent implementation.
    
    Manages policy training, environment interaction, and performance tracking
    for baseline comparison with World Models.
    """
    
    def __init__(
        self,
        env_name: str,
        learning_rate: float = 3e-4,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        buffer_size: int = 2048,
        batch_size: int = 64,
        epochs_per_update: int = 10,
        device: torch.device = torch.device('cpu'),
        logger: Optional[TrainingLogger] = None
    ):
        """
        Initialize PPO agent.
        
        Args:
            env_name: Environment name
            learning_rate: Adam learning rate
            clip_param: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            buffer_size: Rollout buffer size
            batch_size: Mini-batch size for updates
            epochs_per_update: Number of optimization epochs per update
            device: Training device
            logger: Training logger
        """
        self.env_name = env_name
        self.device = device
        self.logger = logger
        
        # Hyperparameters
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        
        # Create environment
        self.env = EnvironmentWrapper(env_name, render_mode='rgb_array')
        
        # Initialize policy network
        self.policy = CNNPolicy(
            input_channels=3,
            action_size=int(self.env.action_size),
            hidden_size=512
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = PPOBuffer(
            buffer_size=buffer_size,
            observation_shape=(64, 64, 3),
            action_size=int(self.env.action_size),
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        
        # Training statistics
        self.update_count = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        print(f"PPO Agent initialized for {env_name}")
        print(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"Action space: {self.env.action_size}")
    
    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect a full buffer of rollout data.
        
        Returns:
            Dictionary of rollout statistics
        """
        self.policy.eval()
        
        # Reset environment
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        rollout_stats = {
            'total_reward': 0.0,
            'num_episodes': 0.0,
            'avg_episode_reward': 0.0,
            'avg_episode_length': 0.0
        }
        
        for step in range(self.buffer.buffer_size):
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                action, log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)
                
            action_np = action.cpu().numpy().item()
            log_prob_np = log_prob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            
            # Take environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            done = terminated or truncated
            
            # Store experience
            self.buffer.store(
                obs=obs,
                action=action_np,
                reward=reward,
                value=value_np,
                log_prob=log_prob_np,
                done=done
            )
            
            episode_reward += reward
            episode_length += 1
            rollout_stats['total_reward'] += reward
            
            obs = next_obs
            
            if done:
                # Finish episode path
                self.buffer.finish_path(last_value=0.0)
                
                # Store episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                rollout_stats['num_episodes'] += 1
                
                # Reset environment
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        # Finish any incomplete path
        if episode_length > 0:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                _, _, _, value = self.policy.get_action_and_value(obs_tensor)
                last_value = value.cpu().numpy().item()
                
            self.buffer.finish_path(last_value=last_value)
        
        # Compute rollout statistics
        if len(self.episode_rewards) > 0:
            rollout_stats['avg_episode_reward'] = float(np.mean(self.episode_rewards))
            rollout_stats['avg_episode_length'] = float(np.mean(self.episode_lengths))
        
        return rollout_stats
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected rollout data.
        
        Returns:
            Dictionary of training statistics
        """
        self.policy.train()
        
        # Get rollout data
        data = self.buffer.get()
        
        # Convert to tensors
        observations = torch.from_numpy(data['observations']).permute(0, 3, 1, 2).float().to(self.device)
        actions = torch.from_numpy(data['actions']).long().to(self.device)
        old_log_probs = torch.from_numpy(data['log_probs']).to(self.device)
        advantages = torch.from_numpy(data['advantages']).to(self.device)
        returns = torch.from_numpy(data['returns']).to(self.device)
        old_values = torch.from_numpy(data['values']).to(self.device)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        num_batches = 0
        
        # Multiple epochs of optimization
        for epoch in range(self.epochs_per_update):
            # Create random mini-batches
            batch_indices = torch.randperm(len(observations))
            
            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_idx = batch_indices[start:end]
                
                # Get batch data
                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_values = old_values[batch_idx]
                
                # Forward pass
                _, log_probs, entropy, values = self.policy.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_param, self.clip_param
                )
                
                value_loss1 = (values - batch_returns) ** 2
                value_loss2 = (value_pred_clipped - batch_returns) ** 2
                
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                num_batches += 1
        
        self.update_count += 1
        
        # Return training statistics
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'entropy_loss': total_entropy_loss / num_batches,
            'total_loss': total_loss / num_batches,
        }
    
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        save_interval: int = 100,
        save_path: str = './checkpoints/ppo_baseline.pth'
    ):
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total number of environment timesteps
            log_interval: Logging interval (in updates)
            save_interval: Model saving interval (in updates)
            save_path: Path to save model
        """
        print(f"Starting PPO training for {total_timesteps:,} timesteps...")
        print(f"Buffer size: {self.buffer.buffer_size}")
        print(f"Updates needed: {total_timesteps // self.buffer.buffer_size}")
        
        timesteps_collected = 0
        
        with tqdm(total=total_timesteps, desc="PPO Training") as pbar:
            while timesteps_collected < total_timesteps:
                # Collect rollouts
                rollout_stats = self.collect_rollouts()
                timesteps_collected += self.buffer.buffer_size
                
                # Update policy
                train_stats = self.update_policy()
                
                # Update progress bar
                pbar.update(self.buffer.buffer_size)
                pbar.set_postfix({
                    'reward': f"{rollout_stats['avg_episode_reward']:.2f}",
                    'episodes': rollout_stats['num_episodes'],
                    'policy_loss': f"{train_stats['policy_loss']:.4f}"
                })
                
                # Logging
                if self.update_count % log_interval == 0:
                    print(f"\nUpdate {self.update_count}")
                    print(f"  Timesteps: {timesteps_collected:,}")
                    print(f"  Avg Episode Reward: {rollout_stats['avg_episode_reward']:.3f}")
                    print(f"  Avg Episode Length: {rollout_stats['avg_episode_length']:.1f}")
                    print(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {train_stats['value_loss']:.4f}")
                    
                    # Log to TensorBoard
                    if self.logger:
                        self.logger.log_dict(rollout_stats, step=self.update_count, prefix="ppo/rollout")
                        self.logger.log_dict(train_stats, step=self.update_count, prefix="ppo/train")
                        
                        if len(self.episode_rewards) > 0:
                            self.logger.log_scalar("ppo/recent_reward_mean", 
                                                 float(np.mean(list(self.episode_rewards)[-10:])), 
                                                 step=self.update_count)
                
                # Save model
                if self.update_count % save_interval == 0:
                    self.save_model(save_path.replace('.pth', f'_update_{self.update_count}.pth'))
        
        # Final save
        self.save_model(save_path)
        print(f"\nPPO training completed! Model saved to {save_path}")
        
        # Final evaluation
        final_reward = self.evaluate(num_episodes=10)
        print(f"Final evaluation reward: {final_reward:.3f}")
        
        if self.logger:
            self.logger.log_scalar("ppo/final_evaluation", final_reward, step=self.update_count)
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> float:
        """
        Evaluate trained policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Average episode reward
        """
        self.policy.eval()
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            
            while True:
                obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                
                with torch.no_grad():
                    action, _, _, _ = self.policy.get_action_and_value(obs_tensor)
                    
                action_np = action.cpu().numpy().item()
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                
                episode_reward += reward
                
                if render:
                    self.env.render()
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            
        return float(np.mean(episode_rewards))
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'episode_rewards': list(self.episode_rewards),
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)

def train_ppo_baseline(
    env_name: str,
    total_timesteps: int = 1000000,
    experiment_name: str = "ppo_baseline",
    device: torch.device = torch.device("auto")
):
    """
    Train PPO baseline for comparison with World Models.
    
    Args:
        env_name: Environment name
        total_timesteps: Total training timesteps
        experiment_name: Experiment name for logging
        device: Training device
    """
    # Setup device
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(device))
    
    print(f"Training PPO baseline on {env_name}")
    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps:,}")
    
    # Create logger
    logger = TrainingLogger("./logs", f"{experiment_name}_{env_name.replace('/', '_')}")
    
    # Create and train PPO agent
    agent = PPOAgent(
        env_name=env_name,
        device=device,
        logger=logger
    )
    
    save_path = f"./checkpoints/ppo_{env_name.replace('/', '_')}.pth"
    
    try:
        agent.train(
            total_timesteps=total_timesteps,
            log_interval=10,
            save_interval=50,
            save_path=save_path
        )
    finally:
        logger.close()
        agent.env.close()

if __name__ == "__main__":
    # Test PPO on Pong
    train_ppo_baseline(
        env_name="ALE/Pong-v5",
        total_timesteps=500000,  # Reduced for testing
        experiment_name="ppo_test"
    )
