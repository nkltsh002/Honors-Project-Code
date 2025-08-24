"""
Environment utilities for World Models

This module provides utilities for environment interaction, data collection,
and preprocessing for the World Models architecture.

Key features:
- Environment setup and configuration
- Frame preprocessing and normalization
- Random rollout collection for VAE training
- Real-time visualization during rollouts
- Support for multiple environments (Atari, LunarLander, CarRacing)

Based on Ha & Schmidhuber (2018): "World Models"
"""

import gymnasium as gym
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from collections import deque
import os
from PIL import Image
import imageio

class FramePreprocessor:
    """
    Frame preprocessing pipeline for consistent input format.
    
    Handles different environment observation formats and converts
    them to standardized 64x64 RGB frames for the VAE.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        """
        Initialize frame preprocessor.
        
        Args:
            target_size: Target frame size (height, width)
        """
        self.target_size = target_size
        
    def preprocess_atari_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess Atari game frames.
        
        Args:
            frame: Raw Atari frame (210, 160, 3) or similar
            
        Returns:
            Preprocessed frame (64, 64, 3)
        """
        # Convert to RGB if necessary
        if frame.shape[-1] == 3 and frame.dtype == np.uint8:
            frame_rgb = frame
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        frame_resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        
        return frame_resized.astype(np.uint8)
    
    def preprocess_carracing_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess CarRacing environment frames.
        
        Args:
            frame: CarRacing frame (96, 96, 3)
            
        Returns:
            Preprocessed frame (64, 64, 3)
        """
        # CarRacing frames are already RGB
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        return frame_resized.astype(np.uint8)
    
    def preprocess_lunarlander_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess LunarLander environment frames.
        
        Args:
            frame: LunarLander frame (400, 600, 3)
            
        Returns:
            Preprocessed frame (64, 64, 3)
        """
        # LunarLander frames are RGB
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        return frame_resized.astype(np.uint8)
    
    def preprocess_frame(self, frame: np.ndarray, env_name: str) -> np.ndarray:
        """
        Preprocess frame based on environment type.
        
        Args:
            frame: Raw environment frame
            env_name: Environment name
            
        Returns:
            Preprocessed frame (64, 64, 3)
        """
        if 'ALE/' in env_name:  # Atari games
            return self.preprocess_atari_frame(frame)
        elif 'CarRacing' in env_name:
            return self.preprocess_carracing_frame(frame)
        elif 'LunarLander' in env_name:
            return self.preprocess_lunarlander_frame(frame)
        else:
            # Default preprocessing
            if len(frame.shape) == 3:
                frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                return frame_resized.astype(np.uint8)
            else:
                # Convert grayscale to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                frame_resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_AREA)
                return frame_resized.astype(np.uint8)

class ActionProcessor:
    """
    Action processing for different environment types.
    
    Handles discrete/continuous action spaces and provides
    consistent interface for controller training.
    """
    
    def __init__(self, env_name: str, action_space):
        """
        Initialize action processor.
        
        Args:
            env_name: Environment name
            action_space: Gym action space
        """
        self.env_name = env_name
        self.action_space = action_space
        
        # Determine action type
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_type = 'discrete'
            self.action_size = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            self.action_type = 'continuous'
            self.action_size = action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")
    
    def process_action(self, action: Union[int, np.ndarray, torch.Tensor]) -> Any:
        """
        Process action for environment execution.
        
        Args:
            action: Raw action from controller
            
        Returns:
            Processed action for environment
        """
        if self.action_type == 'discrete':
            if isinstance(action, torch.Tensor):
                return action.cpu().numpy().item()
            elif isinstance(action, np.ndarray):
                return action.item()
            else:
                return int(action)
        else:
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Clip to action space bounds
            return np.clip(action, self.action_space.low, self.action_space.high)
    
    def random_action(self) -> Any:
        """Get random action from action space"""
        return self.action_space.sample()
    
    def action_to_tensor(self, action: Any) -> torch.Tensor:
        """
        Convert environment action to tensor format.
        
        Args:
            action: Environment action
            
        Returns:
            Action tensor for network input
        """
        if self.action_type == 'discrete':
            # One-hot encoding for discrete actions
            action_tensor = torch.zeros(self.action_size, dtype=torch.float32)
            action_tensor[action] = 1.0
            return action_tensor
        else:
            # Direct tensor conversion for continuous actions
            return torch.tensor(action, dtype=torch.float32)

class EnvironmentWrapper:
    """
    Unified environment wrapper for World Models training.
    
    Provides consistent interface across different environments
    with frame preprocessing, action processing, and data collection.
    """
    
    def __init__(
        self, 
        env_name: str,
        render_mode: str = 'rgb_array',
        seed: Optional[int] = None
    ):
        """
        Initialize environment wrapper.
        
        Args:
            env_name: Environment name
            render_mode: Render mode for visualization
            seed: Random seed for reproducibility
        """
        self.env_name = env_name
        self.render_mode = render_mode
        self.seed = seed
        
        # Create environment
        try:
            self.env = gym.make(env_name, render_mode=render_mode)
        except Exception as e:
            print(f"Failed to create environment {env_name}: {e}")
            print("Trying without render_mode...")
            self.env = gym.make(env_name)
        
        # Set seed
        if seed is not None:
            self.env.reset(seed=seed)
        
        # Initialize processors
        self.frame_processor = FramePreprocessor()
        self.action_processor = ActionProcessor(env_name, self.env.action_space)
        
        # Environment info
        self.action_size = self.action_processor.action_size
        self.action_type = self.action_processor.action_type
        
        print(f"Environment: {env_name}")
        print(f"Action space: {self.env.action_space} ({self.action_type})")
        print(f"Observation space: {self.env.observation_space}")
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return preprocessed observation"""
        obs, info = self.env.reset()
        processed_frame = self.frame_processor.preprocess_frame(obs, self.env_name)
        return processed_frame, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take environment step with action processing.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        processed_action = self.action_processor.process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(processed_action)
        processed_frame = self.frame_processor.preprocess_frame(obs, self.env_name)
        
        return processed_frame, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render environment"""
        try:
            return self.env.render()
        except:
            return None
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def random_action(self) -> Any:
        """Get random action"""
        return self.action_processor.random_action()
    
    def action_to_tensor(self, action: Any) -> torch.Tensor:
        """Convert action to tensor"""
        return self.action_processor.action_to_tensor(action)

def collect_random_rollouts(
    env_name: str,
    num_rollouts: int,
    rollout_length: int = 1000,
    save_dir: str = './data',
    visualize: bool = False,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect random rollouts for VAE training.
    
    Args:
        env_name: Environment name
        num_rollouts: Number of rollouts to collect
        rollout_length: Maximum length per rollout
        save_dir: Directory to save data
        visualize: Whether to visualize rollouts
        seed: Random seed
        
    Returns:
        Tuple of (frames, actions, rewards) arrays
    """
    print(f"Collecting {num_rollouts} random rollouts from {env_name}...")
    
    # Create environment
    env = EnvironmentWrapper(env_name, render_mode='rgb_array', seed=seed)
    
    # Storage
    all_frames = []
    all_actions = []
    all_rewards = []
    
    # Visualization setup
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
    
    for rollout_idx in range(num_rollouts):
        # Reset environment
        frame, info = env.reset()
        
        # Storage for current rollout
        rollout_frames = [frame]
        rollout_actions = []
        rollout_rewards = []
        
        total_reward = 0
        
        for step in range(rollout_length):
            # Take random action
            action = env.random_action()
            
            # Environment step
            next_frame, reward, terminated, truncated, info = env.step(action)
            
            # Store data
            rollout_frames.append(next_frame)
            rollout_actions.append(env.action_to_tensor(action).numpy())
            rollout_rewards.append(reward)
            
            total_reward += reward
            
            # Visualization
            if visualize and rollout_idx < 3:  # Show first few rollouts
                ax.clear()
                ax.imshow(frame)
                ax.set_title(f"Rollout {rollout_idx+1}/{num_rollouts}, Step {step+1}, Reward: {total_reward:.2f}")
                ax.axis('off')
                plt.pause(0.01)
            
            # Check termination
            if terminated or truncated:
                break
                
            frame = next_frame
        
        # Store rollout data
        all_frames.append(np.array(rollout_frames[:-1]))  # Exclude last frame
        all_actions.append(np.array(rollout_actions))
        all_rewards.append(np.array(rollout_rewards))
        
        print(f"Rollout {rollout_idx+1}/{num_rollouts}: {len(rollout_frames)-1} steps, total reward: {total_reward:.2f}")
    
    if visualize:
        plt.ioff()
        plt.close()
    
    env.close()
    
    # Convert to arrays with padding for variable lengths
    max_length = max(len(frames) for frames in all_frames)
    
    # Pad sequences
    padded_frames = []
    padded_actions = []
    padded_rewards = []
    
    for frames, actions, rewards in zip(all_frames, all_actions, all_rewards):
        seq_len = len(frames)
        
        # Pad frames
        pad_frames = np.zeros((max_length, 64, 64, 3), dtype=np.uint8)
        pad_frames[:seq_len] = frames
        padded_frames.append(pad_frames)
        
        # Pad actions
        pad_actions = np.zeros((max_length, env.action_size), dtype=np.float32)
        pad_actions[:seq_len] = actions
        padded_actions.append(pad_actions)
        
        # Pad rewards
        pad_rewards = np.zeros(max_length, dtype=np.float32)
        pad_rewards[:seq_len] = rewards
        padded_rewards.append(pad_rewards)
    
    # Stack into final arrays
    frames_array = np.array(padded_frames)  # (num_rollouts, max_length, 64, 64, 3)
    actions_array = np.array(padded_actions)  # (num_rollouts, max_length, action_size)
    rewards_array = np.array(padded_rewards)  # (num_rollouts, max_length)
    
    print(f"Collected data shapes:")
    print(f"  Frames: {frames_array.shape}")
    print(f"  Actions: {actions_array.shape}")
    print(f"  Rewards: {rewards_array.shape}")
    
    # Save data
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{env_name.replace("/", "_")}_rollouts.npz')
    
    np.savez_compressed(
        save_path,
        frames=frames_array,
        actions=actions_array,
        rewards=rewards_array,
        env_name=env_name,
        num_rollouts=num_rollouts,
        rollout_length=rollout_length
    )
    
    print(f"Saved rollout data to {save_path}")
    
    return frames_array, actions_array, rewards_array

def visualize_rollout(
    frames: np.ndarray,
    actions: Optional[np.ndarray] = None,
    rewards: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    fps: int = 30
):
    """
    Visualize a rollout as video or image sequence.
    
    Args:
        frames: Frame sequence (seq_len, 64, 64, 3)
        actions: Action sequence (seq_len, action_size)
        rewards: Reward sequence (seq_len,)
        save_path: Path to save video
        fps: Frames per second for video
    """
    seq_len = len(frames)
    
    if save_path:
        # Save as video
        with imageio.get_writer(save_path, fps=fps) as writer:
            for i in range(seq_len):
                frame = frames[i]
                
                # Add text overlay if actions/rewards provided
                if actions is not None or rewards is not None:
                    frame_with_text = frame.copy()
                    
                    if actions is not None:
                        action_text = f"Action: {np.argmax(actions[i]) if actions[i].ndim > 0 else actions[i]}"
                        cv2.putText(frame_with_text, action_text, (5, 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    if rewards is not None:
                        reward_text = f"Reward: {rewards[i]:.2f}"
                        cv2.putText(frame_with_text, reward_text, (5, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    writer.append_data(frame_with_text)
                else:
                    writer.append_data(frame)
                    
        print(f"Saved rollout video to {save_path}")
    else:
        # Display with matplotlib
        plt.figure(figsize=(12, 8))
        for i in range(min(seq_len, 100)):  # Show first 100 frames
            plt.subplot(10, 10, i+1)
            plt.imshow(frames[i])
            plt.axis('off')
            if rewards is not None:
                plt.title(f"R:{rewards[i]:.1f}", fontsize=8)
        
        plt.tight_layout()
        plt.show()

def test_environment():
    """Test environment functionality"""
    print("Testing environment utilities...")
    
    # Test environments
    environments = ['ALE/Pong-v5', 'LunarLander-v2']
    
    for env_name in environments:
        try:
            print(f"\nTesting {env_name}...")
            
            # Create environment
            env = EnvironmentWrapper(env_name, seed=42)
            
            # Test reset
            frame, info = env.reset()
            print(f"Initial frame shape: {frame.shape}")
            
            # Test a few steps
            total_reward = 0
            for step in range(10):
                action = env.random_action()
                frame, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"Completed {step+1} steps, total reward: {total_reward:.2f}")
            
            env.close()
            
        except Exception as e:
            print(f"Error testing {env_name}: {e}")
    
    print("Environment test completed!")

if __name__ == "__main__":
    test_environment()
