#!/usr/bin/env python3
"""
Curriculum Trainer with Real-Time Visual Feedback for World Models

This script trains World Models sequentially across a curriculum of increasingly complex environments:
Pong -> LunarLander -> Breakout -> CarRacing, with advanced real-time visualization and progress tracking.

NEW FEATURES (Updated):
- Real-time rendering of live rollouts during training (every N generations)
- Gymnasium render_mode="human" with fallback to rgb_array
- Environment-specific visualization frequency (Pong: every gen, others: every 5 gens)
- Enhanced progress bars with curriculum advancement tracking
- Video recording in ./runs/curriculum/<env_id>/videos/ directories
- Proper threshold-based curriculum progression (only advances when target reached)
- Compatible with Python 3.12, PyTorch 2.8.0+cpu, and Gymnasium 1.2.0

VISUALIZATION FEATURES:
- Live rollout window every N generations showing agent performance
- Progress display: [Env | Generation | Mean Score | Threshold]
- Real-time progress bars with completion percentage
- Automatic video recording with timestamp and generation info
- Keyboard interrupt support (Ctrl+C) to skip individual rollouts

CURRICULUM SYSTEM:
- Only advances to next task when mean score >= threshold
- Consistent performance required (5-generation average)
- Visual feedback for task completion status
- Comprehensive final report with success rates

USAGE:
python3 curriculum_trainer_visual.py --device cpu --max-generations 200 --episodes-per-eval 5 --visualize True --record-video True

Author: GitHub Copilot
Updated: August 2025
"""

# Ensure we're in the repository root
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tools.ensure_cwd import chdir_repo_root
chdir_repo_root()

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import traceback
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import threading
import queue
import warnings
import subprocess

# Suppress gymnasium deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import World Models components
sys.path.insert(0, os.getcwd())
try:
    import torch
    import torch.nn as nn
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo, GrayscaleObservation, ResizeObservation, FrameStackObservation
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    import tqdm

    # Import and register ALE environments
    import ale_py
    gym.register_envs(ale_py)

    from models.conv_vae_dynamic import ConvVAE
    from models.mdnrnn import MDNRNN, mdn_loss_function
    from models.controller import Controller, CMAESController
    from tools.dream_env import DreamEnvironment
    from tools.dataset_utils import FramesToLatentConverter
    from env_hints import print_env_error_hint

    print("[IMPORT] All imports completed successfully!")

except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install required packages:")
    print("pip install torch gymnasium matplotlib tqdm opencv-python")
    sys.exit(1)

print("[MODULE] Module initialization complete!")


def resolve_curriculum(prefer_box2d: Optional[bool] = None) -> List[Tuple[str, float]]:
    """
    Dynamically resolve curriculum based on available environments.

    Tries to create Box2D environments (LunarLander-v3, CarRacing-v3) with timeout.
    Falls back to Classic Control environments if Box2D is not available.

    Args:
        prefer_box2d: Force Box2D preference (True/False) or auto-detect (None)

    Returns:
        List of (env_id, threshold_score) tuples for the curriculum
    """
    def test_env_creation(env_id: str) -> bool:
        """Test if environment can be created."""
        try:
            env = gym.make(env_id)
            env.close()
            return True
        except Exception as e:
            print_env_error_hint(env_id, e)
            return False

    # Box2D curriculum (preferred if available)
    box2d_curriculum = [
        ("ALE/Pong-v5", 18.0),         # ALE Atari Pong
        ("LunarLander-v3", 200.0),     # Box2D Lunar Lander
        ("ALE/Breakout-v5", 50.0),     # ALE Atari Breakout
        ("CarRacing-v3", 800.0)        # Box2D Car Racing
    ]

    # Classic Control fallback curriculum (reliable)
    classic_curriculum = [
        ("ALE/Pong-v5", 18.0),         # ALE Atari Pong
        ("ALE/Breakout-v5", 50.0),     # ALE Atari Breakout
        ("CartPole-v1", 475.0),        # Classic Control CartPole (near-solved: 475/500)
        ("Acrobot-v1", -100.0)         # Classic Control Acrobot (-100 is challenging but feasible)
    ]

    # Force preference if specified
    if prefer_box2d is True:
        print("🎯 CURRICULUM: Box2D environments forced by user preference")
        return box2d_curriculum
    elif prefer_box2d is False:
        print("🎯 CURRICULUM: Classic Control environments forced by user preference")
        return classic_curriculum

    # Auto-detect Box2D availability
    print("🔍 AUTO-DETECTING: Testing Box2D environment availability...")

    box2d_available = (
        test_env_creation("LunarLander-v3") and
        test_env_creation("CarRacing-v3")
    )

    if box2d_available:
        print("✅ CURRICULUM: Box2D environments detected and working")
        print("   Tasks: Pong → LunarLander → Breakout → CarRacing")
        return box2d_curriculum
    else:
        print("⚠️  CURRICULUM: Box2D environments not available, using Classic Control fallback")
        print("   Tasks: Pong → Breakout → CartPole → Acrobot")
        return classic_curriculum


@dataclass
class CurriculumTask:
    """Defines a curriculum task with environment and success criteria."""
    env_id: str
    threshold_score: float
    max_episode_steps: int = 1000
    solved: bool = False
    best_score: float = float('-inf')
    generations_trained: int = 0

@dataclass
class TrainingConfig:
    """Training configuration for curriculum learning."""
    device: str = 'cpu'
    max_generations: int = 1000
    episodes_per_eval: int = 10
    checkpoint_dir: str = './runs/curriculum_visual'
    visualize: bool = True
    record_video: bool = False
    video_every_n_gens: int = 10
    quick_mode: bool = False  # New: quick mode for fast testing
    prefer_box2d: Optional[bool] = None  # New: Box2D preference (True/False/None for auto-detect)

    # Visualization settings
    show_rollout_every_n_gens: Optional[Dict[str, int]] = None  # Environment-specific rendering frequency
    render_mode: str = "human"  # Gymnasium render mode
    video_fps: int = 30

    # GPU Memory Optimization Settings
    use_amp: bool = True  # Automatic Mixed Precision
    use_tf32: bool = True  # TensorFloat-32
    vae_img_size: int = 64  # VAE image size for memory efficiency
    vae_batch_size: int = 32  # VAE batch size
    grad_accumulation_steps: int = 1  # Gradient accumulation
    max_episode_steps: int = 1000

    # VAE hyperparameters
    vae_latent_size: int = 32
    vae_epochs: int = 5
    vae_batch_size: int = 32

    # MDN-RNN hyperparameters
    rnn_size: int = 128
    num_mixtures: int = 5
    mdnrnn_epochs: int = 5
    mdnrnn_batch_size: int = 16

    # Controller hyperparameters
    controller_hidden_size: int = 64
    cma_population_size: int = 16
    cma_sigma: float = 0.1
    patience: int = 50  # Early stopping patience

class CurriculumTrainer:
    """Main curriculum trainer with visualization."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set up directories first
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up video directories
        self.video_base_dir = self.checkpoint_dir / "videos"
        self.video_base_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging before GPU configuration
        self.setup_logging()

        # Configure GPU memory optimizations
        self._configure_gpu_optimizations()

        # Configure visualization settings per environment
        if config.show_rollout_every_n_gens is None:
            self.rollout_frequency = {
                "ALE/Pong-v5": 1,             # Show every generation for Pong
                "PongNoFrameskip-v5": 1,
                "PongNoFrameskip-v4": 1,
                "LunarLander-v2": 5,          # Every 5 generations for others
                "LunarLander-v3": 5,
                "ALE/Breakout-v5": 5,         # Updated Atari naming
                "BreakoutNoFrameskip-v5": 5,
                "BreakoutNoFrameskip-v4": 5,
                "CarRacing-v3": 5,
                "CarRacing-v3": 5
            }
        else:
            self.rollout_frequency = config.show_rollout_every_n_gens

        # Define curriculum dynamically based on available environments
        curriculum_tasks = resolve_curriculum(config.prefer_box2d)
        self.curriculum = [CurriculumTask(env_id, threshold) for env_id, threshold in curriculum_tasks]

        # Apply quick mode modifications if enabled
        if config.quick_mode:
            self.logger.info("[QUICK MODE] Using reduced thresholds for fast debugging.")
            # Reduce thresholds for quick testing
            quick_thresholds = [5.0, 50.0, 10.0, 200.0]
            for i, threshold in enumerate(quick_thresholds):
                self.curriculum[i].threshold_score = threshold

            # Reduce hyperparameters for faster training
            self.config.cma_population_size = max(4, self.config.cma_population_size // 4)
            self.config.vae_epochs = max(1, self.config.vae_epochs // 2)
            self.config.mdnrnn_epochs = max(1, self.config.mdnrnn_epochs // 2)
            self.config.patience = max(5, self.config.patience // 10)

            self.logger.info(f"[QUICK MODE] Reduced thresholds: Pong=5, LunarLander=50, Breakout=10, CarRacing=200")
            self.logger.info(f"[QUICK MODE] Reduced population size: {self.config.cma_population_size}")

        # Training state
        self.current_task_idx = 0
        self.global_generation = 0
        self.training_start_time = time.time()

        # Models (will be created per environment)
        self.vae = None
        self.mdnrnn = None
        self.controller = None
        self.dream_env = None

        # Progress tracking
        self.progress_queue = queue.Queue()
        self.visualization_thread = None
        self.stop_visualization = threading.Event()

        self.logger.info("Curriculum Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Curriculum: {len(self.curriculum)} environments")

    def setup_logging(self):
        """Set up comprehensive logging."""
        log_dir = self.checkpoint_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # File logging
        log_file = log_dir / f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

        self.logger = logging.getLogger('CurriculumTrainer')

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir / "tensorboard")

        # CSV logging
        self.csv_file = log_dir / "curriculum_progress.csv"
        with open(self.csv_file, 'w') as f:
            f.write("timestamp,env_id,generation,mean_score,best_score,threshold,solved,time_elapsed\n")

    def _configure_gpu_optimizations(self):
        """Configure GPU memory optimizations for RTX 3050."""
        if self.device.type == 'cuda' and torch.cuda.is_available():
            # Enable TensorFloat-32 for better performance on modern GPUs
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("[GPU] TensorFloat-32 enabled")

            # Configure memory settings
            torch.cuda.empty_cache()
            self.logger.info(f"[GPU] CUDA Device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"[GPU] Mixed Precision: {self.config.use_amp}")
            self.logger.info(f"[GPU] VAE Image Size: {self.config.vae_img_size}")
            self.logger.info(f"[GPU] VAE Batch Size: {self.config.vae_batch_size}")
            self.logger.info(f"[GPU] Gradient Accumulation Steps: {self.config.grad_accumulation_steps}")
        else:
            self.logger.info("[GPU] Using CPU mode")

        # Create AMP scaler if using mixed precision
        if self.config.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def create_env(self, env_id: str, record_video: bool = False, video_dir: Optional[Path] = None) -> gym.Env:
        """Create and configure environment."""
        try:
            env = gym.make(env_id, render_mode="rgb_array" if record_video else None)

            # Add video recording wrapper if needed
            if record_video and video_dir:
                video_dir.mkdir(parents=True, exist_ok=True)
                env = RecordVideo(
                    env,
                    str(video_dir),
                    episode_trigger=lambda x: True,  # Record all episodes
                    name_prefix=f"{env_id}_gen{self.global_generation}"
                )

            # Apply preprocessing for Atari games
            if "NoFrameskip" in env_id:
                env = GrayscaleObservation(env)
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)
            elif env_id == "LunarLander-v2":
                # LunarLander doesn't need frame preprocessing
                pass
            elif env_id == "CarRacing-v3":
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)

            return env

        except Exception as e:
            self.logger.error(f"Failed to create environment {env_id}: {e}")
            raise

    def collect_random_data(self, env_id: str, num_episodes: int = 100) -> str:
        """Collect random rollout data for VAE training."""
        self.logger.info(f"Collecting {num_episodes} random episodes from {env_id}")

        env = self.create_env(env_id)
        data_dir = self.checkpoint_dir / env_id / "random_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        episodes_data = []

        try:
            for episode in tqdm.tqdm(range(num_episodes), desc="Collecting data"):
                obs, info = env.reset()
                episode_frames = []
                episode_actions = []
                episode_rewards = []
                done = False

                while not done:
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    episode_frames.append(obs)
                    episode_actions.append(action)
                    episode_rewards.append(reward)

                    obs = next_obs
                    done = terminated or truncated

                episodes_data.append({
                    'frames': np.array(episode_frames),
                    'actions': np.array(episode_actions),
                    'rewards': np.array(episode_rewards)
                })

            # Save collected data
            data_file = data_dir / "episodes.npz"
            np.savez(data_file, episodes=episodes_data)

            self.logger.info(f"Collected data saved to {data_file}")
            return str(data_file)

        finally:
            env.close()

    def train_vae(self, env_id: str, data_file: str) -> str:
        """Train VAE on collected data."""
        self.logger.info(f"Training VAE for {env_id}")

        # Load data
        data = np.load(data_file, allow_pickle=True)
        episodes = data['episodes']

        # Extract all frames
        all_frames = []
        for episode in episodes:
            frames = episode['frames']
            all_frames.extend(frames)

        all_frames = np.array(all_frames)
        total_frames = len(all_frames)
        self.logger.info(f"Training VAE on {total_frames} frames")
        self.logger.info(f"Original frame shape: {all_frames.shape}")

        # CRITICAL FIX: Handle multi-channel frames and resize to VAE input size
        if len(all_frames.shape) == 4 and all_frames.shape[-1] > 4:
            self.logger.info(f"Detected {all_frames.shape[-1]} channels, slicing to RGB (first 3)")
            all_frames = all_frames[..., :3]  # Keep only RGB channels
            self.logger.info(f"Sliced frame shape: {all_frames.shape}")

        # Resize frames to match VAE expected input size
        import cv2
        target_size = self.config.vae_img_size
        if all_frames.shape[1] != target_size or all_frames.shape[2] != target_size:
            self.logger.info(f"Resizing frames from {all_frames.shape[1:3]} to {target_size}x{target_size}")
            resized_frames = []
            for frame in all_frames:
                resized_frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
                resized_frames.append(resized_frame)
            all_frames = np.array(resized_frames)
            self.logger.info(f"Resized frame shape: {all_frames.shape}")

        # Calculate memory requirements
        frame_memory = all_frames.nbytes / 1024 / 1024 / 1024  # GB
        self.logger.info(f"Frame data size: {frame_memory:.2f} GB")

        # Create VAE - after channel preprocessing and resizing
        if len(all_frames.shape) == 4:  # Color or grayscale
            input_channels = all_frames.shape[-1]
        else:
            input_channels = 1

        self.logger.info(f"Creating VAE with {input_channels} input channels for frames shape {all_frames.shape}")

        self.vae = ConvVAE(
            img_channels=input_channels,
            img_size=self.config.vae_img_size,
            latent_dim=self.config.vae_latent_size
        )
        self.vae.to(self.device)        # Use streaming data loading if dataset is too large (>1GB)
        if frame_memory > 1.0:
            return self._train_vae_streaming(env_id, all_frames)
        else:
            return self._train_vae_batch(env_id, all_frames)

    def _train_vae_streaming(self, env_id: str, all_frames: np.ndarray) -> str:
        """Train VAE with streaming data loading for large datasets."""
        self.logger.info("Using streaming data loading for memory efficiency")

        # The VAE model is already created and moved to device in the main method

        # Prepare streaming dataset with smaller chunks
        chunk_size = min(500, len(all_frames) // 20)  # Smaller chunks for RTX 3050
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        for epoch in range(self.config.vae_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Process data in chunks
            indices = np.random.permutation(len(all_frames))
            for chunk_start in range(0, len(all_frames), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(all_frames))
                chunk_indices = indices[chunk_start:chunk_end]

                # Load small chunk into memory
                chunk_frames = all_frames[chunk_indices]
                frames_tensor = torch.FloatTensor(chunk_frames).permute(0, 3, 1, 2) / 255.0                # Process chunk in mini-batches
                for batch_start in range(0, len(frames_tensor), self.config.vae_batch_size):
                    batch_end = min(batch_start + self.config.vae_batch_size, len(frames_tensor))
                    batch = frames_tensor[batch_start:batch_end].to(self.device, non_blocking=True)

                    optimizer.zero_grad()

                    # Use mixed precision if enabled
                    if self.config.use_amp and self.scaler:
                        with torch.cuda.amp.autocast():
                            recon, mu, logvar = self.vae(batch)
                            recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            loss = recon_loss + kl_loss

                        # Scaled backward pass
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Standard precision training
                        recon, mu, logvar = self.vae(batch)
                        recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + kl_loss

                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    # Clear cache periodically
                    if num_batches % 5 == 0 and self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                # Clear chunk from memory immediately
                del frames_tensor, chunk_frames
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.logger.info(f"VAE Epoch {epoch+1}/{self.config.vae_epochs}: Loss = {avg_loss:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar(f'{env_id}/VAE_Loss', avg_loss, epoch)

        # Save VAE
        vae_path = self.checkpoint_dir / env_id / "vae.pt"
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.vae.state_dict(), vae_path)

        self.logger.info(f"Streaming VAE saved to {vae_path}")
        return str(vae_path)

    def _train_vae_batch(self, env_id: str, all_frames: np.ndarray) -> str:
        """Train VAE with full batch loading for small datasets."""
        self.logger.info("Using batch data loading")

        # Prepare data loader with optimized batch size
        frames_tensor = torch.FloatTensor(all_frames).permute(0, 3, 1, 2) / 255.0
        dataset = torch.utils.data.TensorDataset(frames_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.vae_batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=0  # Keep 0 for Windows compatibility
        )

        # Train VAE with mixed precision support
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        for epoch in range(self.config.vae_epochs):
            epoch_loss = 0.0
            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Use mixed precision if enabled
                if self.config.use_amp and self.scaler:
                    with torch.cuda.amp.autocast():
                        recon, mu, logvar = self.vae(batch)

                        # VAE loss
                        recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + kl_loss

                    # Scaled backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    recon, mu, logvar = self.vae(batch)

                    # VAE loss
                    recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss

                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

                # Clear cache periodically for memory management
                if batch_idx % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"VAE Epoch {epoch+1}/{self.config.vae_epochs}: Loss = {avg_loss:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar(f'{env_id}/VAE_Loss', avg_loss, epoch)

        # Save VAE
        vae_path = self.checkpoint_dir / env_id / "vae.pt"
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.vae.state_dict(), vae_path)

        self.logger.info(f"Batch VAE saved to {vae_path}")
        return str(vae_path)

    def encode_data_to_latents(self, env_id: str, data_file: str, vae_path: str) -> str:
        """Encode frame data to latent sequences for MDN-RNN training."""
        self.logger.info(f"Encoding data to latents for {env_id}")

        # Instantiate and load VAE
        self.vae = ConvVAE(
            img_channels=3,
            img_size=self.config.vae_img_size,
            latent_dim=self.config.vae_latent_size
        )
        self.vae.to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.vae.eval()

        # Load data
        data = np.load(data_file, allow_pickle=True)
        episodes = data['episodes']

        latent_episodes = []

        with torch.no_grad():
            for episode_data in tqdm.tqdm(episodes, desc="Encoding episodes"):
                episode = episode_data.item()
                frames = episode['frames']
                actions = episode['actions']
                rewards = episode['rewards']

                # Encode frames to latents
                frames_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0
                frames_tensor = frames_tensor.to(self.device)

                _, mu, _ = self.vae(frames_tensor)
                latents = mu.cpu().numpy()

                latent_episodes.append({
                    'latents': latents,
                    'actions': actions,
                    'rewards': rewards
                })

        # Save latent data
        latent_file = self.checkpoint_dir / env_id / "latent_episodes.npz"
        np.savez(latent_file, episodes=latent_episodes)

        self.logger.info(f"Latent data saved to {latent_file}")
        return str(latent_file)

    def train_mdnrnn(self, env_id: str, latent_file: str) -> str:
        """Train MDN-RNN on latent sequences."""
        self.logger.info(f"Training MDN-RNN for {env_id}")

        # Load latent data
        data = np.load(latent_file, allow_pickle=True)
        episodes = data['episodes']

        # Determine action dimensionality
        first_episode = episodes[0].item()
        action_dim = np.array(first_episode['actions']).shape[-1] if len(np.array(first_episode['actions']).shape) > 1 else 1

        # Create MDN-RNN
        self.mdnrnn = MDNRNN(
            z_dim=self.config.vae_latent_size,
            action_dim=action_dim,
            rnn_size=self.config.rnn_size,
            num_mixtures=self.config.num_mixtures
        )
        self.mdnrnn.to(self.device)

        # Prepare sequences for training
        sequences = []
        for episode_data in episodes:
            episode = episode_data.item()
            latents = episode['latents']
            actions = episode['actions']

            # Create sequences of [z_t, a_t] -> z_{t+1}
            for t in range(len(latents) - 1):
                z_t = latents[t]
                a_t = actions[t] if action_dim > 1 else [actions[t]]
                z_next = latents[t + 1]

                sequences.append({
                    'z_t': z_t,
                    'a_t': a_t,
                    'z_next': z_next
                })

        self.logger.info(f"Training MDN-RNN on {len(sequences)} sequences")

        # Create data loader
        z_t_batch = torch.FloatTensor([s['z_t'] for s in sequences])
        a_t_batch = torch.FloatTensor([s['a_t'] for s in sequences])
        z_next_batch = torch.FloatTensor([s['z_next'] for s in sequences])

        dataset = torch.utils.data.TensorDataset(z_t_batch, a_t_batch, z_next_batch)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.mdnrnn_batch_size,
            shuffle=True
        )

        # Train MDN-RNN
        optimizer = torch.optim.Adam(self.mdnrnn.parameters(), lr=1e-3)

        for epoch in range(self.config.mdnrnn_epochs):
            epoch_loss = 0.0
            for z_t, a_t, z_next in dataloader:
                z_t, a_t, z_next = z_t.to(self.device), a_t.to(self.device), z_next.to(self.device)

                # Add sequence dimension
                z_t = z_t.unsqueeze(1)  # (batch, 1, z_dim)
                a_t = a_t.unsqueeze(1)  # (batch, 1, action_dim)

                optimizer.zero_grad()
                pi, mu, sigma = self.mdnrnn(z_t, a_t)

                # MDN loss
                loss = mdn_loss_function(pi, mu, sigma, z_next)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"MDN-RNN Epoch {epoch+1}/{self.config.mdnrnn_epochs}: Loss = {avg_loss:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar(f'{env_id}/MDNRNN_Loss', avg_loss, epoch)

        # Save MDN-RNN
        mdnrnn_path = self.checkpoint_dir / env_id / "mdnrnn.pt"
        torch.save(self.mdnrnn.state_dict(), mdnrnn_path)

        self.logger.info(f"MDN-RNN saved to {mdnrnn_path}")
        return str(mdnrnn_path)

    def create_dream_environment(self, env_id: str, vae_path: str, mdnrnn_path: str) -> DreamEnvironment:
        """Create dream environment for controller training."""
        self.logger.info(f"Creating dream environment for {env_id}")

        # Determine action space size
        real_env = self.create_env(env_id)
        if isinstance(real_env.action_space, gym.spaces.Discrete):
            action_space_size = real_env.action_space.n
        elif hasattr(real_env.action_space, 'shape') and real_env.action_space.shape is not None:
            action_space_size = real_env.action_space.shape[0]
        else:
            # Fallback: treat as scalar action
            action_space_size = 1
        real_env.close()

        self.dream_env = DreamEnvironment(
            vae_model_path=vae_path,
            mdnrnn_model_path=mdnrnn_path,
            action_space_size=int(action_space_size),
            max_episode_steps=200,  # Shorter for faster training
            device=str(self.device)
        )

        return self.dream_env

    def create_visualization_env(self, env_id: str) -> Optional[gym.Env]:
        """Create environment specifically for visualization with human rendering."""
        try:
            # Try to create with human render mode for real-time visualization
            env = gym.make(env_id, render_mode=self.config.render_mode)

            # Apply same preprocessing as training environment
            if "NoFrameskip" in env_id:
                env = GrayscaleObservation(env)
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)
            elif "LunarLander" in env_id:
                # LunarLander doesn't need frame preprocessing
                pass
            elif "CarRacing" in env_id:
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)

            return env

        except Exception as e:
            self.logger.warning(f"Failed to create visualization environment for {env_id}: {e}")
            # Fallback to rgb_array mode
            try:
                env = gym.make(env_id, render_mode="rgb_array")
                if "NoFrameskip" in env_id:
                    env = GrayscaleObservation(env)
                    env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                    env = FrameStackObservation(env, 4)
                elif "CarRacing" in env_id:
                    env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                    env = FrameStackObservation(env, 4)
                return env
            except Exception as e2:
                self.logger.error(f"Failed to create fallback environment: {e2}")
                return None

    def show_live_rollout(self, env_id: str, controller, generation: int, mean_score: float, threshold: float) -> float:
        """Show a live rollout with the current controller."""
        env = self.create_visualization_env(env_id)
        if env is None:
            return 0.0

        try:
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            # Load VAE if available for latent space conversion
            vae_path = self.checkpoint_dir / env_id / "vae_best.pt"
            use_vae = False

            if vae_path.exists() and self.vae is not None:
                try:
                    self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                    self.vae.eval()
                    use_vae = True
                except Exception as e:
                    self.logger.warning(f"Failed to load VAE: {e}")
                    use_vae = False

            print(f"\n[LIVE] {env_id} | Gen {generation:4d} | Score: {mean_score:7.2f} | Target: {threshold:6.1f}")
            print(f"[LIVE] Showing live rollout... (Press Ctrl+C to skip)")

            while not done and step_count < self.config.max_episode_steps:
                try:
                    if use_vae and self.vae is not None:
                        # Convert observation to latent using VAE
                        if len(obs.shape) == 3:
                            obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0) / 255.0
                        else:
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) / 255.0

                        obs_tensor = obs_tensor.to(self.device)

                        with torch.no_grad():
                            _, z, _ = self.vae(obs_tensor)
                            # Create dummy hidden state for controller
                            hidden = torch.zeros(1, self.config.rnn_size).to(self.device)
                            state = torch.cat([z, hidden], dim=1)

                            action_output = controller.get_action(state, deterministic=True)
                            action = action_output.cpu().numpy().flatten()
                    else:
                        # Use observation directly (fallback mode)
                        obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array([obs])
                        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            action_output = controller.get_action(obs_tensor, deterministic=True)
                            action = action_output.cpu().numpy().flatten()

                    # Convert to environment action format
                    if hasattr(env.action_space, 'n'):  # Discrete
                        action = int(np.argmax(action)) if len(action) > 1 else int(action[0])
                    else:  # Continuous
                        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                            action = np.clip(action, env.action_space.low, env.action_space.high)

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    step_count += 1

                    # Add small delay for human viewing
                    time.sleep(0.05)

                except KeyboardInterrupt:
                    print(f"\n[LIVE] Rollout interrupted by user")
                    break
                except Exception as e:
                    self.logger.warning(f"Error during live rollout: {e}")
                    break

            env.close()
            print(f"[LIVE] Rollout completed - Score: {total_reward:.2f}, Steps: {step_count}")
            return total_reward

        except Exception as e:
            self.logger.error(f"Failed to show live rollout: {e}")
            if env:
                env.close()
            return 0.0

    def record_video_rollout(self, env_id: str, controller, generation: int) -> bool:
        """Record a video of the current controller's performance."""
        video_dir = self.video_base_dir / env_id / f"generation_{generation:04d}"
        video_dir.mkdir(parents=True, exist_ok=True)

        try:
            env = self.create_env(env_id, record_video=True, video_dir=video_dir)
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            # Load VAE for latent conversion
            vae_path = self.checkpoint_dir / env_id / "vae_best.pt"
            use_vae = False

            if vae_path.exists() and self.vae is not None:
                try:
                    self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                    self.vae.eval()
                    use_vae = True
                except Exception:
                    use_vae = False

            while not done and step_count < self.config.max_episode_steps:
                if use_vae and self.vae is not None:
                    # Convert observation to latent using VAE
                    if len(obs.shape) == 3:
                        obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0) / 255.0
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0) / 255.0

                    obs_tensor = obs_tensor.to(self.device)

                    with torch.no_grad():
                        _, z, _ = self.vae(obs_tensor)
                        hidden = torch.zeros(1, self.config.rnn_size).to(self.device)
                        state = torch.cat([z, hidden], dim=1)

                        action_output = controller.get_action(state, deterministic=True)
                        action = action_output.cpu().numpy().flatten()
                else:
                    # Fallback mode
                    obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array([obs])
                    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        action_output = controller.get_action(obs_tensor, deterministic=True)
                        action = action_output.cpu().numpy().flatten()

                # Convert to environment action format
                if hasattr(env.action_space, 'n'):  # Discrete
                    action = int(np.argmax(action)) if len(action) > 1 else int(action[0])
                else:  # Continuous
                    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                        action = np.clip(action, env.action_space.low, env.action_space.high)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                step_count += 1

            env.close()
            self.logger.info(f"Video recorded for {env_id} generation {generation}: {total_reward:.2f} points")
            return True

        except Exception as e:
            self.logger.error(f"Failed to record video for {env_id}: {e}")
            return False

    def evaluate_controller_real_env(self, env_id: str, controller: Controller, num_episodes: Optional[int] = None, render: bool = False) -> Tuple[float, List[float]]:
        """Evaluate controller in real environment."""
        if num_episodes is None:
            num_episodes = self.config.episodes_per_eval

        env = self.create_env(env_id)
        episode_rewards = []

        try:
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0.0
                done = False
                step_count = 0

                # Initialize hidden state for MDN-RNN
                hidden = torch.zeros(1, self.config.rnn_size).to(self.device)

                # Ensure VAE is loaded
                if self.vae is None:
                    vae_path = self.checkpoint_dir / env_id / "vae.pt"
                    if not vae_path.exists():
                        raise RuntimeError(f"VAE model not found at {vae_path}")
                    self.vae = ConvVAE(
                        img_channels=3,
                        img_size=self.config.vae_img_size,
                        latent_dim=self.config.vae_latent_size
                    )
                    self.vae.to(self.device)
                    self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                    self.vae.eval()

                while not done and step_count < 1000:
                    # Convert observation to latent
                    if len(obs.shape) == 3:
                        obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0) / 255.0
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0) / 255.0

                    obs_tensor = obs_tensor.to(self.device)

                    with torch.no_grad():
                        # Get latent representation
                        _, z, _ = self.vae(obs_tensor)

                        # Get action from controller
                        action_output = controller.get_action(z, hidden, deterministic=True)
                        if isinstance(action_output, tuple):
                            action_tensor = action_output[0]
                        else:
                            action_tensor = action_output

                        action = action_tensor.cpu().numpy()
                        if len(action.shape) > 1:
                            action = action[0]

                        # Convert to environment action format
                        if hasattr(env.action_space, 'n'):  # Discrete
                            action = int(np.argmax(action))
                        elif isinstance(env.action_space, gym.spaces.Box):  # Continuous
                            action = np.clip(action, env.action_space.low, env.action_space.high)
                        else:
                            # Fallback: do not clip
                            pass

                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += float(reward)
                    done = terminated or truncated
                    step_count += 1

                    if render and episode == 0:  # Only render first episode
                        if hasattr(env, 'render'):
                            env.render()

                episode_rewards.append(episode_reward)

                if render and episode == 0:
                    time.sleep(0.1)  # Brief pause for visualization

            return float(np.mean(episode_rewards)), episode_rewards

        finally:
            env.close()

    def train_controller_cmaes(self, env_id: str, task: CurriculumTask, vae_path: str, mdnrnn_path: str) -> bool:
        """Train controller using CMA-ES."""
        self.logger.info(f"Training controller for {env_id} using CMA-ES")

        # Create dream environment
        dream_env = self.create_dream_environment(env_id, vae_path, mdnrnn_path)

        # Create controller
        input_size = self.config.vae_latent_size + self.config.rnn_size
        action_size = dream_env.action_space_size

        self.controller = Controller(
            input_size=input_size,
            action_size=action_size,
            hidden_sizes=(self.config.controller_hidden_size,),
            action_type='continuous' if not hasattr(dream_env, 'discrete_actions') else 'discrete'
        )

        # CMA-ES optimizer
        cmaes_optimizer = CMAESController(
            controller=self.controller,
            population_size=self.config.cma_population_size,
            sigma=self.config.cma_sigma
        )

        best_score = float('-inf')
        patience_counter = 0
        generation_scores = deque(maxlen=10)

        for generation in range(self.config.max_generations):
            self.global_generation = generation

            # Generate candidate solutions
            candidates = cmaes_optimizer.ask()
            fitness_values = []

            # Evaluate each candidate
            for candidate in candidates:
                # Set controller parameters
                self.controller.set_parameters(candidate)

                # Evaluate in dream environment
                dream_reward = self.evaluate_dream_environment(dream_env, episodes=3)
                fitness_values.append(dream_reward)

            # Update CMA-ES
            cmaes_optimizer.tell(candidates, np.array(fitness_values))

            # Get best candidate for real environment evaluation
            best_candidate = candidates[np.argmax(fitness_values)]
            self.controller.set_parameters(best_candidate)

            # Evaluate in real environment
            mean_score, episode_scores = self.evaluate_controller_real_env(
                env_id, self.controller, render=self.config.visualize
            )

            generation_scores.append(mean_score)

            # Update best score
            if mean_score > best_score:
                best_score = mean_score
                task.best_score = best_score
                patience_counter = 0

                # Save best controller
                controller_path = self.checkpoint_dir / env_id / "controller_best.pt"
                controller_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.controller.state_dict(), controller_path)
            else:
                patience_counter += 1

            # Logging and progress tracking
            self.log_training_progress(env_id, generation, mean_score, best_score, task.threshold_score)

            # Real-time visualization - show live rollout based on environment-specific frequency
            rollout_freq = self.rollout_frequency.get(env_id, 5)  # Default to every 5 generations
            if self.config.visualize and generation % rollout_freq == 0:
                try:
                    live_score = self.show_live_rollout(env_id, self.controller, generation, mean_score, task.threshold_score)
                    self.logger.info(f"Live rollout score: {live_score:.2f}")
                except Exception as e:
                    self.logger.warning(f"Failed to show live rollout: {e}")

            # Record video occasionally
            if self.config.record_video and generation % self.config.video_every_n_gens == 0:
                try:
                    self.record_video_rollout(env_id, self.controller, generation)
                except Exception as e:
                    self.logger.warning(f"Failed to record video: {e}")

            # Enhanced progress display with curriculum information
            progress_bar = "█" * int((mean_score / task.threshold_score) * 20) + "▒" * (20 - int((mean_score / task.threshold_score) * 20))
            progress_pct = min(100, (mean_score / task.threshold_score) * 100)

            print(f"\r[PROGRESS] {env_id:<25} | Gen {generation:4d} | Score: {mean_score:7.2f} | Target: {task.threshold_score:6.1f} | [{progress_bar}] {progress_pct:5.1f}%", end="", flush=True)

            # Check if task is solved - require consistent performance
            recent_avg = np.mean(list(generation_scores)) if len(generation_scores) >= 5 else mean_score
            if recent_avg >= task.threshold_score:
                print(f"\n[SUCCESS] {env_id} SOLVED! Average score: {recent_avg:.2f} >= {task.threshold_score}")
                self.logger.info(f"Task {env_id} SOLVED! Average score: {recent_avg:.2f} >= {task.threshold_score}")
                task.solved = True
                task.generations_trained = generation + 1
                return True

            # Early stopping
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping for {env_id} after {patience_counter} generations without improvement")
                break

        task.generations_trained = self.config.max_generations
        self.logger.info(f"Training completed for {env_id}. Best score: {best_score:.2f}, Target: {task.threshold_score}")
        return False

    def evaluate_dream_environment(self, dream_env: DreamEnvironment, episodes: int = 3) -> float:
        """Evaluate controller in dream environment."""
        total_reward = 0.0

        for _ in range(episodes):
            obs, info = dream_env.reset()
            episode_reward = 0.0
            done = False
            step_count = 0

            while not done and step_count < 200:  # Limit steps for speed
                with torch.no_grad():
                    # Extract latent and hidden states from observation
                    z = torch.FloatTensor(obs[:self.config.vae_latent_size]).unsqueeze(0).to(self.device)
                    h = torch.FloatTensor(obs[self.config.vae_latent_size:]).unsqueeze(0).to(self.device)

                    # Get action
                    if self.controller is None:
                        raise RuntimeError("Controller is not initialized before calling get_action.")
                    action_output = self.controller.get_action(z, h, deterministic=True)
                    if isinstance(action_output, tuple):
                        action = action_output[0].cpu().numpy()[0]
                    else:
                        action = action_output.cpu().numpy()[0]

                obs, reward, terminated, truncated, info = dream_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1

            total_reward += episode_reward

        return total_reward / episodes

    def record_evaluation_video(self, env_id: str, video_dir: Path):
        """Record a video of the current controller performance."""
        video_dir.mkdir(parents=True, exist_ok=True)

        env = self.create_env(env_id, record_video=True, video_dir=video_dir)

        try:
            if self.controller is None:
                raise RuntimeError("Controller is not initialized before recording evaluation video.")
            self.evaluate_controller_real_env(env_id, self.controller, num_episodes=1, render=False)
            self.logger.info(f"Video recorded for {env_id} at generation {self.global_generation}")
        except Exception as e:
            self.logger.error(f"Failed to record video: {e}")
        finally:
            env.close()

    def log_training_progress(self, env_id: str, generation: int, mean_score: float, best_score: float, threshold: float):
        """Log training progress to various outputs."""
        elapsed_time = time.time() - self.training_start_time

        # Console logging with progress
        progress_bar = "=" * int(20 * min(mean_score / threshold, 1.0))
        progress_spaces = " " * (20 - len(progress_bar))

        print(f"\r{env_id:20} | Gen {generation:4d} | "
              f"Score: {mean_score:7.2f} | Best: {best_score:7.2f} | "
              f"Target: {threshold:6.1f} | [{progress_bar}{progress_spaces}] "
              f"{100 * min(mean_score / threshold, 1.0):5.1f}%",
              end="", flush=True)

        if generation % 10 == 0:
            print()  # New line every 10 generations

        # TensorBoard logging
        self.writer.add_scalar(f'{env_id}/Mean_Score', mean_score, generation)
        self.writer.add_scalar(f'{env_id}/Best_Score', best_score, generation)
        self.writer.add_scalar(f'{env_id}/Progress', mean_score / threshold, generation)

        # CSV logging
        with open(self.csv_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            solved = "True" if mean_score >= threshold else "False"
            f.write(f"{timestamp},{env_id},{generation},{mean_score:.4f},{best_score:.4f},{threshold},{solved},{elapsed_time:.2f}\n")

    def train_single_task(self, task: CurriculumTask) -> bool:
        """Train World Models on a single curriculum task."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting training for {task.env_id}")
        self.logger.info(f"Target score: {task.threshold_score}")
        self.logger.info(f"{'='*60}")

        env_id = task.env_id

        try:
            # Step 1: Collect random data
            self.logger.info("Phase 1: Data Collection")
            data_file = self.collect_random_data(env_id, num_episodes=50)

            # Step 2: Train VAE
            self.logger.info("Phase 2: VAE Training")
            vae_path = self.train_vae(env_id, data_file)

            # Step 3: Encode data to latents
            self.logger.info("Phase 3: Latent Encoding")
            latent_file = self.encode_data_to_latents(env_id, data_file, vae_path)

            # Step 4: Train MDN-RNN
            self.logger.info("Phase 4: MDN-RNN Training")
            mdnrnn_path = self.train_mdnrnn(env_id, latent_file)

            # Step 5: Train Controller
            self.logger.info("Phase 5: Controller Training")
            success = self.train_controller_cmaes(env_id, task, vae_path, mdnrnn_path)

            return success

        except Exception as e:
            self.logger.error(f"Failed to train {env_id}: {e}")
            traceback.print_exc()
            return False

    def run_curriculum(self) -> bool:
        """Run the complete curriculum training."""
        self.logger.info("Starting Curriculum Training")
        self.logger.info(f"Tasks: {[task.env_id for task in self.curriculum]}")

        overall_success = True

        for i, task in enumerate(self.curriculum):
            self.current_task_idx = i

            print(f"\n[TARGET] Task {i+1}/{len(self.curriculum)}: {task.env_id}")
            print(f"Target Score: {task.threshold_score}")
            print("-" * 60)

            success = self.train_single_task(task)

            if success:
                print(f"\n[SUCCESS] {task.env_id} COMPLETED!")
                print(f"   Best Score: {task.best_score:.2f}")
                print(f"   Generations: {task.generations_trained}")
            else:
                print(f"\n[FAILED] {task.env_id} FAILED")
                print(f"   Best Score: {task.best_score:.2f}")
                print(f"   Max Generations Reached: {task.generations_trained}")
                overall_success = False

                # Ask whether to continue
                continue_training = input("Continue to next task? (y/n): ").lower() == 'y'
                if not continue_training:
                    break

        return overall_success

    def generate_final_report(self):
        """Generate final curriculum training report."""
        print("\n" + "="*80)
        print("CURRICULUM TRAINING FINAL REPORT")
        print("="*80)

        total_time = time.time() - self.training_start_time
        solved_count = sum(1 for task in self.curriculum if task.solved)

        print(f"Total Training Time: {total_time/3600:.2f} hours")
        print(f"Tasks Completed: {solved_count}/{len(self.curriculum)}")
        print()

        print("Task Summary:")
        print("-" * 60)
        for i, task in enumerate(self.curriculum):
            status = "[OK] SOLVED" if task.solved else "[X] FAILED"
            print(f"{i+1}. {task.env_id:25} | {status} | "
                  f"Score: {task.best_score:8.2f} / {task.threshold_score:6.1f} | "
                  f"Gens: {task.generations_trained}")

        print("-" * 60)

        # Save final results
        results = {
            'total_time_hours': total_time / 3600,
            'tasks_completed': solved_count,
            'total_tasks': len(self.curriculum),
            'success_rate': solved_count / len(self.curriculum),
            'tasks': [
                {
                    'env_id': task.env_id,
                    'solved': task.solved,
                    'best_score': task.best_score,
                    'threshold_score': task.threshold_score,
                    'generations_trained': task.generations_trained
                }
                for task in self.curriculum
            ]
        }

        results_file = self.checkpoint_dir / "curriculum_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[REPORT] Full results saved to: {results_file}")

        # Close TensorBoard writer
        self.writer.close()

        return solved_count == len(self.curriculum)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Curriculum Trainer with Visual Feedback for World Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full curriculum with visuals and video recording:
  python3 curriculum_trainer_visual.py --device cpu --max-generations 200 --episodes-per-eval 5 --visualize True --record-video True

  # Quick test run with reduced thresholds and fast training:
  python3 curriculum_trainer_visual.py --device cpu --quick True --visualize True

  # Basic training with visualization:
  python3 curriculum_trainer_visual.py --device cpu --visualize True

  # Fast training for testing:
  python3 curriculum_trainer_visual.py --max-generations 50 --episodes-per-eval 3 --visualize False
        """
    )

    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device for training (default: cpu)')
    parser.add_argument('--max-generations', type=int, default=200,
                       help='Maximum generations per environment (default: 200)')
    parser.add_argument('--episodes-per-eval', type=int, default=5,
                       help='Episodes per evaluation (default: 5)')
    parser.add_argument('--checkpoint-dir', default='./runs/curriculum_visual',
                       help='Checkpoint directory (default: ./runs/curriculum_visual)')

    # Quick mode for fast testing
    parser.add_argument('--quick', type=str, default='False',
                       help='Quick mode with reduced thresholds and faster training: True/False (default: False)')

    # Visualization arguments - using string parsing for better compatibility
    parser.add_argument('--visualize', type=str, default='True',
                       help='Enable real-time visualization: True/False (default: True)')
    parser.add_argument('--record-video', type=str, default='False',
                       help='Record training videos: True/False (default: False)')
    parser.add_argument('--video-every-n-gens', type=int, default=10,
                       help='Record video every N generations (default: 10)')
    parser.add_argument('--render-mode', default='human',
                       help='Gymnasium render mode: human/rgb_array (default: human)')

    # Curriculum arguments
    parser.add_argument('--prefer-box2d', choices=['true', 'false'], default=None,
                       help='Force Box2D curriculum (true) or Classic Control (false). Default: auto-detect')

    # GPU Memory Optimization Arguments
    parser.add_argument('--amp', type=str, default='True',
                       help='Enable Automatic Mixed Precision for memory efficiency: True/False (default: True)')
    parser.add_argument('--tf32', type=str, default='True',
                       help='Enable TensorFloat-32 for memory efficiency: True/False (default: True)')
    parser.add_argument('--vae-img-size', type=int, default=64,
                       help='VAE image size for memory efficiency: 32/64/96 (default: 64)')
    parser.add_argument('--vae-batch', type=int, default=32,
                       help='VAE batch size for memory efficiency (default: 32)')
    parser.add_argument('--grad-accum', type=int, default=1,
                       help='Gradient accumulation steps (default: 1)')

    return parser.parse_args()

def main():
    """Main function."""
    print("[MAIN] Starting Curriculum Trainer with Visual Feedback...")

    try:
        print("[MAIN] Parsing arguments...")
        args = parse_args()

        # Parse boolean arguments properly
        visualize = args.visualize.lower() in ('true', '1', 'yes', 'on')
        record_video = args.record_video.lower() in ('true', '1', 'yes', 'on')
        quick_mode = args.quick.lower() in ('true', '1', 'yes', 'on')

        # Parse GPU optimization arguments
        use_amp = args.amp.lower() in ('true', '1', 'yes', 'on')
        use_tf32 = args.tf32.lower() in ('true', '1', 'yes', 'on')

        # Parse curriculum preference
        prefer_box2d = None
        if args.prefer_box2d is not None:
            prefer_box2d = args.prefer_box2d.lower() == 'true'

        # Apply quick mode defaults
        max_generations = args.max_generations
        if quick_mode and args.max_generations == 200:  # Only override if using default
            max_generations = 5
            print("[QUICK MODE] Reducing max generations to 5 for fast testing")

        print(f"[MAIN] Configuration: device={args.device}, max_gens={max_generations}, "
              f"visualize={visualize}, record_video={record_video}, quick_mode={quick_mode}")
        print(f"[MAIN] GPU Optimizations: amp={use_amp}, tf32={use_tf32}, "
              f"vae_batch={args.vae_batch}, img_size={args.vae_img_size}")

        # Create training configuration
        print("[MAIN] Creating training configuration...")
        config = TrainingConfig(
            device=args.device,
            max_generations=max_generations,
            episodes_per_eval=args.episodes_per_eval,
            checkpoint_dir=args.checkpoint_dir,
            visualize=visualize,
            record_video=record_video,
            video_every_n_gens=args.video_every_n_gens,
            render_mode=args.render_mode,
            quick_mode=quick_mode,
            prefer_box2d=prefer_box2d,
            # GPU optimization settings
            use_amp=use_amp,
            use_tf32=use_tf32,
            vae_img_size=args.vae_img_size,
            vae_batch_size=args.vae_batch,
            grad_accumulation_steps=args.grad_accum
        )
        print(f"[MAIN] Config created successfully")

        # Create and run curriculum trainer
        print("[MAIN] Creating curriculum trainer...")
        print(f"[MAIN] About to instantiate CurriculumTrainer with config: {config.device}")
        trainer = CurriculumTrainer(config)
        print("[MAIN] CurriculumTrainer created successfully!")

        print("[LAUNCH] Starting World Models Curriculum Training")
        print(f"Device: {config.device}")
        print(f"Visualization: {'ON' if config.visualize else 'OFF'}")
        print(f"Video Recording: {'ON' if config.record_video else 'OFF'}")

        success = trainer.run_curriculum()
        final_success = trainer.generate_final_report()

        if final_success:
            print("\n[SUCCESS] CURRICULUM COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\n[WARNING] Curriculum completed with some failures")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()

    # Print example commands after successful completion
    print("\n" + "="*80)
    print("CURRICULUM TRAINER UPDATE COMPLETE!")
    print("="*80)
    print("\nUpdated Features:")
    print("✅ New environment list: ALE/Pong-v5, LunarLander-v3, ALE/Breakout-v5, CarRacing-v3")
    print("✅ Quick mode support with --quick flag")
    print("✅ Reduced thresholds in quick mode: Pong=5, LunarLander=50, Breakout=10, CarRacing=200")
    print("✅ Quick mode reduces max_generations to 5 (if using default)")
    print("✅ Clear logging when running in QUICK MODE")
    print("\nExample Commands:")
    print("# Full curriculum:")
    print("python3 curriculum_trainer_visual.py --device cpu --max-generations 200 --episodes-per-eval 5 --visualize True --record-video True")
    print("\n# Quick test run:")
    print("python3 curriculum_trainer_visual.py --device cpu --quick True --visualize True")
    print("="*80)
