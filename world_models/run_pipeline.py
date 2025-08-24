#!/usr/bin/env python3
"""
World Models End-to-End Training Pipeline

This script orchestrates the complete World Models training procedure:
1. Random Data Collection from environment
2. Train VAE on collected frames
3. Encode all frames to latent sequences
4. Train MDN-RNN on latent sequences
5. Train Controller with CMA-ES in dream environment
6. Evaluate and visualize results

Example Usage:
  # Quick smoke test (CPU):
  python run_pipeline.py --env PongNoFrameskip-v5 --mode quick --device cpu

  # Full research run (GPU):
  python run_pipeline.py --env CarRacing-v3 --mode full --device cuda

  # Sequential curriculum:
  python run_pipeline.py --mode full --env-list "PongNoFrameskip-v5,LunarLander-v2"

  # Individual stages:
  python run_pipeline.py --stage collect --env PongNoFrameskip-v5
  python run_pipeline.py --stage vae --env PongNoFrameskip-v5
"""

# Ensure we're in the repository root
from tools.ensure_cwd import chdir_repo_root
chdir_repo_root()

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from pathlib import Path
import pickle
from tqdm import tqdm
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from models.vae import ConvVAE
    from models.mdnrnn import MDNRNN, MDNRNNConfig
    from models.controller import Controller, CMAESController
    from tools.dream_env import DreamEnvironment
    from tools.dataset_utils import FrameDataset, create_frame_dataset
    # from train.controller_trainer import ControllerTrainer  # TODO: Import when available
    # from training.ppo_baseline import PPOBaseline  # TODO: Import when available
    HAS_ALL_MODULES = True
except ImportError as e:
    logger.warning("Some modules not available: {}".format(e))
    HAS_ALL_MODULES = False


class WorldModelsPipeline:
    """Main orchestrator for World Models training pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config
        self.device = torch.device(config['device'])

        # Set seeds for reproducibility
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config['logdir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)

        # Pipeline state
        self.vae = None
        self.mdnrnn = None
        self.controller = None
        self.dream_env = None

        logger.info("Pipeline initialized for environment: {}".format(config['env']))
        logger.info("Checkpoint directory: {}".format(self.checkpoint_dir))
        logger.info("Device: {}".format(self.device))

    def collect_random_data(self) -> str:
        """Stage 1: Collect random rollout data from environment."""
        logger.info("=" * 60)
        logger.info("STAGE 1: COLLECTING RANDOM DATA")
        logger.info("=" * 60)

        env_id = self.config['env']
        num_rollouts = self.config['num_random_rollouts']
        max_frames = self.config['max_frames_per_rollout']

        # Create environment
        try:
            env = gym.make(env_id)
            logger.info(f"Created environment: {env_id}")
        except Exception as e:
            logger.error(f"Failed to create environment {env_id}: {e}")
            raise

        # Determine if this is a pixel-based environment
        obs_space = env.observation_space
        is_pixel_env = len(obs_space.shape) == 3  # Height x Width x Channels

        logger.info(f"Observation space: {obs_space.shape}")
        logger.info(f"Action space: {env.action_space}")
        logger.info(f"Pixel environment: {is_pixel_env}")

        # Data storage
        data_dir = self.checkpoint_dir / "raw_data"
        data_dir.mkdir(exist_ok=True)

        frames_dir = data_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        all_episodes = []
        frame_count = 0

        logger.info(f"Starting {num_rollouts} random rollouts...")

        for rollout in tqdm(range(num_rollouts), desc="Collecting rollouts"):
            episode_data = {
                'frames': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'info': []
            }

            obs, info = env.reset(seed=self.config['seed'] + rollout)
            episode_data['info'].append(info)

            for step in range(max_frames):
                # Process observation
                if is_pixel_env:
                    # Resize to 64x64 and save
                    frame = cv2.resize(obs, (64, 64))
                    frame_path = frames_dir / f"frame_{frame_count:08d}.npy"
                    np.save(frame_path, frame)
                    episode_data['frames'].append(str(frame_path))
                else:
                    # Store state vector directly
                    episode_data['frames'].append(obs.copy())

                # Random action
                action = env.action_space.sample()

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                episode_data['info'].append(info)

                frame_count += 1

                if done:
                    break

            all_episodes.append(episode_data)

            # Log progress
            if rollout % 100 == 0:
                avg_length = np.mean([len(ep['rewards']) for ep in all_episodes])
                logger.info(f"Rollout {rollout}: avg episode length = {avg_length:.1f}")

        env.close()

        # Save episode data
        episodes_path = data_dir / "episodes.pkl"
        with open(episodes_path, 'wb') as f:
            pickle.dump(all_episodes, f)

        # Save metadata
        metadata = {
            'env_id': env_id,
            'num_episodes': len(all_episodes),
            'total_frames': frame_count,
            'is_pixel_env': is_pixel_env,
            'obs_shape': obs_space.shape,
            'action_shape': env.action_space.shape if hasattr(env.action_space, 'shape') else env.action_space.n
        }

        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Data collection complete!")
        logger.info(f"Episodes: {len(all_episodes)}")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Data saved to: {data_dir}")

        return str(data_dir)

    def train_vae(self, data_path: Optional[str] = None) -> str:
        """Stage 2: Train VAE on collected frames."""
        logger.info("=" * 60)
        logger.info("STAGE 2: TRAINING VAE")
        logger.info("=" * 60)

        if data_path is None:
            data_path = str(self.checkpoint_dir / "raw_data")

        # Load metadata
        metadata_path = Path(data_path) / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        is_pixel_env = metadata['is_pixel_env']

        if not is_pixel_env:
            logger.info("Skipping VAE training for non-pixel environment")
            return "skipped"

        # Create dataset
        logger.info("Creating dataset...")
        dataset = create_frame_dataset(
            data_path,
            image_size=64,
            transform=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        logger.info(f"Dataset size: {len(dataset)}")

        # Initialize VAE
        self.vae = ConvVAE(
            latent_size=32,
            input_channels=3,
            device=self.device
        ).to(self.device)

        logger.info(f"VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")

        # Training setup
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        epochs = self.config['vae_epochs']

        logger.info(f"Training VAE for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0

            progress_bar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                # Forward pass
                recon_images, mu, logvar = self.vae(images)

                # Compute losses
                recon_loss = nn.functional.mse_loss(recon_images, images, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                beta = 4.0  # Beta-VAE weight
                total_loss = recon_loss + beta * kl_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()

                # Accumulate metrics
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}",
                    'kl': f"{kl_loss.item():.4f}"
                })

                # Log to TensorBoard
                global_step = epoch * len(dataloader) + batch_idx
                if batch_idx % 100 == 0:
                    self.writer.add_scalar('vae/total_loss', total_loss.item(), global_step)
                    self.writer.add_scalar('vae/recon_loss', recon_loss.item(), global_step)
                    self.writer.add_scalar('vae/kl_loss', kl_loss.item(), global_step)

                    # Save sample reconstruction
                    if batch_idx == 0:
                        with torch.no_grad():
                            sample_recon = torch.cat([images[:8], recon_images[:8]], dim=0)
                            self.writer.add_images('vae/reconstruction', sample_recon, global_step)

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches
            avg_kl = epoch_kl_loss / num_batches

            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")

            # Save checkpoint
            if (epoch + 1) % max(1, epochs // 5) == 0:
                checkpoint_path = self.checkpoint_dir / f"vae_epoch_{epoch+1}.pt"
                self.vae.save_checkpoint(str(checkpoint_path))

        # Save final model
        final_path = self.checkpoint_dir / "vae.pt"
        self.vae.save_checkpoint(str(final_path))

        logger.info(f"VAE training complete! Model saved to: {final_path}")
        return str(final_path)

    def encode_latents(self, data_path: Optional[str] = None, vae_path: Optional[str] = None) -> str:
        """Stage 3: Encode all frames to latent sequences."""
        logger.info("=" * 60)
        logger.info("STAGE 3: ENCODING LATENT SEQUENCES")
        logger.info("=" * 60)

        if data_path is None:
            data_path = str(self.checkpoint_dir / "raw_data")

        # Load metadata
        metadata_path = Path(data_path) / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        is_pixel_env = metadata['is_pixel_env']

        if not is_pixel_env:
            logger.info("Using state observations directly as 'latents' for non-pixel environment")
            # For non-pixel environments, we'll use the state directly
            episodes_path = Path(data_path) / "episodes.pkl"
            with open(episodes_path, 'rb') as f:
                episodes = pickle.load(f)

            # Convert episodes to latent format
            latent_sequences = []
            for episode in episodes:
                sequence = {
                    'latents': np.array(episode['frames'][:-1]),  # All but last frame
                    'next_latents': np.array(episode['frames'][1:]),  # All but first frame
                    'actions': np.array(episode['actions'][:-1]),
                    'rewards': np.array(episode['rewards'][:-1]),
                    'dones': np.array(episode['dones'][:-1])
                }
                latent_sequences.append(sequence)

            # Save latent sequences
            latents_path = self.checkpoint_dir / "latent_sequences.pkl"
            with open(latents_path, 'wb') as f:
                pickle.dump(latent_sequences, f)

            logger.info(f"State sequences saved to: {latents_path}")
            return str(latents_path)

        # Load VAE
        if vae_path is None:
            vae_path = str(self.checkpoint_dir / "vae.pt")

        if self.vae is None:
            self.vae = ConvVAE(latent_size=32, input_channels=3, device=self.device)
            self.vae.load_checkpoint(vae_path)
            self.vae.to(self.device)
            self.vae.eval()

        logger.info(f"Loaded VAE from: {vae_path}")

        # Load episode data
        episodes_path = Path(data_path) / "episodes.pkl"
        with open(episodes_path, 'rb') as f:
            episodes = pickle.load(f)

        logger.info(f"Encoding {len(episodes)} episodes...")

        latent_sequences = []

        with torch.no_grad():
            for episode_idx, episode in enumerate(tqdm(episodes, desc="Encoding episodes")):
                # Load frames for this episode
                frames = []
                for frame_path in episode['frames']:
                    if isinstance(frame_path, str):
                        frame = np.load(frame_path)
                    else:
                        frame = frame_path
                    frames.append(frame)

                if len(frames) < 2:
                    continue  # Skip episodes that are too short

                # Convert to tensor
                frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32).to(self.device)
                frames_tensor = frames_tensor.permute(0, 3, 1, 2) / 255.0  # NHWC -> NCHW, normalize

                # Encode in batches
                batch_size = 32
                latents = []

                for i in range(0, len(frames_tensor), batch_size):
                    batch = frames_tensor[i:i+batch_size]
                    z, _, _ = self.vae(batch)
                    latents.append(z.cpu().numpy())

                latents = np.concatenate(latents, axis=0)

                # Create sequence data (current -> next)
                sequence = {
                    'latents': latents[:-1],  # z_t
                    'next_latents': latents[1:],  # z_{t+1}
                    'actions': np.array(episode['actions'][:-1]),  # a_t
                    'rewards': np.array(episode['rewards'][:-1]),  # r_t
                    'dones': np.array(episode['dones'][:-1])  # done_t
                }

                latent_sequences.append(sequence)

                if episode_idx % 100 == 0:
                    logger.info(f"Encoded {episode_idx+1}/{len(episodes)} episodes")

        # Save latent sequences
        latents_path = self.checkpoint_dir / "latent_sequences.pkl"
        with open(latents_path, 'wb') as f:
            pickle.dump(latent_sequences, f)

        # Summary statistics
        total_transitions = sum(len(seq['latents']) for seq in latent_sequences)
        avg_length = np.mean([len(seq['latents']) for seq in latent_sequences])

        logger.info(f"Latent encoding complete!")
        logger.info(f"Episodes: {len(latent_sequences)}")
        logger.info(f"Total transitions: {total_transitions}")
        logger.info(f"Average episode length: {avg_length:.1f}")
        logger.info(f"Latents saved to: {latents_path}")

        return str(latents_path)

    def train_mdnrnn(self, latents_path: Optional[str] = None) -> str:
        """Stage 4: Train MDN-RNN on latent sequences."""
        logger.info("=" * 60)
        logger.info("STAGE 4: TRAINING MDN-RNN")
        logger.info("=" * 60)

        if latents_path is None:
            latents_path = str(self.checkpoint_dir / "latent_sequences.pkl")

        # Load latent sequences
        with open(latents_path, 'rb') as f:
            sequences = pickle.load(f)

        logger.info(f"Loaded {len(sequences)} latent sequences")

        # Get dimensions from first sequence
        sample_seq = sequences[0]
        latent_size = sample_seq['latents'].shape[1]
        action_size = sample_seq['actions'].shape[1] if len(sample_seq['actions'].shape) > 1 else 1

        logger.info(f"Latent size: {latent_size}")
        logger.info(f"Action size: {action_size}")

        # Initialize MDN-RNN
        config = MDNRNNConfig(
            latent_size=latent_size,
            action_size=action_size,
            hidden_size=self.config['rnn_size'],
            num_mixtures=self.config['mdn_mixtures'],
            device=self.device
        )

        self.mdnrnn = MDNRNN(config).to(self.device)

        logger.info(f"MDN-RNN parameters: {sum(p.numel() for p in self.mdnrnn.parameters()):,}")

        # Prepare training data
        logger.info("Preparing training data...")

        all_latents = []
        all_next_latents = []
        all_actions = []
        all_rewards = []
        all_dones = []

        for seq in sequences:
            if len(seq['latents']) > 0:
                all_latents.extend(seq['latents'])
                all_next_latents.extend(seq['next_latents'])
                all_actions.extend(seq['actions'])
                all_rewards.extend(seq['rewards'])
                all_dones.extend(seq['dones'])

        # Convert to tensors
        latents = torch.tensor(np.array(all_latents), dtype=torch.float32)
        next_latents = torch.tensor(np.array(all_next_latents), dtype=torch.float32)
        actions = torch.tensor(np.array(all_actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(np.array(all_dones), dtype=torch.float32).unsqueeze(1)

        logger.info(f"Training data: {len(latents)} transitions")

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(latents, next_latents, actions, rewards, dones)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

        # Training setup
        optimizer = torch.optim.Adam(self.mdnrnn.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        epochs = self.config['mdnrnn_epochs']

        logger.info(f"Training MDN-RNN for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_latent_loss = 0
            epoch_reward_loss = 0
            epoch_done_loss = 0
            num_batches = 0

            progress_bar = tqdm(dataloader, desc=f"MDN-RNN Epoch {epoch+1}/{epochs}")

            for batch_idx, (z_t, z_next, a_t, r_t, d_t) in enumerate(progress_bar):
                z_t = z_t.to(self.device)
                z_next = z_next.to(self.device)
                a_t = a_t.to(self.device)
                r_t = r_t.to(self.device)
                d_t = d_t.to(self.device)

                # Forward pass
                pi, mu, sigma, reward_pred, done_pred, hidden = self.mdnrnn(z_t, a_t)

                # Compute losses
                latent_loss = self.mdnrnn.mdn_loss(pi, mu, sigma, z_next)
                reward_loss = nn.functional.mse_loss(reward_pred, r_t)
                done_loss = nn.functional.binary_cross_entropy_with_logits(done_pred, d_t)

                total_loss = latent_loss + reward_loss + done_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mdnrnn.parameters(), 1.0)
                optimizer.step()

                # Accumulate metrics
                epoch_loss += total_loss.item()
                epoch_latent_loss += latent_loss.item()
                epoch_reward_loss += reward_loss.item()
                epoch_done_loss += done_loss.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'latent': f"{latent_loss.item():.4f}",
                    'reward': f"{reward_loss.item():.4f}",
                    'done': f"{done_loss.item():.4f}"
                })

                # Log to TensorBoard
                global_step = epoch * len(dataloader) + batch_idx
                if batch_idx % 100 == 0:
                    self.writer.add_scalar('mdnrnn/total_loss', total_loss.item(), global_step)
                    self.writer.add_scalar('mdnrnn/latent_loss', latent_loss.item(), global_step)
                    self.writer.add_scalar('mdnrnn/reward_loss', reward_loss.item(), global_step)
                    self.writer.add_scalar('mdnrnn/done_loss', done_loss.item(), global_step)

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_latent = epoch_latent_loss / num_batches
            avg_reward = epoch_reward_loss / num_batches
            avg_done = epoch_done_loss / num_batches

            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, latent={avg_latent:.4f}, "
                       f"reward={avg_reward:.4f}, done={avg_done:.4f}")

            scheduler.step()

            # Save checkpoint
            if (epoch + 1) % max(1, epochs // 5) == 0:
                checkpoint_path = self.checkpoint_dir / f"mdnrnn_epoch_{epoch+1}.pt"
                self.mdnrnn.save_checkpoint(str(checkpoint_path))

        # Save final model
        final_path = self.checkpoint_dir / "mdnrnn.pt"
        self.mdnrnn.save_checkpoint(str(final_path))

        logger.info(f"MDN-RNN training complete! Model saved to: {final_path}")
        return str(final_path)

    def train_controller(self, vae_path: Optional[str] = None, mdnrnn_path: Optional[str] = None) -> str:
        """Stage 5: Train Controller with CMA-ES."""
        logger.info("=" * 60)
        logger.info("STAGE 5: TRAINING CONTROLLER")
        logger.info("=" * 60)

        # Load models if needed
        if vae_path is None:
            vae_path = str(self.checkpoint_dir / "vae.pt")
        if mdnrnn_path is None:
            mdnrnn_path = str(self.checkpoint_dir / "mdnrnn.pt")

        # Load metadata to determine environment type
        metadata_path = self.checkpoint_dir / "raw_data" / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        is_pixel_env = metadata.get('is_pixel_env', True)

        # Initialize controller
        latent_size = 32 if is_pixel_env else metadata['obs_shape'][0]
        hidden_size = self.config['rnn_size']

        # Determine action space
        env = gym.make(self.config['env'])
        if hasattr(env.action_space, 'n'):
            action_size = env.action_space.n
            action_type = 'discrete'
        else:
            action_size = env.action_space.shape[0]
            action_type = 'continuous'
        env.close()

        self.controller = Controller(
            input_size=latent_size + hidden_size,
            action_size=action_size,
            hidden_sizes=(),  # Linear controller for CMA-ES
            action_type=action_type
        ).to(self.device)

        logger.info(f"Controller: {self.controller.get_num_parameters()} parameters")
        logger.info(f"Action type: {action_type}, size: {action_size}")

        # Create dream environment if training in dream
        if self.config['train_in_dream']:
            logger.info("Creating dream environment...")
            self.dream_env = DreamEnvironment(
                vae_model_path=vae_path,
                mdnrnn_model_path=mdnrnn_path,
                action_space_size=action_size,
                max_episode_steps=1000,
                temperature=1.0,
                device=str(self.device)
            )
            training_env = self.dream_env
            logger.info("Training in dream environment")
        else:
            training_env = gym.make(self.config['env'])
            logger.info("Training in real environment")

        # Initialize CMA-ES trainer
        cmaes_trainer = CMAESController(
            controller=self.controller,
            population_size=self.config['cma_pop_size'],
            sigma=0.5,
            device=self.device
        )

        logger.info(f"CMA-ES population size: {self.config['cma_pop_size']}")
        logger.info(f"Generations: {self.config['cma_generations']}")

        # Mock fitness function (replace with actual environment evaluation)
        def evaluate_candidate(params):
            """Evaluate a candidate controller."""
            # Set parameters
            self.controller.set_parameters(params)
            self.controller.eval()

            total_reward = 0
            num_rollouts = self.config['rollouts_per_candidate']

            with torch.no_grad():
                for _ in range(num_rollouts):
                    obs, _ = training_env.reset()
                    episode_reward = 0

                    # Mock latent and hidden state
                    z = torch.randn(1, latent_size, device=self.device)
                    h = torch.randn(1, hidden_size, device=self.device)

                    for step in range(200):  # Max episode length
                        # Get action from controller
                        if action_type == 'continuous':
                            action = self.controller.get_action(z, h, deterministic=True)
                            if isinstance(action, tuple):
                                action = action[0]
                            action = action.cpu().numpy().squeeze()
                        else:
                            action = self.controller.get_action(z, h, deterministic=True)
                            action = action.cpu().numpy().item()

                        # Take step
                        obs, reward, terminated, truncated, _ = training_env.step(action)
                        episode_reward += reward

                        if terminated or truncated:
                            break

                        # Update states (simplified - would use VAE/MDN-RNN in practice)
                        z = torch.randn(1, latent_size, device=self.device)
                        h = torch.randn(1, hidden_size, device=self.device)

                    total_reward += episode_reward

            return total_reward / num_rollouts

        # Training loop
        logger.info("Starting CMA-ES training...")
        best_fitness = -float('inf')
        fitness_history = []

        for generation in tqdm(range(self.config['cma_generations']), desc="CMA-ES Training"):
            # Get candidate solutions
            candidates = cmaes_trainer.ask()

            # Evaluate candidates
            fitness_values = []
            for candidate in candidates:
                fitness = evaluate_candidate(candidate)
                fitness_values.append(fitness)

            fitness_values = np.array(fitness_values)

            # Update CMA-ES
            cmaes_trainer.tell(candidates, fitness_values)

            # Track best fitness
            gen_best = np.max(fitness_values)
            if gen_best > best_fitness:
                best_fitness = gen_best

            fitness_history.append(fitness_values)

            # Log progress
            if generation % 10 == 0:
                stats = cmaes_trainer.get_stats()
                logger.info(f"Generation {generation}: best={gen_best:.3f}, "
                           f"mean={stats['mean_fitness']:.3f}, sigma={stats['sigma']:.3f}")

                # Log to TensorBoard
                self.writer.add_scalar('controller/best_fitness', gen_best, generation)
                self.writer.add_scalar('controller/mean_fitness', stats['mean_fitness'], generation)
                self.writer.add_scalar('controller/sigma', stats['sigma'], generation)

        # Update controller with best parameters
        cmaes_trainer.update_controller()

        # Save controller
        controller_path = self.checkpoint_dir / "controller_best.pt"
        torch.save({
            'state_dict': self.controller.state_dict(),
            'config': {
                'input_size': latent_size + hidden_size,
                'action_size': action_size,
                'hidden_sizes': (),
                'action_type': action_type
            },
            'best_fitness': best_fitness,
            'generation': generation
        }, controller_path)

        # Save training history
        history_path = self.checkpoint_dir / "controller_training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(fitness_history, f)

        if not self.config['train_in_dream']:
            training_env.close()

        logger.info(f"Controller training complete!")
        logger.info(f"Best fitness: {best_fitness:.3f}")
        logger.info(f"Controller saved to: {controller_path}")

        return str(controller_path)

    def evaluate_pipeline(self, controller_path: Optional[str] = None) -> Dict[str, Any]:
        """Stage 6: Evaluate trained controller."""
        logger.info("=" * 60)
        logger.info("STAGE 6: EVALUATION")
        logger.info("=" * 60)

        if controller_path is None:
            controller_path = str(self.checkpoint_dir / "controller_best.pt")

        # Load controller
        checkpoint = torch.load(controller_path, map_location=self.device)
        controller_config = checkpoint['config']

        controller = Controller(**controller_config).to(self.device)
        controller.load_state_dict(checkpoint['state_dict'])
        controller.eval()

        logger.info(f"Loaded controller from: {controller_path}")
        logger.info(f"Training fitness: {checkpoint['best_fitness']:.3f}")

        # Create evaluation environment
        env = gym.make(self.config['env'])

        # Evaluation
        num_eval_episodes = 10
        episode_rewards = []
        episode_lengths = []

        logger.info(f"Evaluating for {num_eval_episodes} episodes...")

        for episode in tqdm(range(num_eval_episodes), desc="Evaluation"):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            # Mock states for evaluation
            latent_size = controller_config['input_size'] - self.config['rnn_size']
            z = torch.randn(1, latent_size, device=self.device)
            h = torch.randn(1, self.config['rnn_size'], device=self.device)

            for step in range(1000):
                with torch.no_grad():
                    if controller_config['action_type'] == 'continuous':
                        action = controller.get_action(z, h, deterministic=True)
                        if isinstance(action, tuple):
                            action = action[0]
                        action = action.cpu().numpy().squeeze()
                    else:
                        action = controller.get_action(z, h, deterministic=True)
                        action = action.cpu().numpy().item()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

                # Update states (simplified)
                z = torch.randn(1, latent_size, device=self.device)
                h = torch.randn(1, self.config['rnn_size'], device=self.device)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        env.close()

        # Compute statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"  Min/Max reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
        logger.info(f"  Mean episode length: {results['mean_length']:.1f}")

        # Save results
        results_path = self.checkpoint_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if not isinstance(v, list)}, f, indent=2)

        return results

    def run_stage(self, stage: str):
        """Run a specific pipeline stage."""
        logger.info(f"Running stage: {stage}")

        if stage == 'collect':
            return self.collect_random_data()
        elif stage == 'vae':
            return self.train_vae()
        elif stage == 'encode':
            return self.encode_latents()
        elif stage == 'mdnrnn':
            return self.train_mdnrnn()
        elif stage == 'controller':
            return self.train_controller()
        elif stage == 'eval':
            return self.evaluate_pipeline()
        elif stage == 'all':
            self.collect_random_data()
            self.train_vae()
            self.encode_latents()
            self.train_mdnrnn()
            self.train_controller()
            return self.evaluate_pipeline()
        else:
            raise ValueError(f"Unknown stage: {stage}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="World Models End-to-End Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Quick smoke test (CPU):
  python run_pipeline.py --env PongNoFrameskip-v5 --mode quick --device cpu

  # Full research run (GPU):
  python run_pipeline.py --env CarRacing-v2 --mode full --device cuda

  # Sequential curriculum:
  python run_pipeline.py --mode full --env-list "PongNoFrameskip-v5,LunarLander-v2"

  # Individual stages:
  python run_pipeline.py --stage collect --env PongNoFrameskip-v5
  python run_pipeline.py --stage vae --env PongNoFrameskip-v5
        """
    )

    # Environment settings
    parser.add_argument('--env-list', type=str,
                       default="PongNoFrameskip-v5,LunarLander-v2,BreakoutNoFrameskip-v5,CarRacing-v2",
                       help='Comma-separated environment IDs')
    parser.add_argument('--env', type=str, help='Single environment override')

    # Training mode
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Training mode: quick (smoke test) or full (research)')

    # Hardware
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Data collection
    parser.add_argument('--num-random-rollouts', type=int, help='Number of random rollouts')
    parser.add_argument('--max-frames-per-rollout', type=int, default=1000,
                       help='Maximum frames per rollout')

    # VAE training
    parser.add_argument('--vae-epochs', type=int, help='VAE training epochs')

    # MDN-RNN training
    parser.add_argument('--mdnrnn-epochs', type=int, help='MDN-RNN training epochs')
    parser.add_argument('--mdn-mixtures', type=int, default=5,
                       help='Number of MDN mixtures')
    parser.add_argument('--rnn-size', type=int, default=256,
                       help='RNN hidden size')

    # Controller training
    parser.add_argument('--cma-pop-size', type=int, help='CMA-ES population size')
    parser.add_argument('--cma-generations', type=int, help='CMA-ES generations')
    parser.add_argument('--rollouts-per-candidate', type=int,
                       help='Rollouts per candidate evaluation')
    parser.add_argument('--train-in-dream', action='store_true', default=True,
                       help='Train controller in dream environment')

    # Paths
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Checkpoint directory')
    parser.add_argument('--logdir', type=str,
                       help='Log directory for TensorBoard')
    parser.add_argument('--save-videos', action='store_true',
                       help='Save evaluation videos')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint')

    # Stage selection
    parser.add_argument('--stage', choices=['collect', 'vae', 'encode', 'mdnrnn', 'controller', 'eval', 'all'],
                       default='all', help='Pipeline stage to run')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'

    # Set mode-specific defaults
    if args.mode == 'quick':
        defaults = {
            'num_random_rollouts': 50,
            'vae_epochs': 1,
            'mdnrnn_epochs': 2,
            'cma_pop_size': 8,
            'cma_generations': 5,
            'rollouts_per_candidate': 2,
        }
    else:  # full mode
        defaults = {
            'num_random_rollouts': 10000,
            'vae_epochs': 10,
            'mdnrnn_epochs': 20,
            'cma_pop_size': 64,
            'cma_generations': 1000,
            'rollouts_per_candidate': 16,
        }

    # Apply defaults if not specified
    for key, default_value in defaults.items():
        if getattr(args, key.replace('_', '-').replace('-', '_')) is None:
            setattr(args, key, default_value)

    # Get environment list
    if args.env:
        env_list = [args.env]
    else:
        env_list = [env.strip() for env in args.env_list.split(',')]

    logger.info("World Models Pipeline Starting")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Environments: {env_list}")
    logger.info(f"Stage: {args.stage}")

    # Run pipeline for each environment
    for env_id in env_list:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING ENVIRONMENT: {env_id}")
        logger.info(f"{'='*80}")

        # Set up configuration
        config = {
            'env': env_id,
            'device': args.device,
            'seed': args.seed,
            'num_random_rollouts': args.num_random_rollouts,
            'max_frames_per_rollout': args.max_frames_per_rollout,
            'vae_epochs': args.vae_epochs,
            'mdnrnn_epochs': args.mdnrnn_epochs,
            'mdn_mixtures': args.mdn_mixtures,
            'rnn_size': args.rnn_size,
            'cma_pop_size': args.cma_pop_size,
            'cma_generations': args.cma_generations,
            'rollouts_per_candidate': args.rollouts_per_candidate,
            'train_in_dream': args.train_in_dream,
            'checkpoint_dir': args.checkpoint_dir or f"./runs/{env_id.replace(':', '_')}_worldmodel",
            'logdir': args.logdir or f"./runs/logs/{env_id.replace(':', '_')}",
            'save_videos': args.save_videos,
        }

        try:
            # Initialize pipeline
            pipeline = WorldModelsPipeline(config)

            # Run specified stage
            start_time = time.time()
            result = pipeline.run_stage(args.stage)
            end_time = time.time()

            logger.info(f"Pipeline completed for {env_id} in {end_time - start_time:.2f} seconds")

            if args.stage == 'eval' and isinstance(result, dict):
                logger.info(f"Final evaluation: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")

        except Exception as e:
            logger.error(f"Pipeline failed for {env_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("All environments processed!")

    # Print example commands
    print("\n" + "="*80)
    print("EXAMPLE COMMANDS:")
    print("="*80)
    print("# Quick smoke test:")
    print("python run_pipeline.py --env PongNoFrameskip-v5 --mode quick --device cpu")
    print()
    print("# Full training run:")
    print("python run_pipeline.py --env CarRacing-v2 --mode full --device cuda")
    print()
    print("# Individual stages:")
    print("python run_pipeline.py --stage collect --env PongNoFrameskip-v5")
    print("python run_pipeline.py --stage vae --env PongNoFrameskip-v5")
    print("="*80)


if __name__ == "__main__":
    main()
