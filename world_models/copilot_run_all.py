#!/usr/bin/env python3
"""
World Models Complete Pipeline Orchestration Script

This script executes the full World Models training pipeline synchronously:
1. Collect random rollouts
2. Train VAE on frames
3. Encode frames to latent sequences  
4. Train MDN-RNN on latent dynamics
5. Train Controller with CMA-ES
6. Evaluate and save results

Author: GitHub Copilot
Created: August 2025
"""

import os
import sys
import time
import logging
import argparse
import json
import csv
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Try importing PyTorch and check CUDA
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch available. Default device: {DEVICE_DEFAULT}")
except ImportError:
    logger.error("PyTorch not found. Install with: pip install torch torchvision")
    sys.exit(1)

# Try importing project modules with helpful error messages
try:
    from models.vae import ConvVAE
    logger.info("✓ VAE module imported successfully")
except ImportError as e:
    logger.error(f"❌ VAE import failed: {e}")
    logger.error("   Expected: models/vae.py with ConvVAE class")
    VAE_AVAILABLE = False
else:
    VAE_AVAILABLE = True

try:
    from models.mdnrnn import MDNRNN, MDNRNNConfig
    logger.info("✓ MDN-RNN module imported successfully")
except ImportError as e:
    logger.error(f"❌ MDN-RNN import failed: {e}")
    logger.error("   Expected: models/mdnrnn.py with MDNRNN and MDNRNNConfig classes")
    MDNRNN_AVAILABLE = False
else:
    MDNRNN_AVAILABLE = True

try:
    from models.controller import Controller, CMAESController
    logger.info("✓ Controller module imported successfully")
except ImportError as e:
    logger.error(f"❌ Controller import failed: {e}")
    logger.error("   Expected: models/controller.py with Controller and CMAESController classes")
    CONTROLLER_AVAILABLE = False
else:
    CONTROLLER_AVAILABLE = True

try:
    from tools.dream_env import DreamEnvironment
    logger.info("✓ Dream environment imported successfully")
except ImportError as e:
    logger.error(f"❌ Dream environment import failed: {e}")
    logger.error("   Expected: tools/dream_env.py with DreamEnvironment class")
    DREAM_ENV_AVAILABLE = False
else:
    DREAM_ENV_AVAILABLE = True

try:
    from tools.dataset_utils import create_frame_dataset, FrameDataset
    logger.info("✓ Dataset utilities imported successfully")
except ImportError as e:
    logger.error(f"❌ Dataset utilities import failed: {e}")
    logger.error("   Expected: tools/dataset_utils.py with create_frame_dataset function")
    DATASET_UTILS_AVAILABLE = False
else:
    DATASET_UTILS_AVAILABLE = True

# Optional imports
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
    logger.info("✓ Gymnasium available")
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
        logger.info("✓ OpenAI Gym available (legacy)")
    except ImportError:
        logger.error("❌ Gymnasium/Gym not found. Install with: pip install gymnasium[atari]")
        GYM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    logger.info("✓ TensorBoard available")
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("⚠️ TensorBoard not available. Install with: pip install tensorboard")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("⚠️ OpenCV not available. Install with: pip install opencv-python")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a simple progress bar fallback
    def tqdm(iterable, desc="Processing", total=None):
        """Simple tqdm fallback"""
        if hasattr(iterable, '__len__'):
            total = len(iterable)
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 10) == 0:
                print(f"{desc}: {i+1}/{total}")
        return iterable


class WorldModelsPipeline:
    """Complete World Models training pipeline orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config
        self.device = torch.device(config['device'])
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = self.checkpoint_dir / 'logs'
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up TensorBoard logging if available
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.log_dir)
            logger.info(f"TensorBoard logs: {self.log_dir}")
        
        # Pipeline state tracking
        self.stage_results = {}
        self.stage_timings = {}
        
        # Set seeds for reproducibility
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        logger.info(f"🚀 Pipeline initialized")
        logger.info(f"   Environment: {config['env']}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"   Mode: {config['mode']}")
    
    def log_stage_completion(self, stage: str, duration: float, success: bool, metric: Optional[float] = None):
        """Log stage completion to CSV summary."""
        csv_path = self.checkpoint_dir / 'pipeline_summary.csv'
        
        # Create CSV if it doesn't exist
        if not csv_path.exists():
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['stage', 'duration_sec', 'success', 'metric', 'timestamp'])
        
        # Append result
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([stage, f"{duration:.2f}", success, metric or "", time.strftime('%Y-%m-%d %H:%M:%S')])
        
        self.stage_timings[stage] = duration
        logger.info(f"✓ {stage} completed in {duration:.2f}s")
    
    def verify_stage_output(self, stage: str, expected_path: str) -> bool:
        """Verify that stage produced expected output."""
        path = Path(expected_path)
        if path.exists():
            logger.info(f"✓ {stage} output verified: {path}")
            return True
        else:
            logger.error(f"❌ {stage} output missing: {path}")
            return False
    
    def stage_1_collect_data(self) -> str:
        """Stage 1: Collect random rollouts from environment."""
        logger.info("=" * 60)
        logger.info("🎮 STAGE 1: COLLECTING RANDOM DATA")
        logger.info("=" * 60)
        
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium/Gym required for data collection. Install with: pip install gymnasium[atari]")
        
        start_time = time.time()
        
        try:
            env_id = self.config['env']
            num_rollouts = self.config['num_random_rollouts']
            max_frames = 1000  # Max frames per episode
            
            logger.info(f"Environment: {env_id}")
            logger.info(f"Rollouts: {num_rollouts}")
            
            # Create environment
            env = gym.make(env_id)
            obs_space = env.observation_space
            action_space = env.action_space
            
            # Determine environment type
            is_pixel_env = len(obs_space.shape) == 3
            logger.info(f"Environment type: {'pixel' if is_pixel_env else 'state'}")
            logger.info(f"Observation space: {obs_space.shape}")
            logger.info(f"Action space: {action_space}")
            
            # Data collection
            data_dir = self.checkpoint_dir / "raw_data"
            data_dir.mkdir(exist_ok=True)
            
            episodes_data = []
            total_frames = 0
            
            for rollout in tqdm(range(num_rollouts), desc="Collecting rollouts"):
                episode = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'dones': [],
                }
                
                obs, info = env.reset(seed=self.config.get('seed', 42) + rollout)
                episode['observations'].append(obs.copy())
                
                for step in range(max_frames):
                    # Random action
                    if hasattr(action_space, 'sample'):
                        action = action_space.sample()
                    else:
                        action = np.random.randint(action_space.n)
                    
                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Store transition
                    episode['actions'].append(action)
                    episode['rewards'].append(reward)
                    episode['dones'].append(done)
                    
                    if not done:
                        episode['observations'].append(obs.copy())
                    
                    total_frames += 1
                    
                    if done:
                        break
                
                episodes_data.append(episode)
            
            env.close()
            
            # Save data
            import pickle
            with open(data_dir / "episodes.pkl", 'wb') as f:
                pickle.dump(episodes_data, f)
            
            # Save metadata
            metadata = {
                'env_id': env_id,
                'num_episodes': len(episodes_data),
                'total_frames': total_frames,
                'is_pixel_env': is_pixel_env,
                'obs_shape': list(obs_space.shape),
                'action_space_type': 'discrete' if hasattr(action_space, 'n') else 'continuous',
                'action_space_size': getattr(action_space, 'n', action_space.shape[0])
            }
            
            with open(data_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            duration = time.time() - start_time
            self.log_stage_completion('collect', duration, True, total_frames)
            
            data_path = str(data_dir)
            if self.verify_stage_output('collect', str(data_dir / "episodes.pkl")):
                return data_path
            else:
                raise RuntimeError("Data collection verification failed")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_stage_completion('collect', duration, False)
            logger.error(f"Data collection failed: {e}")
            raise
    
    def stage_2_train_vae(self, data_path: str) -> str:
        """Stage 2: Train VAE on collected frames."""
        logger.info("=" * 60)
        logger.info("🧠 STAGE 2: TRAINING VAE")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load metadata
            with open(Path(data_path) / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            if not metadata['is_pixel_env']:
                logger.info("Skipping VAE training for state-based environment")
                duration = time.time() - start_time
                self.log_stage_completion('vae', duration, True, 0)
                return "skipped"
            
            if not VAE_AVAILABLE:
                raise ImportError("VAE module not available. Check models/vae.py")
            
            # Load episode data
            import pickle
            with open(Path(data_path) / "episodes.pkl", 'rb') as f:
                episodes = pickle.load(f)
            
            # Prepare frames
            all_frames = []
            for episode in episodes:
                all_frames.extend(episode['observations'])
            
            logger.info(f"Training VAE on {len(all_frames)} frames")
            
            # Convert frames to tensor
            frames_np = np.array(all_frames)
            if len(frames_np.shape) == 4:  # NHWC -> NCHW
                frames_np = frames_np.transpose(0, 3, 1, 2)
            
            # Resize to 64x64 if needed
            if CV2_AVAILABLE and frames_np.shape[-1] != 64:
                resized_frames = []
                for frame in frames_np:
                    if len(frame.shape) == 3:
                        frame = frame.transpose(1, 2, 0)  # CHW -> HWC for cv2
                    resized = cv2.resize(frame, (64, 64))
                    if len(resized.shape) == 2:
                        resized = resized[..., None]
                    resized_frames.append(resized.transpose(2, 0, 1))  # HWC -> CHW
                frames_np = np.array(resized_frames)
            
            # Create dataset and dataloader
            frames_tensor = torch.tensor(frames_np, dtype=torch.float32) / 255.0
            dataset = torch.utils.data.TensorDataset(frames_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
            
            # Initialize VAE
            input_channels = frames_tensor.shape[1]
            vae = ConvVAE(latent_size=32, input_channels=input_channels, device=self.device)
            vae = vae.to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
            epochs = self.config['vae_epochs']
            
            logger.info(f"Training VAE for {epochs} epochs")
            
            # Training loop
            total_loss = 0
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_idx, (batch,) in enumerate(dataloader):
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    recon_batch, mu, logvar = vae(batch)
                    
                    # Compute losses
                    recon_loss = nn.functional.mse_loss(recon_batch, batch, reduction='mean')
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    loss = recon_loss + 4.0 * kl_loss  # Beta-VAE
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if self.writer and batch_idx % 10 == 0:
                        step = epoch * len(dataloader) + batch_idx
                        self.writer.add_scalar('vae/loss', loss.item(), step)
                        self.writer.add_scalar('vae/recon_loss', recon_loss.item(), step)
                        self.writer.add_scalar('vae/kl_loss', kl_loss.item(), step)
                
                avg_loss = epoch_loss / len(dataloader)
                total_loss += avg_loss
                logger.info(f"VAE Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
            
            # Save VAE
            vae_path = self.checkpoint_dir / "vae.pt"
            vae.save_checkpoint(str(vae_path))
            
            duration = time.time() - start_time
            final_loss = total_loss / epochs
            self.log_stage_completion('vae', duration, True, final_loss)
            
            if self.verify_stage_output('vae', str(vae_path)):
                return str(vae_path)
            else:
                raise RuntimeError("VAE training verification failed")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_stage_completion('vae', duration, False)
            logger.error(f"VAE training failed: {e}")
            raise
    
    def stage_3_encode_latents(self, data_path: str, vae_path: str) -> str:
        """Stage 3: Encode observations to latent sequences."""
        logger.info("=" * 60)
        logger.info("🔄 STAGE 3: ENCODING LATENT SEQUENCES")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load metadata
            with open(Path(data_path) / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load episode data
            import pickle
            with open(Path(data_path) / "episodes.pkl", 'rb') as f:
                episodes = pickle.load(f)
            
            logger.info(f"Encoding {len(episodes)} episodes")
            
            if metadata['is_pixel_env'] and vae_path != "skipped":
                # Load VAE for pixel environments
                if not VAE_AVAILABLE:
                    raise ImportError("VAE module not available for encoding")
                
                vae = ConvVAE(latent_size=32, input_channels=3, device=self.device)
                vae.load_checkpoint(vae_path)
                vae = vae.to(self.device)
                vae.eval()
                
                logger.info("Using VAE for pixel encoding")
            else:
                vae = None
                logger.info("Using state observations directly")
            
            # Encode sequences
            latent_sequences = []
            total_transitions = 0
            
            with torch.no_grad():
                for ep_idx, episode in enumerate(tqdm(episodes, desc="Encoding episodes")):
                    obs = episode['observations']
                    actions = episode['actions']
                    rewards = episode['rewards']
                    dones = episode['dones']
                    
                    if len(obs) < 2:  # Need at least 2 observations for transitions
                        continue
                    
                    if vae is not None:
                        # Encode pixels to latents
                        obs_np = np.array(obs)
                        if len(obs_np.shape) == 4:  # NHWC -> NCHW
                            obs_np = obs_np.transpose(0, 3, 1, 2)
                        
                        # Resize if needed
                        if CV2_AVAILABLE and obs_np.shape[-1] != 64:
                            resized_obs = []
                            for frame in obs_np:
                                if len(frame.shape) == 3:
                                    frame = frame.transpose(1, 2, 0)  # CHW -> HWC
                                resized = cv2.resize(frame, (64, 64))
                                if len(resized.shape) == 2:
                                    resized = resized[..., None]
                                resized_obs.append(resized.transpose(2, 0, 1))  # HWC -> CHW
                            obs_np = np.array(resized_obs)
                        
                        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(self.device) / 255.0
                        
                        # Encode in batches
                        latents = []
                        batch_size = 32
                        for i in range(0, len(obs_tensor), batch_size):
                            batch = obs_tensor[i:i+batch_size]
                            z, _, _ = vae(batch)
                            latents.append(z.cpu().numpy())
                        
                        latents = np.concatenate(latents, axis=0)
                    else:
                        # Use state observations directly
                        latents = np.array(obs)
                    
                    # Create transition sequences
                    if len(latents) >= 2:
                        sequence = {
                            'latents': latents[:-1],        # z_t
                            'next_latents': latents[1:],    # z_{t+1}
                            'actions': np.array(actions),   # a_t
                            'rewards': np.array(rewards),   # r_t
                            'dones': np.array(dones)        # done_t
                        }
                        latent_sequences.append(sequence)
                        total_transitions += len(sequence['latents'])
            
            # Save latent sequences
            latents_path = self.checkpoint_dir / "latent_sequences.pkl"
            with open(latents_path, 'wb') as f:
                pickle.dump(latent_sequences, f)
            
            # Save sequence metadata
            seq_metadata = {
                'num_sequences': len(latent_sequences),
                'total_transitions': total_transitions,
                'latent_dim': latent_sequences[0]['latents'].shape[1] if latent_sequences else 0
            }
            
            with open(self.checkpoint_dir / "latent_metadata.json", 'w') as f:
                json.dump(seq_metadata, f, indent=2)
            
            duration = time.time() - start_time
            self.log_stage_completion('encode', duration, True, total_transitions)
            
            logger.info(f"Encoded {len(latent_sequences)} sequences, {total_transitions} transitions")
            
            if self.verify_stage_output('encode', str(latents_path)):
                return str(latents_path)
            else:
                raise RuntimeError("Latent encoding verification failed")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_stage_completion('encode', duration, False)
            logger.error(f"Latent encoding failed: {e}")
            raise
    
    def stage_4_train_mdnrnn(self, latents_path: str) -> str:
        """Stage 4: Train MDN-RNN on latent sequences."""
        logger.info("=" * 60)
        logger.info("🔮 STAGE 4: TRAINING MDN-RNN")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            if not MDNRNN_AVAILABLE:
                raise ImportError("MDN-RNN module not available. Check models/mdnrnn.py")
            
            # Load latent sequences
            import pickle
            with open(latents_path, 'rb') as f:
                sequences = pickle.load(f)
            
            if not sequences:
                raise ValueError("No latent sequences found")
            
            # Get dimensions
            sample_seq = sequences[0]
            latent_size = sample_seq['latents'].shape[1]
            
            # Determine action size
            if len(sample_seq['actions'].shape) > 1:
                action_size = sample_seq['actions'].shape[1]
            else:
                # For discrete actions, use one-hot encoding size
                max_action = max(max(seq['actions']) for seq in sequences)
                action_size = int(max_action) + 1
            
            logger.info(f"Latent size: {latent_size}")
            logger.info(f"Action size: {action_size}")
            
            # Initialize MDN-RNN
            config = MDNRNNConfig(
                latent_size=latent_size,
                action_size=action_size,
                hidden_size=256,
                num_mixtures=5,
                device=self.device
            )
            
            mdnrnn = MDNRNN(config).to(self.device)
            
            # Prepare training data
            all_latents = []
            all_next_latents = []
            all_actions = []
            all_rewards = []
            all_dones = []
            
            for seq in sequences:
                latents = seq['latents']
                next_latents = seq['next_latents']
                actions = seq['actions']
                rewards = seq['rewards']
                dones = seq['dones']
                
                # Ensure same length
                min_len = min(len(latents), len(next_latents), len(actions), len(rewards), len(dones))
                if min_len > 0:
                    all_latents.extend(latents[:min_len])
                    all_next_latents.extend(next_latents[:min_len])
                    all_actions.extend(actions[:min_len])
                    all_rewards.extend(rewards[:min_len])
                    all_dones.extend(dones[:min_len])
            
            logger.info(f"Training on {len(all_latents)} transitions")
            
            # Convert to tensors
            latents_tensor = torch.tensor(np.array(all_latents), dtype=torch.float32)
            next_latents_tensor = torch.tensor(np.array(all_next_latents), dtype=torch.float32)
            
            # Handle actions (discrete vs continuous)
            actions_np = np.array(all_actions)
            if len(actions_np.shape) == 1:  # Discrete actions
                # One-hot encode
                actions_onehot = np.zeros((len(actions_np), action_size))
                actions_onehot[np.arange(len(actions_np)), actions_np.astype(int)] = 1
                actions_tensor = torch.tensor(actions_onehot, dtype=torch.float32)
            else:  # Continuous actions
                actions_tensor = torch.tensor(actions_np, dtype=torch.float32)
            
            rewards_tensor = torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(1)
            dones_tensor = torch.tensor(np.array(all_dones), dtype=torch.float32).unsqueeze(1)
            
            # Create dataset and dataloader
            dataset = torch.utils.data.TensorDataset(
                latents_tensor, next_latents_tensor, actions_tensor, rewards_tensor, dones_tensor
            )
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
            
            # Training setup
            optimizer = torch.optim.Adam(mdnrnn.parameters(), lr=1e-3)
            epochs = self.config['mdnrnn_epochs']
            
            logger.info(f"Training MDN-RNN for {epochs} epochs")
            
            # Training loop
            total_loss = 0
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_idx, (z_t, z_next, a_t, r_t, d_t) in enumerate(dataloader):
                    z_t = z_t.to(self.device)
                    z_next = z_next.to(self.device)
                    a_t = a_t.to(self.device)
                    r_t = r_t.to(self.device)
                    d_t = d_t.to(self.device)
                    
                    # Forward pass
                    pi, mu, sigma, reward_pred, done_pred, hidden = mdnrnn(z_t, a_t)
                    
                    # Compute losses
                    latent_loss = mdnrnn.mdn_loss(pi, mu, sigma, z_next)
                    reward_loss = nn.functional.mse_loss(reward_pred, r_t)
                    done_loss = nn.functional.binary_cross_entropy_with_logits(done_pred, d_t)
                    
                    loss = latent_loss + reward_loss + done_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if self.writer and batch_idx % 10 == 0:
                        step = epoch * len(dataloader) + batch_idx
                        self.writer.add_scalar('mdnrnn/total_loss', loss.item(), step)
                        self.writer.add_scalar('mdnrnn/latent_loss', latent_loss.item(), step)
                        self.writer.add_scalar('mdnrnn/reward_loss', reward_loss.item(), step)
                        self.writer.add_scalar('mdnrnn/done_loss', done_loss.item(), step)
                
                avg_loss = epoch_loss / len(dataloader)
                total_loss += avg_loss
                logger.info(f"MDN-RNN Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
            
            # Save MDN-RNN
            mdnrnn_path = self.checkpoint_dir / "mdnrnn.pt"
            mdnrnn.save_checkpoint(str(mdnrnn_path))
            
            duration = time.time() - start_time
            final_loss = total_loss / epochs
            self.log_stage_completion('mdnrnn', duration, True, final_loss)
            
            if self.verify_stage_output('mdnrnn', str(mdnrnn_path)):
                return str(mdnrnn_path)
            else:
                raise RuntimeError("MDN-RNN training verification failed")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_stage_completion('mdnrnn', duration, False)
            logger.error(f"MDN-RNN training failed: {e}")
            raise
    
    def stage_5_train_controller(self, vae_path: str, mdnrnn_path: str) -> str:
        """Stage 5: Train Controller with CMA-ES."""
        logger.info("=" * 60)
        logger.info("🎯 STAGE 5: TRAINING CONTROLLER")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            if not CONTROLLER_AVAILABLE:
                raise ImportError("Controller module not available. Check models/controller.py")
            
            # Load metadata to determine environment configuration
            with open(self.checkpoint_dir / "raw_data" / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Determine input/output sizes
            if metadata['is_pixel_env'] and vae_path != "skipped":
                latent_size = 32  # VAE latent size
            else:
                latent_size = metadata['obs_shape'][0]  # State size
            
            hidden_size = 256  # MDN-RNN hidden size
            input_size = latent_size + hidden_size
            
            action_size = metadata['action_space_size']
            action_type = metadata['action_space_type']
            
            logger.info(f"Controller input size: {input_size}")
            logger.info(f"Controller action size: {action_size}")
            logger.info(f"Action type: {action_type}")
            
            # Initialize controller (linear for CMA-ES efficiency)
            controller = Controller(
                input_size=input_size,
                action_size=action_size,
                hidden_sizes=(),  # Linear controller
                action_type=action_type
            ).to(self.device)
            
            logger.info(f"Controller parameters: {controller.get_num_parameters()}")
            
            # Set up training environment
            if self.config['train_in_dream'] and DREAM_ENV_AVAILABLE and vae_path != "skipped":
                logger.info("Setting up dream environment training")
                dream_env = DreamEnvironment(
                    vae_model_path=vae_path,
                    mdnrnn_model_path=mdnrnn_path,
                    action_space_size=action_size,
                    max_episode_steps=200,
                    temperature=1.0,
                    device=str(self.device)
                )
                training_env = dream_env
                logger.info("Training in dream environment")
            else:
                # Fall back to real environment
                if not GYM_AVAILABLE:
                    raise ImportError("Gymnasium required for real environment training")
                training_env = gym.make(self.config['env'])
                logger.info("Training in real environment")
            
            # Initialize CMA-ES trainer
            cmaes_trainer = CMAESController(
                controller=controller,
                population_size=self.config['cma_pop_size'],
                sigma=0.5,
                device=self.device
            )
            
            logger.info(f"CMA-ES population: {self.config['cma_pop_size']}")
            logger.info(f"CMA-ES generations: {self.config['cma_generations']}")
            
            # Training function for CMA-ES
            def evaluate_controller(params):
                controller.set_parameters(params)
                controller.eval()
                
                total_reward = 0
                num_rollouts = self.config['rollouts_per_candidate']
                
                with torch.no_grad():
                    for _ in range(num_rollouts):
                        obs, _ = training_env.reset()
                        episode_reward = 0
                        
                        # Initialize states
                        z = torch.zeros(1, latent_size, device=self.device)
                        h = torch.zeros(1, hidden_size, device=self.device)
                        
                        for step in range(200):  # Max episode length
                            # Get action from controller
                            if action_type == 'continuous':
                                action_tensor = controller.get_action(z, h, deterministic=True)
                                if isinstance(action_tensor, tuple):
                                    action_tensor = action_tensor[0]
                                action = action_tensor.cpu().numpy().squeeze()
                            else:
                                action = controller.get_action(z, h, deterministic=True)
                                action = action.cpu().numpy().item()
                            
                            # Take step
                            obs, reward, terminated, truncated, _ = training_env.step(action)
                            episode_reward += reward
                            
                            if terminated or truncated:
                                break
                            
                            # Update states (simplified for demo)
                            # In practice, would use VAE/MDN-RNN to compute next z, h
                            z = torch.randn(1, latent_size, device=self.device) * 0.1
                            h = torch.randn(1, hidden_size, device=self.device) * 0.1
                        
                        total_reward += episode_reward
                
                return total_reward / num_rollouts
            
            # CMA-ES training loop
            best_fitness = -float('inf')
            best_params = None
            
            logger.info("Starting CMA-ES training...")
            
            for generation in tqdm(range(self.config['cma_generations']), desc="CMA-ES"):
                # Get candidate solutions
                candidates = cmaes_trainer.ask()
                
                # Evaluate candidates
                fitness_values = []
                for candidate in candidates:
                    fitness = evaluate_controller(candidate)
                    fitness_values.append(fitness)
                
                fitness_values = np.array(fitness_values)
                
                # Update CMA-ES
                cmaes_trainer.tell(candidates, fitness_values)
                
                # Track best
                gen_best_idx = np.argmax(fitness_values)
                gen_best_fitness = fitness_values[gen_best_idx]
                
                if gen_best_fitness > best_fitness:
                    best_fitness = gen_best_fitness
                    best_params = candidates[gen_best_idx]
                
                # Log progress
                if generation % max(1, self.config['cma_generations'] // 10) == 0:
                    stats = cmaes_trainer.get_stats()
                    logger.info(f"Generation {generation}: best={gen_best_fitness:.3f}, mean={stats['mean_fitness']:.3f}")
                    
                    if self.writer:
                        self.writer.add_scalar('controller/best_fitness', gen_best_fitness, generation)
                        self.writer.add_scalar('controller/mean_fitness', stats['mean_fitness'], generation)
            
            # Set best parameters
            if best_params is not None:
                controller.set_parameters(best_params)
            
            # Save controller
            controller_path = self.checkpoint_dir / "controller_best.pt"
            torch.save({
                'state_dict': controller.state_dict(),
                'config': {
                    'input_size': input_size,
                    'action_size': action_size,
                    'hidden_sizes': (),
                    'action_type': action_type
                },
                'best_fitness': best_fitness,
                'best_params': best_params
            }, controller_path)
            
            # Clean up
            if hasattr(training_env, 'close'):
                training_env.close()
            
            duration = time.time() - start_time
            self.log_stage_completion('controller', duration, True, best_fitness)
            
            logger.info(f"Best controller fitness: {best_fitness:.3f}")
            
            if self.verify_stage_output('controller', str(controller_path)):
                return str(controller_path)
            else:
                raise RuntimeError("Controller training verification failed")
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_stage_completion('controller', duration, False)
            logger.error(f"Controller training failed: {e}")
            raise
    
    def stage_6_evaluate(self, controller_path: str) -> Dict[str, Any]:
        """Stage 6: Evaluate trained controller and save videos."""
        logger.info("=" * 60)
        logger.info("📊 STAGE 6: EVALUATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            if not GYM_AVAILABLE:
                raise ImportError("Gymnasium required for evaluation")
            
            # Load controller
            checkpoint = torch.load(controller_path, map_location=self.device)
            controller_config = checkpoint['config']
            
            controller = Controller(**controller_config).to(self.device)
            controller.load_state_dict(checkpoint['state_dict'])
            controller.eval()
            
            logger.info(f"Loaded controller with fitness: {checkpoint.get('best_fitness', 'N/A')}")
            
            # Create evaluation environment
            env = gym.make(self.config['env'])
            
            # Evaluation settings
            num_eval_episodes = 5
            episode_rewards = []
            episode_lengths = []
            
            # Video recording setup
            videos_dir = None
            if self.config.get('save_videos', False):
                videos_dir = self.checkpoint_dir / "videos"
                videos_dir.mkdir(exist_ok=True)
                logger.info(f"Saving videos to: {videos_dir}")
            
            logger.info(f"Evaluating for {num_eval_episodes} episodes...")
            
            # Load metadata for state sizes
            with open(self.checkpoint_dir / "raw_data" / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            if metadata['is_pixel_env']:
                latent_size = 32
            else:
                latent_size = metadata['obs_shape'][0]
            hidden_size = 256
            
            for episode in range(num_eval_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Initialize states
                z = torch.zeros(1, latent_size, device=self.device)
                h = torch.zeros(1, hidden_size, device=self.device)
                
                frames = []
                
                for step in range(1000):  # Max episode length
                    if self.config.get('save_videos', False):
                        if hasattr(env, 'render'):
                            frame = env.render()
                            if frame is not None:
                                frames.append(frame)
                    
                    with torch.no_grad():
                        if controller_config['action_type'] == 'continuous':
                            action_tensor = controller.get_action(z, h, deterministic=True)
                            if isinstance(action_tensor, tuple):
                                action_tensor = action_tensor[0]
                            action = action_tensor.cpu().numpy().squeeze()
                        else:
                            action = controller.get_action(z, h, deterministic=True)
                            action = action.cpu().numpy().item()
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if terminated or truncated:
                        break
                    
                    # Update states (simplified)
                    z = torch.randn(1, latent_size, device=self.device) * 0.1
                    h = torch.randn(1, hidden_size, device=self.device) * 0.1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                logger.info(f"Episode {episode+1}: reward={episode_reward:.2f}, length={episode_length}")
                
                # Save video if requested
                if self.config.get('save_videos', False) and frames and videos_dir:
                    try:
                        import imageio
                        video_path = videos_dir / f"episode_{episode+1}.mp4"
                        imageio.mimsave(video_path, frames, fps=30)
                        logger.info(f"Saved video: {video_path}")
                    except ImportError:
                        logger.warning("imageio not available for video saving")
                    except Exception as e:
                        logger.warning(f"Video saving failed: {e}")
            
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
            
            # Save results
            results_path = self.checkpoint_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_results = {}
                for k, v in results.items():
                    if isinstance(v, (np.ndarray, list)):
                        json_results[k] = [float(x) for x in v]
                    else:
                        json_results[k] = float(v)
                json.dump(json_results, f, indent=2)
            
            duration = time.time() - start_time
            self.log_stage_completion('eval', duration, True, results['mean_reward'])
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            logger.info(f"  Min/Max reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
            logger.info(f"  Mean episode length: {results['mean_length']:.1f}")
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_stage_completion('eval', duration, False)
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_pipeline(self, stages: List[str]) -> Dict[str, Any]:
        """Run the complete pipeline or specified stages."""
        logger.info("🚀 STARTING WORLD MODELS PIPELINE")
        logger.info(f"Stages: {stages}")
        logger.info(f"Environment: {self.config['env']}")
        logger.info(f"Device: {self.device}")
        
        pipeline_start = time.time()
        stage_outputs = {}
        
        try:
            # Stage 1: Data Collection
            if 'collect' in stages:
                stage_outputs['data_path'] = self.stage_1_collect_data()
            else:
                # Check if data exists from previous run
                data_path = self.checkpoint_dir / "raw_data"
                if data_path.exists():
                    stage_outputs['data_path'] = str(data_path)
                    logger.info(f"Using existing data: {data_path}")
                else:
                    raise RuntimeError("Data collection required but not requested and no existing data found")
            
            # Stage 2: VAE Training
            if 'vae' in stages:
                stage_outputs['vae_path'] = self.stage_2_train_vae(stage_outputs['data_path'])
            else:
                vae_path = self.checkpoint_dir / "vae.pt"
                if vae_path.exists():
                    stage_outputs['vae_path'] = str(vae_path)
                    logger.info(f"Using existing VAE: {vae_path}")
                else:
                    # Check if it's a non-pixel environment
                    with open(Path(stage_outputs['data_path']) / "metadata.json", 'r') as f:
                        metadata = json.load(f)
                    if not metadata['is_pixel_env']:
                        stage_outputs['vae_path'] = "skipped"
                    else:
                        raise RuntimeError("VAE training required but not requested and no existing VAE found")
            
            # Stage 3: Latent Encoding
            if 'encode' in stages:
                stage_outputs['latents_path'] = self.stage_3_encode_latents(
                    stage_outputs['data_path'], 
                    stage_outputs['vae_path']
                )
            else:
                latents_path = self.checkpoint_dir / "latent_sequences.pkl"
                if latents_path.exists():
                    stage_outputs['latents_path'] = str(latents_path)
                    logger.info(f"Using existing latents: {latents_path}")
                else:
                    raise RuntimeError("Latent encoding required but not requested and no existing latents found")
            
            # Stage 4: MDN-RNN Training
            if 'mdnrnn' in stages:
                stage_outputs['mdnrnn_path'] = self.stage_4_train_mdnrnn(stage_outputs['latents_path'])
            else:
                mdnrnn_path = self.checkpoint_dir / "mdnrnn.pt"
                if mdnrnn_path.exists():
                    stage_outputs['mdnrnn_path'] = str(mdnrnn_path)
                    logger.info(f"Using existing MDN-RNN: {mdnrnn_path}")
                else:
                    raise RuntimeError("MDN-RNN training required but not requested and no existing MDN-RNN found")
            
            # Stage 5: Controller Training
            if 'controller' in stages:
                stage_outputs['controller_path'] = self.stage_5_train_controller(
                    stage_outputs['vae_path'],
                    stage_outputs['mdnrnn_path']
                )
            else:
                controller_path = self.checkpoint_dir / "controller_best.pt"
                if controller_path.exists():
                    stage_outputs['controller_path'] = str(controller_path)
                    logger.info(f"Using existing controller: {controller_path}")
                else:
                    raise RuntimeError("Controller training required but not requested and no existing controller found")
            
            # Stage 6: Evaluation
            if 'eval' in stages:
                stage_outputs['eval_results'] = self.stage_6_evaluate(stage_outputs['controller_path'])
            
            # Pipeline completion
            total_time = time.time() - pipeline_start
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time:.2f} seconds")
            
            # Print timing summary
            logger.info("Stage timings:")
            for stage, duration in self.stage_timings.items():
                logger.info(f"  {stage}: {duration:.2f}s")
            
            if 'eval_results' in stage_outputs:
                results = stage_outputs['eval_results']
                logger.info(f"Final performance: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            
            # Print output locations
            logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
            logger.info(f"Logs saved to: {self.log_dir}")
            
            # Example replay command
            if 'controller_path' in stage_outputs:
                logger.info(f"To replay best controller:")
                logger.info(f"  python copilot_run_all.py --env {self.config['env']} --stage eval --resume {self.checkpoint_dir}")
            
            return stage_outputs
            
        except Exception as e:
            total_time = time.time() - pipeline_start
            logger.error("=" * 60)
            logger.error("❌ PIPELINE FAILED!")
            logger.error("=" * 60)
            logger.error(f"Error after {total_time:.2f} seconds: {e}")
            logger.error("Full traceback:")
            traceback.print_exc()
            
            # Cleanup
            if self.writer:
                self.writer.close()
            
            raise
        
        finally:
            # Always cleanup
            if self.writer:
                self.writer.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="World Models Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python copilot_run_all.py --env PongNoFrameskip-v5 --mode quick
  
  # Full training run
  python copilot_run_all.py --env CarRacing-v2 --mode full --device cuda
  
  # Run specific stages
  python copilot_run_all.py --stage collect,vae --env LunarLander-v2
        """
    )
    
    parser.add_argument('--env', default='PongNoFrameskip-v5',
                       help='Environment name (default: PongNoFrameskip-v5)')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Training mode (default: quick)')
    parser.add_argument('--device', default=DEVICE_DEFAULT,
                       help=f'Device for training (default: {DEVICE_DEFAULT})')
    parser.add_argument('--stage', default='all',
                       help='Stages to run: all or comma-separated (collect,vae,encode,mdnrnn,controller,eval)')
    parser.add_argument('--checkpoint-dir', 
                       help='Checkpoint directory (default: ./runs/<env>_worldmodel)')
    
    # Training parameters
    parser.add_argument('--num-random-rollouts', type=int,
                       help='Number of random rollouts (quick: 200, full: 10000)')
    parser.add_argument('--vae-epochs', type=int,
                       help='VAE training epochs (quick: 1, full: 10)')
    parser.add_argument('--mdnrnn-epochs', type=int,
                       help='MDN-RNN training epochs (quick: 2, full: 20)')
    parser.add_argument('--cma-pop-size', type=int,
                       help='CMA-ES population size (quick: 8, full: 64)')
    parser.add_argument('--cma-generations', type=int,
                       help='CMA-ES generations (quick: 5, full: 800)')
    parser.add_argument('--rollouts-per-candidate', type=int,
                       help='Rollouts per candidate (quick: 2, full: 16)')
    
    # Flags
    parser.add_argument('--train-in-dream', action='store_true', default=True,
                       help='Train controller in dream environment (default: True)')
    parser.add_argument('--save-videos', action='store_true', default=False,
                       help='Save evaluation videos (default: False)')
    parser.add_argument('--resume', 
                       help='Resume from checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Determine stages to run
    if args.stage == 'all':
        stages = ['collect', 'vae', 'encode', 'mdnrnn', 'controller', 'eval']
    else:
        stages = [s.strip() for s in args.stage.split(',')]
    
    valid_stages = {'collect', 'vae', 'encode', 'mdnrnn', 'controller', 'eval'}
    invalid_stages = set(stages) - valid_stages
    if invalid_stages:
        logger.error(f"Invalid stages: {invalid_stages}")
        logger.error(f"Valid stages: {valid_stages}")
        sys.exit(1)
    
    # Set mode-specific defaults
    if args.mode == 'quick':
        defaults = {
            'num_random_rollouts': 200,
            'vae_epochs': 1,
            'mdnrnn_epochs': 2,
            'cma_pop_size': 8,
            'cma_generations': 5,
            'rollouts_per_candidate': 2
        }
    else:  # full
        defaults = {
            'num_random_rollouts': 10000,
            'vae_epochs': 10,
            'mdnrnn_epochs': 20,
            'cma_pop_size': 64,
            'cma_generations': 800,
            'rollouts_per_candidate': 16
        }
    
    # Apply defaults if not specified
    for key, default_val in defaults.items():
        arg_key = key.replace('_', '-')
        if getattr(args, key) is None:
            setattr(args, key, default_val)
    
    # Set checkpoint directory
    if args.checkpoint_dir is None:
        env_name = args.env.replace(':', '_').replace('-', '_')
        args.checkpoint_dir = f"./runs/{env_name}_worldmodel"
    
    # Resume from checkpoint if specified
    if args.resume:
        args.checkpoint_dir = args.resume
        logger.info(f"Resuming from: {args.resume}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    # Create configuration
    config = {
        'env': args.env,
        'mode': args.mode,
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'num_random_rollouts': args.num_random_rollouts,
        'vae_epochs': args.vae_epochs,
        'mdnrnn_epochs': args.mdnrnn_epochs,
        'cma_pop_size': args.cma_pop_size,
        'cma_generations': args.cma_generations,
        'rollouts_per_candidate': args.rollouts_per_candidate,
        'train_in_dream': args.train_in_dream,
        'save_videos': args.save_videos,
        'seed': args.seed
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Initialize and run pipeline
        pipeline = WorldModelsPipeline(config)
        results = pipeline.run_pipeline(stages)
        
        logger.info("🎉 SUCCESS: Pipeline completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error("Please check the logs above for detailed error information")
        
        # Print helpful suggestions
        if "gym" in str(e).lower() or "gymnasium" in str(e).lower():
            logger.error("💡 Try: pip install gymnasium[atari]")
        elif "cuda" in str(e).lower():
            logger.error("💡 Try: --device cpu")
        elif "import" in str(e).lower():
            logger.error("💡 Try: pip install -r requirements.txt")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
