"""
Training utilities for World Models

This module provides training loops, data loading, logging, and visualization
utilities for training the World Models architecture components.

Key features:
- VAE training with β-annealing and reconstruction visualization
- MDN-RNN training with sequence handling and loss monitoring  
- Controller training with CMA-ES and optional PPO
- Logging and plotting utilities
- Checkpoint saving/loading
- Progress tracking and visualization

Based on Ha & Schmidhuber (2018): "World Models"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Callable
import os
from tqdm import tqdm
import pickle
from collections import defaultdict
import time

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RolloutDataset(Dataset):
    """
    Dataset for loading rollout data for VAE training.
    
    Handles frame sequences from environment rollouts and provides
    efficient batching and data augmentation.
    """
    
    def __init__(
        self,
        frames: np.ndarray,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize rollout dataset.
        
        Args:
            frames: Frame sequences (num_rollouts, seq_len, 64, 64, 3)
            actions: Action sequences (num_rollouts, seq_len, action_size)
            rewards: Reward sequences (num_rollouts, seq_len)
            transform: Optional data augmentation transform
        """
        # Flatten sequences: (num_rollouts * seq_len, 64, 64, 3)
        self.frames = frames.reshape(-1, *frames.shape[2:])
        
        if actions is not None:
            self.actions = actions.reshape(-1, actions.shape[2])
        else:
            self.actions = None
            
        if rewards is not None:
            self.rewards = rewards.reshape(-1)
        else:
            self.rewards = None
            
        self.transform = transform
        
        # Remove zero-padded frames (assuming zero frames are padding)
        valid_mask = self.frames.sum(axis=(1, 2, 3)) > 0
        self.frames = self.frames[valid_mask]
        
        if self.actions is not None:
            self.actions = self.actions[valid_mask]
        if self.rewards is not None:
            self.rewards = self.rewards[valid_mask]
            
        print(f"Dataset created with {len(self.frames)} frames")
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame = self.frames[idx]
        
        # Apply transforms if provided
        if self.transform:
            frame = self.transform(frame)
        
        # Convert to tensor and permute to (C, H, W)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        
        item = {'frame': frame_tensor}
        
        if self.actions is not None:
            item['action'] = torch.from_numpy(self.actions[idx]).float()
        if self.rewards is not None:
            item['reward'] = torch.tensor(self.rewards[idx], dtype=torch.float32)
            
        return item

class SequenceDataset(Dataset):
    """
    Dataset for sequence-based training (MDN-RNN).
    
    Creates training sequences from rollout data with proper
    temporal alignment for next-step prediction.
    """
    
    def __init__(
        self,
        latent_sequences: torch.Tensor,
        action_sequences: torch.Tensor,
        sequence_length: int = 100,
        overlap: int = 50
    ):
        """
        Initialize sequence dataset.
        
        Args:
            latent_sequences: Latent state sequences (num_rollouts, seq_len, latent_size)
            action_sequences: Action sequences (num_rollouts, seq_len, action_size)
            sequence_length: Length of training sequences
            overlap: Overlap between consecutive sequences
        """
        self.latent_sequences = latent_sequences
        self.action_sequences = action_sequences
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        # Create sequence indices
        self.sequence_indices = []
        
        num_rollouts, max_seq_len = latent_sequences.shape[:2]
        
        for rollout_idx in range(num_rollouts):
            # Find actual sequence length (non-zero entries)
            rollout_latents = latent_sequences[rollout_idx]
            actual_length = self._find_actual_length(rollout_latents)
            
            # Create overlapping sequences
            start_indices = range(0, actual_length - sequence_length, 
                                sequence_length - overlap)
            
            for start_idx in start_indices:
                if start_idx + sequence_length < actual_length:
                    self.sequence_indices.append((rollout_idx, start_idx))
        
        print(f"Created {len(self.sequence_indices)} training sequences")
    
    def _find_actual_length(self, sequence: torch.Tensor) -> int:
        """Find actual sequence length by detecting padding"""
        # Assume padding is zeros
        non_zero_mask = (sequence != 0).any(dim=-1)
        if non_zero_mask.any():
            return non_zero_mask.nonzero()[-1].item() + 1
        return 0
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rollout_idx, start_idx = self.sequence_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract sequences
        z_seq = self.latent_sequences[rollout_idx, start_idx:end_idx]
        action_seq = self.action_sequences[rollout_idx, start_idx:end_idx]
        
        # Target is next latent states (shifted by 1)
        z_target = self.latent_sequences[rollout_idx, start_idx+1:end_idx+1]
        
        return {
            'z': z_seq,
            'actions': action_seq,
            'z_target': z_target
        }

class TrainingLogger:
    """
    Training logger for metrics, losses, and visualizations.
    
    Integrates with TensorBoard for real-time monitoring and
    provides utilities for plotting and progress tracking.
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of current experiment
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.step_counters = defaultdict(int)
        
        print(f"Logging to {self.log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar metric"""
        if step is None:
            step = self.step_counters[tag]
            
        self.writer.add_scalar(tag, value, step)
        self.metrics[tag].append((step, value))
        self.step_counters[tag] = max(self.step_counters[tag], step + 1)
    
    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """Log image to TensorBoard"""
        if step is None:
            step = self.step_counters[tag]
            
        self.writer.add_image(tag, image, step)
        self.step_counters[tag] = max(self.step_counters[tag], step + 1)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram to TensorBoard"""
        if step is None:
            step = self.step_counters[tag]
            
        self.writer.add_histogram(tag, values, step)
        self.step_counters[tag] = max(self.step_counters[tag], step + 1)
    
    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log dictionary of metrics"""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)
    
    def plot_metrics(self, tags: List[str], save_path: Optional[str] = None):
        """Plot training metrics"""
        num_plots = len(tags)
        fig, axes = plt.subplots((num_plots + 1) // 2, 2, figsize=(15, 5 * ((num_plots + 1) // 2)))
        
        if num_plots == 1:
            axes = [axes]
        elif num_plots <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, tag in enumerate(tags):
            if tag in self.metrics:
                steps, values = zip(*self.metrics[tag])
                axes[i].plot(steps, values, linewidth=2)
                axes[i].set_title(tag)
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(tags), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics plot to {save_path}")
        
        plt.show()
    
    def close(self):
        """Close logger"""
        self.writer.close()

class VAETrainer:
    """
    Trainer for the Convolutional VAE (Vision Model).
    
    Handles VAE training with β-annealing, reconstruction visualization,
    and comprehensive logging of training progress.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        beta: float = 4.0,
        device: torch.device = torch.device('cpu'),
        logger: Optional[TrainingLogger] = None
    ):
        """
        Initialize VAE trainer.
        
        Args:
            model: VAE model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate for Adam optimizer
            beta: β parameter for β-VAE
            device: Training device
            logger: Training logger
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.beta = beta
        
    def train_epoch(self, warmup_factor: float = 1.0) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            warmup_factor: β warmup factor (gradually increase KL weight)
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                frames = batch['frame'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                losses = self.model.compute_loss(
                    frames, 
                    warmup_factor=warmup_factor
                )
                
                # Backward pass
                losses['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate metrics
                for key, value in losses.items():
                    epoch_metrics[key] += value.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'recon': f"{losses['reconstruction_loss'].item():.4f}",
                    'kl': f"{losses['kl_loss'].item():.4f}"
                })
                
                # Log batch metrics
                if self.logger and batch_idx % 100 == 0:
                    step = self.epoch * num_batches + batch_idx
                    self.logger.log_dict(
                        {k: v.item() for k, v in losses.items()},
                        step=step,
                        prefix="train_batch"
                    )
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return dict(epoch_metrics)
    
    def validate(self) -> Dict[str, float]:
        """Validate model and return metrics"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        val_metrics = defaultdict(float)
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch['frame'].to(self.device)
                
                # Forward pass
                losses = self.model.compute_loss(frames)
                
                # Accumulate metrics
                for key, value in losses.items():
                    val_metrics[key] += value.item()
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return dict(val_metrics)
    
    def visualize_reconstructions(self, num_samples: int = 8):
        """Visualize model reconstructions"""
        self.model.eval()
        
        # Get sample batch
        sample_batch = next(iter(self.train_loader))
        frames = sample_batch['frame'][:num_samples].to(self.device)
        
        with torch.no_grad():
            reconstructions = self.model.reconstruct(frames)
        
        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
        
        for i in range(num_samples):
            # Original
            orig = frames[i].cpu().permute(1, 2, 0).numpy() / 255.0
            axes[0, i].imshow(np.clip(orig, 0, 1))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstruction
            recon = reconstructions[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(np.clip(recon, 0, 1))
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
        
        plt.suptitle(f'VAE Reconstructions - Epoch {self.epoch+1}')
        plt.tight_layout()
        
        # Log to TensorBoard if available
        if self.logger:
            # Create grid image
            grid_img = torch.cat([
                frames[:num_samples] / 255.0,
                reconstructions[:num_samples]
            ], dim=0)
            
            from torchvision.utils import make_grid
            grid = make_grid(grid_img, nrow=num_samples, padding=2, pad_value=1.0)
            self.logger.log_image('reconstructions', grid, step=self.epoch)
        
        plt.show()
    
    def train(
        self,
        num_epochs: int,
        warmup_epochs: int = 5,
        save_frequency: int = 10,
        save_path: str = './checkpoints/vae.pth'
    ):
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of training epochs
            warmup_epochs: Number of β-annealing epochs
            save_frequency: Save model every N epochs
            save_path: Path to save best model
        """
        print(f"Starting VAE training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # β-annealing: gradually increase KL weight
            if epoch < warmup_epochs:
                warmup_factor = epoch / warmup_epochs
            else:
                warmup_factor = 1.0
            
            # Train epoch
            start_time = time.time()
            train_metrics = self.train_epoch(warmup_factor)
            train_time = time.time() - start_time
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            if self.logger:
                self.logger.log_dict(train_metrics, step=epoch, prefix="train")
                self.logger.log_dict(val_metrics, step=epoch, prefix="val")
                self.logger.log_scalar("warmup_factor", warmup_factor, step=epoch)
                self.logger.log_scalar("learning_rate", 
                                     self.optimizer.param_groups[0]['lr'], step=epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} ({train_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Recon: {train_metrics['reconstruction_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}")
            
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                      f"Recon: {val_metrics['reconstruction_loss']:.4f}, "
                      f"KL: {val_metrics['kl_loss']:.4f}")
                
                # Update learning rate
                self.scheduler.step(val_metrics['total_loss'])
                
                # Save best model
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(save_path)
                    print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
            
            # Visualize reconstructions periodically
            if (epoch + 1) % 10 == 0:
                self.visualize_reconstructions()
            
            # Save checkpoint periodically
            if (epoch + 1) % save_frequency == 0:
                checkpoint_path = save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path)
        
        print("VAE training completed!")
        
        # Final visualization
        self.visualize_reconstructions()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")

def create_data_loaders(
    frames: np.ndarray,
    actions: Optional[np.ndarray] = None,
    rewards: Optional[np.ndarray] = None,
    batch_size: int = 128,
    val_split: float = 0.2,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        frames: Frame data
        actions: Action data
        rewards: Reward data
        batch_size: Batch size
        val_split: Validation split ratio
        num_workers: Number of data loader workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = RolloutDataset(frames, actions, rewards)
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader

def test_training_utilities():
    """Test training utilities"""
    print("Testing training utilities...")
    
    # Create dummy data
    frames = np.random.randint(0, 256, (10, 100, 64, 64, 3), dtype=np.uint8)
    actions = np.random.randn(10, 100, 6).astype(np.float32)
    rewards = np.random.randn(10, 100).astype(np.float32)
    
    # Test dataset
    dataset = RolloutDataset(frames, actions, rewards)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample frame shape: {sample['frame'].shape}")
    print(f"Sample action shape: {sample['action'].shape}")
    
    # Test data loaders
    train_loader, val_loader = create_data_loaders(
        frames, actions, rewards, batch_size=8, num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"Batch frame shape: {batch['frame'].shape}")
    
    # Test logger
    logger = TrainingLogger('./test_logs', 'test_experiment')
    logger.log_scalar('test_metric', 1.0, step=0)
    logger.log_dict({'loss': 0.5, 'accuracy': 0.8}, step=0)
    logger.close()
    
    print("Training utilities test passed!")

if __name__ == "__main__":
    test_training_utilities()
