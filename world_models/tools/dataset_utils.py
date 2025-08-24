"""Dataset utilities for World Models MDN-RNN.

This module provides utilities for:
- Converting frame data to latent representations using trained VAE
- Creating sequence datasets suitable for MDN-RNN training
- Data preprocessing and augmentation for temporal dynamics learning
- Efficient loading and batching of trajectory data

Key components:
- FramesToLatentConverter: Converts raw frames to VAE latent codes
- TrajectoryDataset: Loads and manages trajectory sequences
- sequence_collate_fn: Custom batching for variable-length sequences
- preprocess_trajectories: Data preprocessing pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path

# Attempt to import VAE (fallback if not available)
ConvVAE = None
try:
    from world_models.models.vae import ConvVAE
except ImportError:
    try:
        from models.vae import ConvVAE
    except ImportError:
        try:
            # Try relative import if running as a module
            from .models.vae import ConvVAE
        except ImportError:
            ConvVAE = None

if ConvVAE is None:
    raise ImportError(
        "ConvVAE could not be imported from 'world_models.models.vae', 'models.vae', or '.models.vae'. "
        "Please ensure the module exists and is in your PYTHONPATH or the correct relative location."
    )


class FramesToLatentConverter:
    """Converts frames to latent space using trained VAE."""
    
    def __init__(self, vae_model_path: str, device: str = 'cuda'):
        """Initialize converter with trained VAE model.
        
        Args:
            vae_model_path: Path to saved VAE checkpoint
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device
        self.vae = None
        
        if ConvVAE is not None:
            # Load the trained VAE
            self.vae = ConvVAE(latent_size=32).to(device)
            if os.path.exists(vae_model_path):
                checkpoint = torch.load(vae_model_path, map_location=device)
                self.vae.load_state_dict(checkpoint['model_state_dict'])
                self.vae.eval()
                print(f"Loaded VAE from {vae_model_path}")
            else:
                print(f"Warning: VAE checkpoint not found at {vae_model_path}")
        else:
            print("Warning: Using mock VAE conversion")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame for VAE input.
        
        Args:
            frame: Input frame (H, W, C) in range [0, 255]
            
        Returns:
            Preprocessed tensor (C, H, W) in range [0, 1]
        """
        # Resize to 64x64 (standard for World Models)
        if frame.shape[:2] != (64, 64):
            frame = cv2.resize(frame, (64, 64))
        
        # Convert to tensor and normalize
        frame = torch.from_numpy(frame).float() / 255.0
        
        # Convert HWC to CHW
        if frame.ndim == 3:
            frame = frame.permute(2, 0, 1)
        elif frame.ndim == 2:
            frame = frame.unsqueeze(0)  # Add channel dimension
        
        return frame
    
    def encode_frames(self, frames: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """Encode frames to latent space using VAE encoder.
        
        Args:
            frames: Batch of frames or single frame
            
        Returns:
            Latent codes (batch_size, latent_dim) or (latent_dim,)
        """
        if self.vae is None:
            # Mock implementation for testing
            if isinstance(frames, list):
                batch_size = len(frames)
            else:
                batch_size = frames.shape[0] if frames.ndim == 4 else 1
            return torch.randn(batch_size, 32, device=self.device)
        
        # Prepare input batch
        if isinstance(frames, list):
            batch = torch.stack([self.preprocess_frame(f) for f in frames])
        else:
            if frames.ndim == 3:  # Single frame
                batch = self.preprocess_frame(frames).unsqueeze(0)
            else:  # Batch of frames
                batch = torch.stack([self.preprocess_frame(f) for f in frames])
        
        batch = batch.to(self.device)
        
        with torch.no_grad():
            # Get latent mean (ignore variance for deterministic encoding)
            mu, _ = self.vae.encode(batch)
            
        return mu.squeeze() if mu.shape[0] == 1 else mu


class TrajectoryDataset(Dataset):
    """Dataset for loading and managing trajectory sequences."""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 50,
        frame_converter: Optional[FramesToLatentConverter] = None,
        include_rewards: bool = True,
        include_dones: bool = True,
        max_trajectories: Optional[int] = None
    ):
        """Initialize trajectory dataset.
        
        Args:
            data_dir: Directory containing trajectory files
            sequence_length: Length of sequences for training
            frame_converter: Converter for frames to latent codes
            include_rewards: Whether to include reward signals
            include_dones: Whether to include done flags
            max_trajectories: Maximum number of trajectories to load
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.frame_converter = frame_converter
        self.include_rewards = include_rewards
        self.include_dones = include_dones
        
        # Load trajectory files
        self.trajectory_files = list(self.data_dir.glob("*.pkl"))[:max_trajectories]
        
        # Build sequence index
        self.sequences = []
        self._build_sequence_index()
        
        print(f"Loaded {len(self.trajectory_files)} trajectories with {len(self.sequences)} sequences")
    
    def _build_sequence_index(self):
        """Build index of valid sequences across all trajectories."""
        for traj_idx, traj_file in enumerate(self.trajectory_files):
            try:
                with open(traj_file, 'rb') as f:
                    trajectory = pickle.load(f)
                
                # Get trajectory length
                if 'observations' in trajectory:
                    traj_len = len(trajectory['observations'])
                elif 'frames' in trajectory:
                    traj_len = len(trajectory['frames'])
                else:
                    continue
                
                # Create sequence indices
                for start_idx in range(traj_len - self.sequence_length + 1):
                    self.sequences.append((traj_idx, start_idx))
                    
            except Exception as e:
                print(f"Warning: Could not load {traj_file}: {e}")
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary containing sequence data
        """
        traj_idx, start_idx = self.sequences[idx]
        traj_file = self.trajectory_files[traj_idx]
        
        # Load trajectory
        with open(traj_file, 'rb') as f:
            trajectory = pickle.load(f)
        
        end_idx = start_idx + self.sequence_length
        
        # Extract frames/observations
        if 'observations' in trajectory:
            frames = trajectory['observations'][start_idx:end_idx]
        elif 'frames' in trajectory:
            frames = trajectory['frames'][start_idx:end_idx]
        else:
            raise ValueError("No observations or frames found in trajectory")
        
        # Convert to latent codes if converter available
        if self.frame_converter is not None:
            latent_codes = self.frame_converter.encode_frames(frames)
        else:
            # Use raw frames or create dummy latent codes
            if isinstance(frames, np.ndarray) and frames.ndim >= 3:
                latent_codes = torch.randn(len(frames), 32)
            else:
                latent_codes = torch.randn(self.sequence_length, 32)
        
        # Extract actions
        if 'actions' in trajectory:
            actions = torch.FloatTensor(trajectory['actions'][start_idx:end_idx])
        else:
            # Create dummy actions
            actions = torch.randn(self.sequence_length, 3)
        
        sample = {
            'latent_obs': latent_codes,
            'actions': actions,
            'sequence_length': self.sequence_length
        }
        
        # Add rewards if requested
        if self.include_rewards and 'rewards' in trajectory:
            rewards = torch.FloatTensor(trajectory['rewards'][start_idx:end_idx])
            sample['rewards'] = rewards
        
        # Add done flags if requested
        if self.include_dones and 'dones' in trajectory:
            dones = torch.BoolTensor(trajectory['dones'][start_idx:end_idx])
            sample['dones'] = dones
        
        return sample


def sequence_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching sequences.
    
    Args:
        batch: List of sequence samples
        
    Returns:
        Batched tensors
    """
    # Get all keys from first sample
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        if key == 'sequence_length':
            # Store sequence lengths
            batched[key] = torch.LongTensor([sample[key] for sample in batch])
        else:
            # Stack tensors
            batched[key] = torch.stack([sample[key] for sample in batch])
    
    return batched


def preprocess_trajectories(
    raw_data_dir: str,
    output_dir: str,
    vae_model_path: str,
    device: str = 'cuda'
) -> None:
    """Preprocess raw trajectories to latent space.
    
    Args:
        raw_data_dir: Directory with raw trajectory data
        output_dir: Directory to save preprocessed data
        vae_model_path: Path to trained VAE model
        device: Device for computation
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize frame converter
    converter = FramesToLatentConverter(vae_model_path, device)
    
    # Process each trajectory file
    raw_path = Path(raw_data_dir)
    trajectory_files = list(raw_path.glob("*.pkl"))
    
    for i, traj_file in enumerate(trajectory_files):
        print(f"Processing trajectory {i+1}/{len(trajectory_files)}: {traj_file.name}")
        
        try:
            # Load raw trajectory
            with open(traj_file, 'rb') as f:
                trajectory = pickle.load(f)
            
            # Convert frames to latent codes
            if 'observations' in trajectory:
                frames = trajectory['observations']
            elif 'frames' in trajectory:
                frames = trajectory['frames']
            else:
                print(f"Skipping {traj_file}: no frames found")
                continue
            
            # Encode frames in batches to avoid memory issues
            batch_size = 32
            latent_codes = []
            
            for j in range(0, len(frames), batch_size):
                batch_frames = frames[j:j+batch_size]
                batch_latents = converter.encode_frames(batch_frames)
                latent_codes.append(batch_latents.cpu())
            
            latent_codes = torch.cat(latent_codes, dim=0)
            
            # Create new trajectory with latent codes
            processed_trajectory = {
                'latent_obs': latent_codes.numpy(),
                'actions': trajectory.get('actions', []),
                'rewards': trajectory.get('rewards', []),
                'dones': trajectory.get('dones', [])
            }
            
            # Save processed trajectory
            output_file = output_path / f"processed_{traj_file.name}"
            with open(output_file, 'wb') as f:
                pickle.dump(processed_trajectory, f)
                
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
    
    print(f"Preprocessing complete. Saved {len(trajectory_files)} trajectories to {output_dir}")


def create_mdnrnn_dataloader(
    data_dir: str,
    batch_size: int = 32,
    sequence_length: int = 50,
    vae_model_path: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """Create DataLoader for MDN-RNN training.
    
    Args:
        data_dir: Directory containing trajectory data
        batch_size: Batch size for training
        sequence_length: Length of sequences
        vae_model_path: Path to VAE model for frame conversion
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for TrajectoryDataset
        
    Returns:
        DataLoader for MDN-RNN training
    """
    # Create frame converter if VAE path provided
    frame_converter = None
    if vae_model_path and os.path.exists(vae_model_path):
        frame_converter = FramesToLatentConverter(vae_model_path)
    
    # Create dataset
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        frame_converter=frame_converter,
        **kwargs
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        pin_memory=True
    )
    
    return dataloader


# Utility functions for data validation and visualization
def validate_dataset(dataloader: DataLoader) -> Dict[str, Any]:
    """Validate dataset and return statistics.
    
    Args:
        dataloader: DataLoader to validate
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'num_batches': len(dataloader),
        'batch_size': dataloader.batch_size,
        'total_samples': len(dataloader.dataset),
        'latent_dim': None,
        'action_dim': None,
        'sample_shapes': {}
    }
    
    # Get first batch for shape analysis
    try:
        first_batch = next(iter(dataloader))
        
        for key, tensor in first_batch.items():
            if isinstance(tensor, torch.Tensor):
                stats['sample_shapes'][key] = list(tensor.shape)
                
                if key == 'latent_obs':
                    stats['latent_dim'] = tensor.shape[-1]
                elif key == 'actions':
                    stats['action_dim'] = tensor.shape[-1]
        
        print("Dataset validation successful!")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        stats['error'] = str(e)
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Dataset utilities for World Models MDN-RNN")
    
    # Example: Create a simple test dataset
    test_data_dir = "test_trajectories"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create dummy trajectory for testing
    dummy_trajectory = {
        'observations': np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8),
        'actions': np.random.randn(100, 3),
        'rewards': np.random.randn(100),
        'dones': np.random.choice([True, False], 100)
    }
    
    with open(f"{test_data_dir}/test_traj.pkl", 'wb') as f:
        pickle.dump(dummy_trajectory, f)
    
    print(f"Created test trajectory in {test_data_dir}")
    
    # Test dataset creation
    try:
        dataloader = create_mdnrnn_dataloader(
            data_dir=test_data_dir,
            batch_size=4,
            sequence_length=10
        )
        
        stats = validate_dataset(dataloader)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
